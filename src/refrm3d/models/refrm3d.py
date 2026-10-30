"""radiomics-enhanced fused residual multiparametric 3d network."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from refrm3d.models.layers import (
    ConvBlock3d,
    FusedMultiScaleFeatureFusion,
    HybridUpsamplingResidualIntegration,
    ResidualSkip3d,
)
from refrm3d.radiomics.features import (
    FEATURES_PER_REGION_MODALITY,
    extract_radiomics,
)


class ReFRM3D(nn.Module):
    """3d u-net with fmff, huri, learned residual skips, and radiomic film.

    inputs use flair, t1ce, and t2 channels. outputs are mutually exclusive
    background, ncr/net, edema, and et classes. the ``regions`` output contains
    whole-tumor, tumor-core, and enhancing-tumor probabilities. when radiomics
    are enabled, the output also includes a voxel-wise subregion classifier.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        channels: Sequence[int] = (8, 16, 32, 64, 128),
        radiomic_embedding_dim: int = 32,
        use_radiomics: bool = True,
    ) -> None:
        super().__init__()

        if len(channels) != 5:
            raise ValueError(
                "channels must define five resolutions from input to bottleneck"
            )

        self.num_classes = num_classes
        self.channels = tuple(channels)
        self.use_radiomics = use_radiomics
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        encoder_dropout = (0.1, 0.1, 0.2, 0.2, 0.3)
        encoders = []
        current_channels = in_channels

        for width, dropout in zip(self.channels, encoder_dropout):
            encoders.append(ConvBlock3d(current_channels, width, dropout=dropout))
            current_channels = width
        self.encoders = nn.ModuleList(encoders)

        # combine local bottleneck context with features from every deeper
        # encoder level before passing them to the decoder.
        self.context = nn.Sequential(
            nn.Conv3d(
                channels[-1],
                channels[-1],
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            nn.BatchNorm3d(channels[-1]),
            nn.ReLU(inplace=True),
        )
        self.fmff = FusedMultiScaleFeatureFusion(channels[1:], channels[-1])
        self.bottleneck_fuse = ConvBlock3d(channels[-1], channels[-1])

        decoder_pairs = list(zip(reversed(channels[1:]), reversed(channels[:-1])))
        decoder_dropout = (0.2, 0.2, 0.1, 0.1)
        self.decoders = nn.ModuleList(
            HybridUpsamplingResidualIntegration(
                in_channels,
                skip_channels,
                out_channels,
                dropout,
            )
            for (in_channels, skip_channels), out_channels, dropout in zip(
                decoder_pairs, reversed(channels[:-1]), decoder_dropout
            )
        )

        # decoder features at c3-c0 pair with encoder features e3-e0.
        # learned weights let the network balance skip information by scale.
        self.residual_skips = nn.ModuleList(
            ResidualSkip3d(width, width, channels[0])
            for width in reversed(channels[:-1])
        )

        self.skip_weights = nn.Parameter(torch.zeros(len(self.residual_skips)))

        self.coarse_head = nn.Conv3d(channels[2], 3, kernel_size=1)
        self.feature_count = 3 * 3 * FEATURES_PER_REGION_MODALITY

        if use_radiomics:
            self.radiomic_projection = nn.Linear(
                self.feature_count,
                radiomic_embedding_dim,
            )
            self.radiomic_scale = nn.Linear(radiomic_embedding_dim, channels[0])
            self.radiomic_shift = nn.Linear(radiomic_embedding_dim, channels[0])
            self.region_classifier = nn.Conv3d(
                channels[0] + radiomic_embedding_dim,
                3,
                kernel_size=1,
            )
            self.register_buffer("radiomic_mean", torch.zeros(self.feature_count))
            self.register_buffer("radiomic_std", torch.ones(self.feature_count))
            nn.init.zeros_(self.radiomic_scale.weight)
            nn.init.ones_(self.radiomic_scale.bias)
            nn.init.zeros_(self.radiomic_shift.weight)
            nn.init.zeros_(self.radiomic_shift.bias)

        self.output_head = nn.Conv3d(channels[0], num_classes, kernel_size=1)

    def set_radiomic_normalization(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> None:
        """set training-set feature statistics before radiomic projection."""
        if not hasattr(self, "radiomic_mean"):
            raise RuntimeError("Radiomics are disabled for this model")

        mean = torch.as_tensor(
            mean,
            device=self.radiomic_mean.device,
            dtype=self.radiomic_mean.dtype,
        )
        std = torch.as_tensor(
            std,
            device=self.radiomic_std.device,
            dtype=self.radiomic_std.dtype,
        )

        if (
            mean.shape != self.radiomic_mean.shape
            or std.shape != self.radiomic_std.shape
        ):
            raise ValueError(
                f"expected radiomic statistics with shape ({self.feature_count},)"
            )

        self.radiomic_mean.copy_(mean)
        self.radiomic_std.copy_(std.clamp_min(1e-6))

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError("Input must have shape (B,C,D,H,W)")

        original_size = x.shape[2:]
        encoder_features = []
        current = x

        for index, encoder in enumerate(self.encoders):
            if index:
                current = self.pool(current)
            current = encoder(current)
            encoder_features.append(current)

        # fuse bottleneck context before decoding from coarse to fine scales.
        bottleneck = encoder_features[-1]
        fused = self.fmff(encoder_features[1:])
        current = self.bottleneck_fuse(self.context(bottleneck) + fused)

        decoder_features = []

        for decoder, encoder_feature in zip(
            self.decoders, reversed(encoder_features[:-1])
        ):
            current = decoder(current, encoder_feature)
            decoder_features.append(current)

        # D3 (D_{L-1}) supplies the coarse whole-tumor, tumor-core, and
        # enhancing-tumor estimates used by the radiomic conditioning branch.
        coarse_logits = self.coarse_head(decoder_features[1])
        coarse_logits = F.interpolate(
            coarse_logits, size=original_size, mode="trilinear", align_corners=False
        )
        coarse_probabilities = coarse_logits.sigmoid()

        # combine residual skip features at the highest decoder resolution.
        skip_features = [
            skip(decoder_feature, encoder_feature)
            for skip, decoder_feature, encoder_feature in zip(
                self.residual_skips,
                decoder_features,
                reversed(encoder_features[:-1]),
            )
        ]

        target_size = skip_features[-1].shape[2:]
        weights = self.skip_weights.softmax(dim=0)
        aggregate = None

        for weight, feature in zip(weights, skip_features):
            if feature.shape[2:] != target_size:
                feature = F.interpolate(
                    feature,
                    size=target_size,
                    mode="trilinear",
                    align_corners=False,
                )
            weighted = weight * feature
            aggregate = weighted if aggregate is None else aggregate + weighted

        radiomic_embedding = None
        classification_logits = None

        if self.use_radiomics:
            radiomic_features = extract_radiomics(coarse_probabilities, x)
            radiomic_features = (
                radiomic_features - self.radiomic_mean
            ) / self.radiomic_std
            radiomic_embedding = self.radiomic_projection(radiomic_features)
            scale = self.radiomic_scale(radiomic_embedding)[..., None, None, None]
            shift = self.radiomic_shift(radiomic_embedding)[..., None, None, None]
            aggregate = scale * aggregate + shift
            embedding_map = radiomic_embedding[..., None, None, None].expand(
                -1,
                -1,
                *aggregate.shape[2:],
            )
            classification_features = torch.cat(
                (aggregate, embedding_map),
                dim=1,
            )
            classification_logits = self.region_classifier(classification_features)

        # return final class logits at the input resolution.
        logits = self.output_head(aggregate)
        if logits.shape[2:] != original_size:
            logits = F.interpolate(
                logits,
                size=original_size,
                mode="trilinear",
                align_corners=False,
            )

        class_probabilities = logits.softmax(dim=1)
        # derive overlapping brats regions from exclusive class probabilities.
        # these probabilities follow the standard nested region definitions.
        regions = torch.stack(
            (
                class_probabilities[:, 1:].sum(dim=1),
                class_probabilities[:, 1] + class_probabilities[:, 3],
                class_probabilities[:, 3],
            ),
            dim=1,
        )
        return {
            "logits": logits,
            "probabilities": class_probabilities,
            "coarse_logits": coarse_logits,
            "coarse_probabilities": coarse_probabilities,
            "regions": regions,
            "radiomic_embedding": radiomic_embedding,
            "classification_logits": classification_logits,
        }
