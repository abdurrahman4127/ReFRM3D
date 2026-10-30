"""reusable 3d blocks for the refrm3d architecture."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock3d(nn.Module):
    """apply two padded convolutions with normalization and relu activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class FusedMultiScaleFeatureFusion(nn.Module):
    """align encoder channels and spatial sizes before summing features."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
    ) -> None:
        super().__init__()

        self.projections = nn.ModuleList(
            nn.Conv3d(
                channels,
                out_channels,
                kernel_size=1,
            )
            for channels in in_channels
        )

    def forward(
        self,
        features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if len(features) != len(self.projections):
            raise ValueError(
                f"expected {len(self.projections)} encoder features, "
                f"received {len(features)}"
            )

        target_size = features[-1].shape[2:]
        fused = None

        for feature, projection in zip(features, self.projections):
            aligned = projection(feature)

            if aligned.shape[2:] != target_size:
                aligned = F.adaptive_avg_pool3d(aligned, target_size)

            fused = aligned if fused is None else fused + aligned
        return fused


class HybridUpsamplingResidualIntegration(nn.Module):
    """upsample decoder features and refine their residual fusion with the encoder."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.upsample = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.skip_projection = (
            nn.Identity()
            if skip_channels == out_channels
            else nn.Conv3d(
                skip_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            )
        )
        self.refine = ConvBlock3d(out_channels, out_channels, dropout=dropout)

    def forward(
        self,
        decoder: torch.Tensor,
        encoder: torch.Tensor,
    ) -> torch.Tensor:
        upsampled = self.upsample(decoder)

        if upsampled.shape[2:] != encoder.shape[2:]:
            upsampled = F.interpolate(
                upsampled,
                size=encoder.shape[2:],
                mode="trilinear",
                align_corners=False,
            )

        return self.refine(upsampled + self.skip_projection(encoder))


class ResidualSkip3d(nn.Module):
    """filter encoder features before residual fusion at one resolution."""

    def __init__(
        self,
        decoder_channels: int,
        encoder_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.encoder_filter = nn.Conv3d(
            encoder_channels,
            decoder_channels,
            kernel_size=1,
        )
        self.refine = nn.Sequential(
            nn.Conv3d(
                decoder_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        decoder: torch.Tensor,
        encoder: torch.Tensor,
    ) -> torch.Tensor:
        encoder_features = self.encoder_filter(encoder)

        return self.refine(decoder + encoder_features)
