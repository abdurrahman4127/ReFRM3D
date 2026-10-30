"""class-balanced segmentation and coarse-mask objectives."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def labels_to_regions(labels: torch.Tensor) -> torch.Tensor:
    """convert exclusive labels 0/1/2/3 to overlapping wt/tc/et masks."""
    return torch.stack(
        (
            (labels > 0),
            (labels == 1) | (labels == 3),
            labels == 3,
        ),
        dim=1,
    ).float()


class DiceFocalLoss(nn.Module):
    """combine weighted multiclass soft dice and categorical focal loss."""

    def __init__(
        self,
        class_weights: tuple[float, ...] = (0.1, 1.0, 1.0, 1.0),
        gamma: float = 2.0,
        include_background: bool = False,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32),
        )

        self.gamma = gamma
        self.include_background = include_background
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if logits.ndim != 5 or target.ndim != 4:
            raise ValueError("Expected logits (B,C,D,H,W) and target (B,D,H,W)")

        if logits.shape[0] != target.shape[0] or logits.shape[2:] != target.shape[1:]:
            raise ValueError("Logit and target spatial shapes must match")

        classes = logits.shape[1]

        if self.class_weights.numel() != classes:
            raise ValueError(
                f"expected {classes} class weights, "
                f"received {self.class_weights.numel()}"
            )

        probabilities = logits.softmax(dim=1)
        one_hot = F.one_hot(
            target.long(),
            num_classes=classes,
        ).movedim(-1, 1).to(probabilities.dtype)

        reduce_dims = (0, 2, 3, 4)
        intersection = (probabilities * one_hot).sum(dim=reduce_dims)
        denominator = (
            probabilities.sum(dim=reduce_dims)
            + one_hot.sum(dim=reduce_dims)
        )
        dice_per_class = 1.0 - (2.0 * intersection + self.smooth) / (
            denominator + self.smooth
        )

        start = 0 if self.include_background else 1
        weights = self.class_weights[start:]
        weights = weights / weights.sum().clamp_min(self.smooth)

        dice_loss = (dice_per_class[start:] * weights).sum()

        # focal loss emphasizes voxels with uncertain or incorrect class
        # predictions while the dice term balances overlap across classes.
        log_probabilities = F.log_softmax(logits, dim=1)
        target_log_probability = log_probabilities.gather(
            1,
            target.long().unsqueeze(1),
        ).squeeze(1)
        target_probability = target_log_probability.exp()
        alpha = self.class_weights.to(logits.device)[target.long()]
        focal = (
            -alpha
            * (1.0 - target_probability).pow(self.gamma)
            * target_log_probability
        )

        focal_loss = focal.mean()
        return dice_loss + focal_loss, {"dice": dice_loss, "focal": focal_loss}


def coarse_region_dice_loss(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """compute soft dice loss for coarse wt/tc/et predictions."""
    if probabilities.shape[1] != 3:
        raise ValueError("Coarse prediction must contain WT, TC, and ET channels")

    target_regions = labels_to_regions(target).to(probabilities.dtype)

    if target_regions.shape[2:] != probabilities.shape[2:]:
        target_regions = F.interpolate(
            target_regions,
            size=probabilities.shape[2:],
            mode="nearest",
        )

    reduce_dims = (0, 2, 3, 4)
    intersection = (probabilities * target_regions).sum(dim=reduce_dims)
    denominator = probabilities.sum(dim=reduce_dims) + target_regions.sum(
        dim=reduce_dims
    )
    return 1.0 - ((2.0 * intersection + smooth) / (denominator + smooth)).mean()


class ReFRM3DLoss(nn.Module):
    """combine segmentation, coarse-region, and radiomic classification losses."""

    def __init__(
        self,
        class_weights: tuple[float, ...] = (0.1, 1.0, 1.0, 1.0),
        focal_gamma: float = 2.0,
        coarse_weight: float = 0.2,
        classification_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.final_loss = DiceFocalLoss(class_weights, focal_gamma)
        self.coarse_weight = coarse_weight
        self.classification_weight = classification_weight

    def forward(
        self,
        output: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        final, components = self.final_loss(output["logits"], target)

        coarse = coarse_region_dice_loss(output["coarse_probabilities"], target)
        total = final + self.coarse_weight * coarse
        components = {**components, "coarse": coarse, "total": total}

        classification_logits = output.get("classification_logits")
        if classification_logits is not None:
            classification_target = labels_to_regions(target)
            classification = F.binary_cross_entropy_with_logits(
                classification_logits,
                classification_target,
            )
            components["classification"] = classification
            total = total + self.classification_weight * classification

        components["total"] = total
        return total, components
