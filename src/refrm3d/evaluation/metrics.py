"""brats whole-tumor, tumor-core, and enhancing-tumor metrics."""

from __future__ import annotations

import torch


def region_masks(labels: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            labels > 0,
            (labels == 1) | (labels == 3),
            labels == 3,
        ),
        dim=1,
    )


def dice_per_case(
    prediction: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1e-5,
) -> dict[str, torch.Tensor]:
    """compute per-case dice scores for wt, tc, and et from exclusive labels."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must both have shape (B,D,H,W)")

    predicted_regions = region_masks(prediction.long())
    target_regions = region_masks(target.long())
    dims = (2, 3, 4)

    intersection = (predicted_regions & target_regions).sum(dim=dims).float()
    denominator = predicted_regions.sum(dim=dims) + target_regions.sum(dim=dims)
    scores = (2.0 * intersection + smooth) / (denominator.float() + smooth)
    return {name: scores[:, index] for index, name in enumerate(("WT", "TC", "ET"))}


@torch.no_grad()
def batch_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    predicted_labels = logits.argmax(dim=1)
    scores = dice_per_case(predicted_labels, target)

    scores["mean"] = torch.stack(tuple(scores.values()), dim=0).mean(dim=0)
    scores["accuracy"] = (predicted_labels == target).flatten(1).float().mean(dim=1)
    return scores


@torch.no_grad()
def classification_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """measure voxel-wise wt, tc, and et predictions."""
    actual_regions = torch.stack(
        (
            target > 0,
            (target == 1) | (target == 3),
            target == 3,
        ),
        dim=1,
    )
    predicted = logits >= 0
    actual = actual_regions.bool()
    reduce_dims = (0, 2, 3, 4)
    true_positive = (predicted & actual).sum(dim=reduce_dims).float()
    true_negative = (~predicted & ~actual).sum(dim=reduce_dims).float()
    false_positive = (predicted & ~actual).sum(dim=reduce_dims).float()
    false_negative = (~predicted & actual).sum(dim=reduce_dims).float()
    denominator = max(actual.numel() // actual.shape[1], 1)
    names = ("WT", "TC", "ET")
    scores = {}

    for index, name in enumerate(names):
        scores[f"classification_accuracy_{name}"] = (
            true_positive[index] + true_negative[index]
        ) / denominator
        scores[f"classification_precision_{name}"] = true_positive[index] / (
            true_positive[index] + false_positive[index]
        ).clamp_min(1.0)
        scores[f"classification_recall_{name}"] = true_positive[index] / (
            true_positive[index] + false_negative[index]
        ).clamp_min(1.0)
        scores[f"classification_specificity_{name}"] = true_negative[index] / (
            true_negative[index] + false_positive[index]
        ).clamp_min(1.0)

    return scores
