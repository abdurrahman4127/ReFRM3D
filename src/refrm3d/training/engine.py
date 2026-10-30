"""training and validation epoch routines."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch import nn

from refrm3d.evaluation.metrics import (
    batch_metrics,
    classification_metrics,
)


def _move_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if "label" not in batch:
        raise KeyError("Segmentation training requires a 'label' tensor in every batch")

    image = batch["image"].to(device, non_blocking=True)
    label = batch["label"].to(device, non_blocking=True)
    return image, label


def run_epoch(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """run one training epoch or one validation epoch."""
    training = optimizer is not None
    model.train(training)

    totals: defaultdict[str, float] = defaultdict(float)
    case_count = 0

    for batch in loader:
        image, target = _move_batch(batch, device)

        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(image)
            loss, components = criterion(output, target)

            if training:
                loss.backward()
                optimizer.step()
        batch_size = image.shape[0]
        case_count += batch_size

        for name, value in components.items():
            totals[f"loss_{name}"] += float(value.detach()) * batch_size

        scores = batch_metrics(output["logits"].detach(), target)

        for name, value in scores.items():
            metric_name = "accuracy" if name == "accuracy" else f"dice_{name}"
            totals[metric_name] += float(value.sum())

        if output["classification_logits"] is not None:
            case_scores = classification_metrics(
                output["classification_logits"].detach(),
                target,
            )
            for name, value in case_scores.items():
                totals[name] += float(value) * batch_size
    if case_count == 0:
        raise ValueError("The data loader produced no batches")

    return {name: total / case_count for name, total in totals.items()}
