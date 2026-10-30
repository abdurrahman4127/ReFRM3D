from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from refrm3d.data import BraTSDataset, discover_cases
from refrm3d.models import ReFRM3D
from refrm3d.radiomics.features import extract_radiomics
from refrm3d.training.engine import run_epoch
from refrm3d.training.losses import ReFRM3DLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw"),
        help="root to search recursively for labeled MRI cases",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--output-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument("--disable-radiomics", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_radiomic_normalization(
    model: ReFRM3D,
    loader: DataLoader,
    device: torch.device,
) -> None:
    """set feature statistics from coarse predictions on the training split."""
    previous_mode = model.training
    radiomics_enabled = model.use_radiomics
    model.use_radiomics = False
    model.eval()
    feature_batches = []

    try:
        with torch.inference_mode():
            for batch in loader:
                image = batch["image"].to(device, non_blocking=True)
                output = model(image)
                features = extract_radiomics(
                    output["coarse_probabilities"],
                    image,
                )
                feature_batches.append(features.cpu())
    finally:
        model.use_radiomics = radiomics_enabled
        model.train(previous_mode)

    if not feature_batches:
        raise ValueError("Cannot normalize radiomic features from an empty split")

    training_features = torch.cat(feature_batches, dim=0)
    feature_mean = training_features.mean(dim=0)
    feature_std = training_features.std(dim=0, unbiased=False)
    model.set_radiomic_normalization(feature_mean, feature_std)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must be between zero and one")

    # collect labeled cases across the complete dataset root before splitting.
    cases = discover_cases(args.data_root, require_labels=True)
    shuffled = list(cases)
    random.Random(args.seed).shuffle(shuffled)

    validation_count = max(1, round(len(shuffled) * args.validation_fraction))
    validation_cases = shuffled[:validation_count]
    training_cases = shuffled[validation_count:]
    if not training_cases:
        raise ValueError(
            "at least two labeled cases are required for a train/validation split"
        )

    cache_dir = args.data_root.parent / "processed" / "refrm3d"

    training_set = BraTSDataset(
        args.data_root,
        training_cases,
        output_size=args.output_size,
        cache_dir=cache_dir,
        select_tumor_slices=True,
    )
    validation_set = BraTSDataset(
        args.data_root,
        validation_cases,
        output_size=args.output_size,
        cache_dir=cache_dir,
    )

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.device.startswith("cuda"),
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        training_set,
        shuffle=True,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_set,
        shuffle=False,
        **loader_options,
    )

    # keep model, objective, and optimizer on the selected compute device.
    device = torch.device(args.device)
    model = ReFRM3D(use_radiomics=not args.disable_radiomics).to(device)
    criterion = ReFRM3DLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    radiomics_ready = not model.use_radiomics
    if model.use_radiomics:
        # train the coarse predictor before measuring its feature distribution.
        model.use_radiomics = False

    checkpoint_dir = args.output_dir / "checkpoints"
    log_dir = args.output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "metrics.jsonl"
    best_dice = float("-inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )

        validation_metrics = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
        )

        if not radiomics_ready:
            fit_radiomic_normalization(model, train_loader, device)
            model.use_radiomics = True
            radiomics_ready = True

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")

        # persist the latest epoch and retain the best validation checkpoint.
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "validation": validation_metrics,
            "seed": args.seed,
            "training_case_ids": [case.case_id for case in training_cases],
            "validation_case_ids": [case.case_id for case in validation_cases],
        }
        torch.save(checkpoint, checkpoint_dir / "last.pt")
        score = validation_metrics["dice_mean"]

        if score > best_dice:
            best_dice = score
            torch.save(checkpoint, checkpoint_dir / "best.pt")

        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_metrics['loss_total']:.4f} "
            f"val_loss={validation_metrics['loss_total']:.4f} "
            f"val_dice={score:.4f} "
            f"val_accuracy={validation_metrics['accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
