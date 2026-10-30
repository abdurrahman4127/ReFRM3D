from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader

from refrm3d.data import BraTSDataset, discover_cases
from refrm3d.evaluation.metrics import (
    classification_metrics,
    dice_per_case,
)
from refrm3d.models import ReFRM3D
from refrm3d.preprocessing import restore_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw"),
        help="root to search recursively for MRI cases",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/predictions"),
    )

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-size", type=int, default=128)

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument(
        "--unlabeled",
        action="store_true",
        help="evaluate unlabeled cases without dice scores",
    )
    parser.add_argument(
        "--split",
        choices=("training", "validation", "all"),
        default="validation",
        help="select the saved training or validation split",
    )
    parser.add_argument("--disable-radiomics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    discovered_cases = discover_cases(
        args.data_root,
        require_labels=not args.unlabeled,
    )

    if args.unlabeled or args.split == "all":
        cases = discovered_cases
    else:
        split_key = f"{args.split}_case_ids"
        case_ids = checkpoint.get(split_key)
        if case_ids is None:
            raise ValueError(
                f"checkpoint does not contain the '{args.split}' case split; "
                "use --split all to evaluate every discovered case"
            )
        selected_ids = set(case_ids)
        cases = [case for case in discovered_cases if case.case_id in selected_ids]
        if not cases:
            raise ValueError(
                f"no {args.split} cases from the checkpoint were found under "
                f"{args.data_root}"
            )

    # retain preprocessing metadata so predictions can be mapped back to each
    # original nifti grid before writing the output volumes.
    dataset = BraTSDataset(
        args.data_root,
        cases,
        output_size=args.output_size,
        cache_dir=args.data_root.parent / "processed" / "refrm3d",
        return_metadata=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = ReFRM3D(use_radiomics=not args.disable_radiomics).to(device)
    model.load_state_dict(state)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases_by_id = {case.case_id: case for case in cases}

    rows = []
    classification_logits = []
    classification_targets = []

    with torch.inference_mode():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            output = model(image)
            predictions = (
                output["logits"]
                .argmax(dim=1)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            batch_classification_logits = output["classification_logits"]
            if batch_classification_logits is not None and "label" in batch:
                classification_logits.append(
                    batch_classification_logits.detach().cpu()
                )
                classification_targets.append(batch["label"].cpu())

            batch_scores = None
            if "label" in batch:
                batch_scores = dice_per_case(
                    torch.as_tensor(predictions),
                    batch["label"],
                )

            for index, case_id in enumerate(batch["case_id"]):
                native = restore_prediction(
                    predictions[index],
                    batch["original_shape"][index].numpy(),
                    batch["crop_shape"][index].numpy(),
                    batch["crop_origin"][index].numpy(),
                    batch["crop_basis"][index].numpy(),
                )
                brats_labels = np.zeros_like(native, dtype=np.uint8)
                brats_labels[native == 1] = 1
                brats_labels[native == 2] = 2
                brats_labels[native == 3] = 4

                case = cases_by_id[case_id]
                reference = nib.load(str(case.modalities["flair"]))
                prediction_image = nib.Nifti1Image(
                    brats_labels,
                    reference.affine,
                    reference.header,
                )
                nib.save(
                    prediction_image,
                    args.output_dir / f"{case_id}_seg.nii.gz",
                )

                row = {"case_id": case_id}

                if batch_classification_logits is not None:
                    probabilities = (
                        batch_classification_logits[index]
                        .sigmoid()
                        .cpu()
                        .numpy()
                    )
                    for region_index, region_name in enumerate(("WT", "TC", "ET")):
                        row[f"{region_name}_mean_probability"] = float(
                            probabilities[region_index].mean()
                        )

                if batch_scores is not None:
                    row.update(
                        {
                            name: float(values[index])
                            for name, values in batch_scores.items()
                        }
                    )
                rows.append(row)

    fieldnames = ["case_id"]
    if rows and "WT" in rows[0]:
        fieldnames.extend(("WT", "TC", "ET"))
    if rows and "WT_mean_probability" in rows[0]:
        fieldnames.extend(
            (
                "WT_mean_probability",
                "TC_mean_probability",
                "ET_mean_probability",
            )
        )

    if rows and "WT" in rows[0]:
        metrics_path = args.output_dir / "metrics.csv"
        with metrics_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    elif rows:
        predictions_path = args.output_dir / "case_predictions.csv"
        with predictions_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    if classification_logits and classification_targets:
        scores = classification_metrics(
            torch.cat(classification_logits),
            torch.cat(classification_targets),
        )
        metrics_path = args.output_dir / "classification_metrics.csv"
        with metrics_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("metric", "value"))
            writer.writeheader()
            writer.writerows(
                {"metric": name, "value": float(value)}
                for name, value in scores.items()
            )


if __name__ == "__main__":
    main()
