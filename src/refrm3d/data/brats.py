"""brats case discovery and pytorch dataset."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from refrm3d.preprocessing.brats import preprocess_case


MODALITIES = ("flair", "t1ce", "t2")
ALL_MODALITIES = ("flair", "t1", "t1ce", "t2")


@dataclass(frozen=True)
class BraTSCase:
    case_id: str
    directory: Path
    modalities: dict[str, Path]
    segmentation: Path | None


def _strip_nii_suffix(filename: str) -> str:
    return re.sub(r"\.nii(?:\.gz)?$", "", filename, flags=re.IGNORECASE)


def discover_cases(
    root: str | Path,
    require_labels: bool = True,
) -> list[BraTSCase]:

    root = Path(root).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"BraTS data directory does not exist: {root}")

    cases: list[BraTSCase] = []

    for flair_path in sorted(root.rglob("*_flair.nii*")):
        if not flair_path.is_file():
            continue

        prefix = _strip_nii_suffix(flair_path.name)
        if not prefix.lower().endswith("_flair"):
            continue

        case_id = prefix[:-6]
        modality_paths: dict[str, Path] = {"flair": flair_path}

        for modality in ALL_MODALITIES:
            if modality == "flair":
                continue
            modality_names = (modality,)

            if modality == "t1ce":
                modality_names = ("t1ce", "t1gd")

            candidates = [
                flair_path.with_name(f"{case_id}_{modality_name}{suffix}")
                for modality_name in modality_names
                for suffix in (".nii.gz", ".nii")
            ]
            match = next(
                (candidate for candidate in candidates if candidate.is_file()),
                None,
            )
            if match is not None:
                modality_paths[modality] = match

        if not all(key in modality_paths for key in MODALITIES):
            missing = sorted(set(MODALITIES) - modality_paths.keys())
            raise ValueError(
                f"case {case_id} is missing required modalities: {missing}"
            )

        segmentation = next(
            (
                path
                for path in (
                    flair_path.with_name(f"{case_id}_seg.nii.gz"),
                    flair_path.with_name(f"{case_id}_seg.nii"),
                )
                if path.is_file()
            ),
            None,
        )
        if segmentation is None:
            # one case names its mask with the subject id rather than the case id
            alternate_masks = sorted(
                path
                for path in flair_path.parent.glob("*.nii*")
                if path.is_file() and "seg" in path.name.lower()
            )
            if len(alternate_masks) == 1:
                segmentation = alternate_masks[0]
            elif len(alternate_masks) > 1:
                raise ValueError(
                    f"Case {case_id} has multiple possible segmentation files: "
                    f"{[path.name for path in alternate_masks]}"
                )

        if require_labels and segmentation is None:
            continue
        cases.append(
            BraTSCase(
                case_id=case_id,
                directory=flair_path.parent,
                modalities=modality_paths,
                segmentation=segmentation,
            )
        )
    if not cases:
        label_note = " with segmentation files" if require_labels else ""
        raise FileNotFoundError(f"No BraTS cases{label_note} found under {root}")
    return cases


def load_case(case: BraTSCase) -> tuple[np.ndarray, np.ndarray | None]:
    """load a case in flair, t1ce, t2 channel order."""
    volumes = []
    reference_shape = None

    for modality in MODALITIES:
        volume = np.asarray(
            nib.load(str(case.modalities[modality])).dataobj,
            dtype=np.float32,
        )
        if reference_shape is None:
            reference_shape = volume.shape
        elif volume.shape != reference_shape:
            raise ValueError(
                f"Case {case.case_id} has unregistered array shapes: "
                f"{modality} {volume.shape} != {reference_shape}"
            )

        volumes.append(volume)

    label = None
    if case.segmentation is not None:
        label = np.asarray(nib.load(str(case.segmentation)).dataobj, dtype=np.uint8)
        if label.shape != reference_shape:
            raise ValueError(
                f"Case {case.case_id} segmentation shape {label.shape} "
                f"does not match MRI shape {reference_shape}"
            )
    return np.stack(volumes, axis=0), label


class BraTSDataset(Dataset[dict[str, Any]]):
    """load nifti volumes on demand and optionally cache preprocessing."""

    def __init__(
        self,
        root: str | Path,
        cases: list[BraTSCase] | None = None,
        require_labels: bool = True,
        output_size: int = 128,
        cache_dir: str | Path | None = None,
        return_metadata: bool = False,
        select_tumor_slices: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()

        self.cases = (
            cases
            if cases is not None
            else discover_cases(self.root, require_labels)
        )

        self.output_size = output_size
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.select_tumor_slices = select_tumor_slices

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.cases)

    def _load_preprocessed(
        self,
        case: BraTSCase,
    ) -> tuple[np.ndarray, np.ndarray | None, dict[str, np.ndarray]]:
        cache_path = None
        if self.cache_dir is not None:
            cache_suffix = "_tumor_slices" if self.select_tumor_slices else ""
            cache_path = self.cache_dir / (
                f"{case.case_id}_{self.output_size}{cache_suffix}.npz"
            )

            if cache_path.exists():
                with np.load(cache_path, allow_pickle=False) as archive:
                    has_label = bool(archive["has_label"].item())
                    label = archive["label"] if has_label else None

                    metadata = {
                        "original_shape": archive["original_shape"],
                        "crop_shape": archive["crop_shape"],
                        "crop_origin": archive["crop_origin"],
                        "crop_basis": archive["crop_basis"],
                    }
                    return archive["image"], label, metadata

        modalities, label = load_case(case)

        processed = preprocess_case(
            modalities,
            label,
            output_size=self.output_size,
            select_tumor_slices=self.select_tumor_slices,
        )
        metadata = {
            "original_shape": processed.original_shape,
            "crop_shape": processed.crop_shape,
            "crop_origin": processed.crop_origin,
            "crop_basis": processed.crop_basis,
        }

        if cache_path is not None:
            np.savez_compressed(
                cache_path,
                image=processed.image,
                label=(
                    processed.label
                    if processed.label is not None
                    else np.empty((0,), dtype=np.uint8)
                ),
                has_label=np.asarray(processed.label is not None),
                original_shape=processed.original_shape,
                crop_shape=processed.crop_shape,
                crop_origin=processed.crop_origin,
                crop_basis=processed.crop_basis,
            )

        return processed.image, processed.label, metadata


    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        case = self.cases[index]
        image, label, metadata = self._load_preprocessed(case)
        sample: dict[str, Any] = {
            "image": torch.from_numpy(np.ascontiguousarray(image)).float(),
            "case_id": case.case_id,
        }

        if label is not None:
            sample["label"] = torch.from_numpy(
                np.ascontiguousarray(label)
            ).long()

        if self.return_metadata:
            sample.update(metadata)
        return sample
