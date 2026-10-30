"""brats volume normalization, pca cropping, and fixed-size resampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.ndimage import affine_transform, zoom


@dataclass
class PreprocessedCase:
    image: np.ndarray
    label: np.ndarray | None
    brain_mask: np.ndarray
    original_shape: np.ndarray
    crop_shape: np.ndarray
    crop_origin: np.ndarray
    crop_basis: np.ndarray


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndimage.label(mask)

    if count == 0:
        return mask

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(sizes.argmax())


def _brain_mask_from_intensity(modalities: np.ndarray) -> np.ndarray:
    """combine 99th-percentile intensity masks across the selected mri channels."""
    threshold_masks = []

    for modality in modalities:
        finite_values = modality[np.isfinite(modality)]

        if finite_values.size == 0:
            continue

        threshold = np.percentile(finite_values, 99.0)
        threshold_masks.append(np.isfinite(modality) & (modality > threshold))

    if not threshold_masks:
        return np.zeros(modalities.shape[1:], dtype=bool)

    return np.any(np.stack(threshold_masks, axis=0), axis=0)


def _select_tumor_slices(
    image: np.ndarray,
    label: np.ndarray | None,
    brain_mask: np.ndarray,
    crop_origin: np.ndarray,
    crop_basis: np.ndarray,
    slice_axis: int = 2,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """trim a labeled volume to its first and last tumor-bearing slices."""
    if label is None or not np.any(label):
        return image, label, brain_mask, crop_origin

    other_axes = tuple(axis for axis in range(label.ndim) if axis != slice_axis)
    slice_has_tumor = np.any(label > 0, axis=other_axes)
    tumor_slices = np.flatnonzero(slice_has_tumor)

    if tumor_slices.size == 0:
        return image, label, brain_mask, crop_origin

    start = int(tumor_slices[0])
    stop = int(tumor_slices[-1]) + 1
    selection = [slice(None)] * label.ndim
    selection[slice_axis] = slice(start, stop)
    selection_tuple = tuple(selection)
    crop_origin = crop_origin + crop_basis[:, slice_axis] * start

    return (
        image[(slice(None),) + selection_tuple],
        label[selection_tuple],
        brain_mask[selection_tuple],
        crop_origin,
    )


def _pca_crop(
    volume: np.ndarray,
    mask: np.ndarray,
    label: np.ndarray | None,
    margin_std: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.argwhere(mask)

    if coordinates.shape[0] < 8:
        identity = np.eye(3, dtype=np.float64)
        return volume, label, mask, np.zeros(3), identity

    # principal components align the crop with the dominant axes of the
    # nonzero brain support while retaining a reversible mapping to the input.
    center = coordinates.mean(axis=0)
    covariance = np.cov(coordinates - center, rowvar=False, bias=True)
    eigenvalues, basis = np.linalg.eigh(covariance)

    basis = basis[:, np.argsort(eigenvalues)[::-1]]
    projected = (coordinates - center) @ basis
    ordered_eigenvalues = eigenvalues[np.argsort(eigenvalues)[::-1]]
    margin = margin_std * np.sqrt(np.maximum(ordered_eigenvalues, 0.0))
    lower = projected.min(axis=0) - margin
    upper = projected.max(axis=0) + margin
    output_shape = np.maximum(np.ceil(upper - lower).astype(int) + 1, 1)
    offset = center + basis @ lower

    # resample each modality and its masks into the same pca-aligned crop.
    cropped_image = np.stack(
        [
            affine_transform(
                modality,
                matrix=basis,
                offset=offset,
                output_shape=tuple(output_shape),
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
            for modality in volume
        ],
        axis=0,
    )
    cropped_label = None
    if label is not None:
        cropped_label = affine_transform(
            label,
            matrix=basis,
            offset=offset,
            output_shape=tuple(output_shape),
            order=0,
            mode="constant",
            cval=0,
            prefilter=False,
        )
    cropped_mask = affine_transform(
        mask.astype(np.uint8),
        matrix=basis,
        offset=offset,
        output_shape=tuple(output_shape),
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ).astype(bool)
    return cropped_image, cropped_label, cropped_mask, offset, basis


def _resize(
    array: np.ndarray,
    target_shape: tuple[int, int, int],
    order: int,
) -> np.ndarray:
    factors = np.asarray(target_shape, dtype=np.float64) / np.asarray(array.shape[-3:])

    if array.ndim == 4:
        factors = np.concatenate(([1.0], factors))
    return zoom(array, factors, order=order, mode="nearest", prefilter=order > 1)


def preprocess_case(
    modalities: np.ndarray,
    label: np.ndarray | None = None,
    output_size: int | tuple[int, int, int] = 128,
    crop_margin_std: float = 1.5,
    select_tumor_slices: bool = False,
) -> PreprocessedCase:
    """preprocess flair, t1ce, and t2 arrays to model-ready ``(C,D,H,W)``.

    the largest component of the combined 99th-percentile intensity masks
    defines the pca crop and normalization mask. raw segmentation labels
    0/1/2/4 map to exclusive training labels 0/1/2/3.
    """
    modalities = np.asarray(modalities, dtype=np.float32)

    if modalities.ndim != 4 or modalities.shape[0] != 3:
        raise ValueError("modalities must have shape (3, X, Y, Z): FLAIR, T1ce, T2")

    if len({tuple(m.shape) for m in modalities}) != 1:
        raise ValueError("All MRI modalities must share a voxel grid")

    if label is not None and tuple(label.shape) != tuple(modalities.shape[1:]):
        raise ValueError("Segmentation and MRI volumes must share a voxel grid")

    brain_mask = _brain_mask_from_intensity(modalities)
    brain_mask = _largest_component(brain_mask)

    if brain_mask.sum() == 0:
        raise ValueError("Could not find nonzero brain voxels in the MRI modalities")

    exclusive_label = None

    if label is not None:
        raw = np.asarray(label)
        unknown_labels = np.setdiff1d(np.unique(raw), np.asarray([0, 1, 2, 4]))

        if unknown_labels.size:
            raise ValueError(
                f"unexpected brats segmentation labels: {unknown_labels.tolist()}"
            )
        exclusive_label = np.zeros(raw.shape, dtype=np.uint8)
        exclusive_label[raw == 1] = 1
        exclusive_label[raw == 2] = 2
        exclusive_label[raw == 4] = 3

    cropped = _pca_crop(
        modalities,
        brain_mask,
        exclusive_label,
        crop_margin_std,
    )
    cropped_image, cropped_label, cropped_mask, crop_origin, crop_basis = cropped

    normalized_image = np.zeros_like(cropped_image, dtype=np.float32)

    for index, modality in enumerate(cropped_image):
        valid_voxels = cropped_mask & np.isfinite(modality)
        values = modality[valid_voxels]
        mean = float(values.mean())
        std = float(values.std())

        if std < 1e-6:
            std = 1.0

        normalized_image[index, valid_voxels] = (
            modality[valid_voxels] - mean
        ) / std

    cropped_image = normalized_image

    if select_tumor_slices:
        (
            cropped_image,
            cropped_label,
            cropped_mask,
            crop_origin,
        ) = _select_tumor_slices(
            cropped_image,
            cropped_label,
            cropped_mask,
            crop_origin,
            crop_basis,
        )

    target_shape = (
        (output_size,) * 3
        if isinstance(output_size, int)
        else tuple(output_size)
    )

    if len(target_shape) != 3:
        raise ValueError("output_size must be an int or a three-element shape")
    resized_image = _resize(cropped_image, target_shape, order=1).astype(np.float32)

    resized_mask = _resize(
        cropped_mask.astype(np.uint8),
        target_shape,
        order=0,
    ).astype(bool)
    resized_label = None

    if cropped_label is not None:
        resized_label = _resize(
            cropped_label,
            target_shape,
            order=0,
        ).astype(np.uint8)

    # keep crop geometry with each sample for native-grid prediction export.
    return PreprocessedCase(
        image=resized_image,
        label=resized_label,
        brain_mask=resized_mask,
        original_shape=np.asarray(modalities.shape[1:], dtype=np.int64),
        crop_shape=np.asarray(cropped_image.shape[1:], dtype=np.int64),
        crop_origin=np.asarray(crop_origin),
        crop_basis=np.asarray(crop_basis),
    )


def restore_prediction(
    prediction: np.ndarray,
    original_shape: tuple[int, int, int] | np.ndarray,
    crop_shape: tuple[int, int, int] | np.ndarray,
    crop_origin: np.ndarray,
    crop_basis: np.ndarray,
) -> np.ndarray:
    """map a resized label prediction to the original nifti array grid."""
    prediction = np.asarray(prediction)

    if prediction.ndim != 3:
        raise ValueError("prediction must be a 3D label volume")

    crop_shape = tuple(int(value) for value in crop_shape)
    original_shape = tuple(int(value) for value in original_shape)

    cropped_prediction = _resize(
        prediction,
        crop_shape,
        order=0,
    ).astype(prediction.dtype)

    basis = np.asarray(crop_basis, dtype=np.float64)
    origin = np.asarray(crop_origin, dtype=np.float64)
    return affine_transform(
        cropped_prediction,
        matrix=basis.T,
        offset=-basis.T @ origin,
        output_shape=original_shape,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ).astype(prediction.dtype)
