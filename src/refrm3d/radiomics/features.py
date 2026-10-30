"""non-differentiable first-order and texture descriptors for conditioning."""

from __future__ import annotations

import numpy as np
import torch


FEATURES_PER_REGION_MODALITY = 20


def _region_features(
    image: np.ndarray,
    mask: np.ndarray,
    levels: int = 16,
) -> np.ndarray:
    values = image[mask]

    if values.size == 0:
        return np.zeros(FEATURES_PER_REGION_MODALITY, dtype=np.float32)

    values = values.astype(np.float32, copy=False)
    mean = float(values.mean())
    std = float(values.std())

    centered = values - mean
    skew = float(np.mean(centered**3) / (std**3 + 1e-6))
    kurtosis = float(np.mean(centered**4) / (std**4 + 1e-6))

    quantiles = np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9]).astype(np.float32)

    hist, _ = np.histogram(values, bins=levels, density=False)
    probabilities = hist.astype(np.float64) / max(int(hist.sum()), 1)
    entropy = float(-np.sum(probabilities * np.log2(probabilities + 1e-12)))
    energy = float(np.mean(values**2))

    # estimate glcm descriptors from one-voxel offsets along each axis.
    # the same quantization range is used for every direction so the
    # directional measurements remain comparable.
    low = float(np.quantile(values, 0.01))
    high = float(np.quantile(values, 0.99))

    if high <= low:
        quantized = np.zeros(image.shape, dtype=np.uint8)
    else:
        quantized = np.clip(
            (image - low) / (high - low) * (levels - 1),
            0,
            levels - 1,
        )
        quantized = quantized.astype(np.uint8)

    contrasts, dissimilarities, homogeneities, asm_values = [], [], [], []
    max_run_length = sum(image.shape)
    run_level = np.zeros((levels, max_run_length), dtype=np.float64)

    for axis in range(image.ndim):
        left = [slice(None)] * image.ndim
        right = [slice(None)] * image.ndim
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        pair_mask = mask[tuple(left)] & mask[tuple(right)]

        if not np.any(pair_mask):
            continue
        a = quantized[tuple(left)][pair_mask].astype(np.float32)
        b = quantized[tuple(right)][pair_mask].astype(np.float32)
        delta = np.abs(a - b)

        contrasts.append(float(np.mean(delta**2)))
        dissimilarities.append(float(np.mean(delta)))
        homogeneities.append(float(np.mean(1.0 / (1.0 + delta**2))))
        pair_codes = a.astype(np.int32) * levels + b.astype(np.int32)
        pair_counts = np.bincount(
            pair_codes,
            minlength=levels * levels,
        ).astype(np.float64)
        probs = pair_counts[pair_counts > 0] / pair_counts.sum()
        asm_values.append(float(np.sum(probs**2)))

        # traverse each one-dimensional line to accumulate gray-level runs.
        axis_values = np.moveaxis(quantized, axis, -1).reshape(-1, image.shape[axis])
        axis_mask = np.moveaxis(mask, axis, -1).reshape(-1, image.shape[axis])
        positions = np.arange(axis_values.shape[1])[None, :]
        same_as_previous = np.zeros_like(axis_mask)
        same_as_previous[:, 1:] = (
            axis_mask[:, 1:]
            & axis_mask[:, :-1]
            & (axis_values[:, 1:] == axis_values[:, :-1])
        )
        same_as_next = np.zeros_like(axis_mask)
        same_as_next[:, :-1] = (
            axis_mask[:, :-1]
            & axis_mask[:, 1:]
            & (axis_values[:, :-1] == axis_values[:, 1:])
        )
        run_starts = axis_mask & ~same_as_previous
        run_ends = axis_mask & ~same_as_next
        end_positions = np.minimum.accumulate(
            np.where(run_ends, positions, axis_values.shape[1])[:, ::-1],
            axis=1,
        )[:, ::-1]
        line_indices, start_positions = np.nonzero(run_starts)
        stop_positions = end_positions[line_indices, start_positions]
        run_lengths = stop_positions - start_positions + 1
        run_gray_levels = axis_values[line_indices, start_positions].astype(
            np.int64
        )
        run_codes = (
            run_gray_levels * max_run_length
            + run_lengths.astype(np.int64)
            - 1
        )
        run_level += np.bincount(
            run_codes,
            minlength=levels * max_run_length,
        ).reshape(levels, max_run_length)

    texture = [
        float(np.mean(contrasts)) if contrasts else 0.0,
        float(np.mean(dissimilarities)) if dissimilarities else 0.0,
        float(np.mean(homogeneities)) if homogeneities else 0.0,
        float(np.mean(asm_values)) if asm_values else 0.0,
    ]
    run_count = float(run_level.sum())

    if run_count > 0:
        run_lengths = np.arange(1, run_level.shape[1] + 1, dtype=np.float64)
        gray_counts = run_level.sum(axis=1)
        length_counts = run_level.sum(axis=0)

        glrlm = [
            float(np.sum(run_level / (run_lengths[None, :] ** 2)) / run_count),
            float(np.sum(run_level * (run_lengths[None, :] ** 2)) / run_count),
            float(np.sum(gray_counts**2) / run_count),
            float(np.sum(length_counts**2) / run_count),
            float(run_count / max(mask.sum(), 1)),
        ]
    else:
        glrlm = [0.0] * 5
    return np.asarray(
        [
            mean,
            std,
            skew,
            kurtosis,
            energy,
            entropy,
            *quantiles.tolist(),
            *texture,
            *glrlm,
        ],
        dtype=np.float32,
    )


def extract_radiomics(
    coarse_probabilities: torch.Tensor,
    image: torch.Tensor,
) -> torch.Tensor:
    """extract descriptors from coarse whole-tumor, core, and enhancing masks.

    args:
        coarse_probabilities: probabilities shaped ``(B, 3, D, H, W)``.
        image: mri input shaped ``(B, 3, D, H, W)``.

    returns:
        detached tensor of shape ``(B, 3 * 3 * 20)`` on the input device.
    """
    if coarse_probabilities.ndim != 5 or image.ndim != 5:
        raise ValueError(
            "expected coarse probabilities and images with shape (B,C,D,H,W)"
        )
    if coarse_probabilities.shape[0] != image.shape[0]:
        raise ValueError("Image and coarse-mask batch sizes must match")
    if coarse_probabilities.shape[1] != 3 or image.shape[1] != 3:
        raise ValueError("Expected three WT/TC/ET masks and three MRI modalities")

    predictions = coarse_probabilities.detach().float().cpu().numpy()
    volumes = image.detach().float().cpu().numpy()
    batch_features = []

    for sample_masks, sample_image in zip(predictions, volumes):
        modality_features = []

        for region in sample_masks:
            mask = region >= 0.5

            for modality in sample_image:
                modality_features.append(_region_features(modality, mask))

        batch_features.append(np.concatenate(modality_features))
    return torch.as_tensor(
        np.stack(batch_features),
        device=image.device,
        dtype=image.dtype,
    )
