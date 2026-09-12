"""Shared 3D preprocessing: resample, normalize, crop/pad.

Merged from three near-duplicate implementations (``3d-ct-segmentation``'s
HU windowing, ``3d-medical-keypoint-detection``'s spacing resample,
``3d-breast-tumor-segmentation``'s zscore/minmax/percentile normalization) --
every feature app now calls the one copy below instead of maintaining its
own resample/normalize pair.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

_VALID_NORMALIZATIONS = ("hu_window", "zscore", "minmax", "percentile")


class PreprocessError(RuntimeError):
    """A volume could not be preprocessed, or the config is invalid."""


@dataclass
class PreprocessConfig:
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    crop_size: tuple[int, int, int] | None = None
    normalization: str = "hu_window"
    hu_window: tuple[float, float] = (-1000.0, 400.0)
    percentile_clip: tuple[float, float] = (0.5, 99.5)
    output_dtype: str = "float32"

    def __post_init__(self) -> None:
        self.target_spacing = tuple(float(x) for x in self.target_spacing)
        if any(s <= 0 for s in self.target_spacing) or len(self.target_spacing) != 3:
            raise PreprocessError(f"target_spacing must be 3 positive values, got {self.target_spacing}")
        if self.crop_size is not None:
            self.crop_size = tuple(int(x) for x in self.crop_size)
            if len(self.crop_size) != 3 or any(s < 1 for s in self.crop_size):
                raise PreprocessError(f"crop_size must be 3 positive ints, got {self.crop_size}")
        if self.normalization not in _VALID_NORMALIZATIONS:
            raise PreprocessError(f"normalization must be one of {_VALID_NORMALIZATIONS}, got {self.normalization!r}")


def resample_volume(
    array: np.ndarray,
    current_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
    *,
    order: int = 1,
) -> np.ndarray:
    """Resample ``array`` from ``current_spacing`` to ``target_spacing``.

    ``order=1`` (trilinear) for intensity data, ``order=0`` (nearest
    neighbor) for label/mask volumes so resampling never invents a
    fractional label.
    """
    current = np.asarray(current_spacing, dtype=np.float64)
    target = np.asarray(target_spacing, dtype=np.float64)
    if len(target) != 3 or np.any(target <= 0):
        raise PreprocessError(f"target_spacing must be 3 positive values, got {target_spacing}")
    zoom_factors = current / target
    return ndimage.zoom(array, zoom_factors, order=order)


def normalize_intensity(
    array: np.ndarray,
    method: str = "hu_window",
    *,
    hu_window: tuple[float, float] = (-1000.0, 400.0),
    percentile_clip: tuple[float, float] = (0.5, 99.5),
) -> np.ndarray:
    """Rescale intensities to a model-friendly range.

    - ``hu_window``: clip to a Hounsfield-unit window, then min-max to [0, 1]
      (standard CT preprocessing).
    - ``zscore``: (x - mean) / std over the whole volume.
    - ``minmax``: linearly rescaled to [0, 1] using the volume's own min/max.
    - ``percentile``: clip to ``percentile_clip`` percentiles, then min-max
      rescale to [0, 1] -- robust to a few extreme outlier voxels (typical
      for MRI, which has no fixed intensity scale like CT's HU).
    """
    array = array.astype(np.float64)
    if method == "hu_window":
        lo, hi = hu_window
        if lo >= hi:
            raise PreprocessError(f"hu_window must be (low, high) with low < high, got {hu_window}")
        clipped = np.clip(array, lo, hi)
        return ((clipped - lo) / (hi - lo)).astype(np.float32)
    if method == "zscore":
        std = array.std()
        return (array - array.mean()) / std if std > 1e-8 else array - array.mean()
    if method == "minmax":
        lo, hi = array.min(), array.max()
        return (array - lo) / (hi - lo) if hi > lo else np.zeros_like(array)
    if method == "percentile":
        lo_pct, hi_pct = percentile_clip
        lo, hi = np.percentile(array, [lo_pct, hi_pct])
        clipped = np.clip(array, lo, hi)
        return (clipped - lo) / (hi - lo) if hi > lo else np.zeros_like(array)
    raise PreprocessError(f"Unknown normalization method: {method!r}")


def crop_or_pad(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Center-crop or zero-pad a 3D array to exactly ``target_shape``."""
    out = volume
    for axis, target in enumerate(target_shape):
        current = out.shape[axis]
        if current == target:
            continue
        if current > target:
            start = (current - target) // 2
            slicer = [slice(None)] * out.ndim
            slicer[axis] = slice(start, start + target)
            out = out[tuple(slicer)]
        else:
            pad_total = target - current
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            pad_width = [(0, 0)] * out.ndim
            pad_width[axis] = (pad_before, pad_after)
            out = np.pad(out, pad_width, mode="constant", constant_values=0)
    return out


def resize_to_shape(array: np.ndarray, target_shape: tuple[int, int, int], order: int = 1) -> np.ndarray:
    """Zoom-resize a whole 3D array to exactly ``target_shape`` (unlike
    :func:`crop_or_pad`, this rescales the *entire* volume down/up rather
    than cropping around the center -- used by the small demo models below
    to fit a full scan into a fixed, CPU-friendly input size)."""
    factors = [t / s for t, s in zip(target_shape, array.shape)]
    resized = ndimage.zoom(array, factors, order=order)
    # Guard against off-by-one rounding from zoom().
    return crop_or_pad(resized, target_shape).astype(array.dtype)


def preprocess_volume(
    array: np.ndarray,
    spacing: tuple[float, float, float],
    config: PreprocessConfig | None = None,
) -> np.ndarray:
    """Resample to ``config.target_spacing``, normalize, then (if
    ``config.crop_size`` is set) center-crop/pad to it."""
    config = config or PreprocessConfig()
    resampled = resample_volume(array, spacing, config.target_spacing, order=1)
    normalized = normalize_intensity(
        resampled,
        config.normalization,
        hu_window=config.hu_window,
        percentile_clip=config.percentile_clip,
    ).astype(config.output_dtype)
    if config.crop_size is not None:
        normalized = crop_or_pad(normalized, config.crop_size)
    return normalized
