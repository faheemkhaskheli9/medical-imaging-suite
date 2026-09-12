"""Shared visualization: 3D orthogonal-slice PNGs and 2D overlay PNGs.

3D Slicer (the original tech-stack choice in several of the source repos) is
a desktop GUI app with no headless automation path, so every 3D task renders
axial/coronal/sagittal mid-slices via matplotlib's Agg backend instead --
still exportable as NIfTI for manual inspection in 3D Slicer.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, no GUI/display required

import matplotlib.pyplot as plt
import numpy as np


def _mid_slices(volume: np.ndarray) -> list[tuple[str, np.ndarray]]:
    d, h, w = volume.shape
    return [
        ("axial", volume[d // 2, :, :]),
        ("coronal", volume[:, h // 2, :]),
        ("sagittal", volume[:, :, w // 2]),
    ]


def save_orthogonal_slices(
    volume: np.ndarray,
    mask: np.ndarray | None = None,
    pred: np.ndarray | None = None,
    out_path: str | Path = "slices.png",
    title: str | None = None,
) -> Path:
    """Save axial/coronal/sagittal mid-slices of ``volume``, with ``mask``
    (ground truth, green) and/or ``pred`` (prediction, dashed red) contours
    overlaid where given."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vol_slices = _mid_slices(volume)
    mask_slices = dict(_mid_slices(mask)) if mask is not None else None
    pred_slices = dict(_mid_slices(pred)) if pred is not None else None

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (name, sl) in zip(axes, vol_slices):
        ax.imshow(sl, cmap="gray", vmin=0.0, vmax=1.0)
        if mask_slices is not None and mask_slices[name].max() > 0:
            ax.contour(mask_slices[name], levels=[0.5], colors="lime", linewidths=1.2)
        if pred_slices is not None and pred_slices[name].max() > 0:
            ax.contour(pred_slices[name], levels=[0.5], colors="red", linewidths=1.0, linestyles="dashed")
        ax.set_title(name)
        ax.axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def save_keypoint_slices(
    volume: np.ndarray,
    keypoints: np.ndarray | None = None,
    pred_keypoints: np.ndarray | None = None,
    out_path: str | Path = "keypoints.png",
    title: str | None = None,
) -> Path:
    """Like :func:`save_orthogonal_slices`, but marks point coordinates
    (``(z, y, x)`` rows) instead of contouring a mask -- ground truth as
    green circles, predictions as red x's, only on the slice each point
    actually falls on."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    d, h, w = volume.shape
    mid = {"axial": d // 2, "coronal": h // 2, "sagittal": w // 2}
    vol_slices = _mid_slices(volume)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (name, sl) in zip(axes, vol_slices):
        ax.imshow(sl, cmap="gray", vmin=0.0, vmax=1.0)
        for pts, marker, color, label in (
            (keypoints, "o", "lime", "ground truth"),
            (pred_keypoints, "x", "red", "prediction"),
        ):
            if pts is None or len(pts) == 0:
                continue
            for z, y, x in pts:
                if name == "axial" and round(z) == mid["axial"]:
                    ax.plot(x, y, marker, color=color, markersize=8, mew=2)
                elif name == "coronal" and round(y) == mid["coronal"]:
                    ax.plot(x, z, marker, color=color, markersize=8, mew=2)
                elif name == "sagittal" and round(x) == mid["sagittal"]:
                    ax.plot(y, z, marker, color=color, markersize=8, mew=2)
        ax.set_title(name)
        ax.axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def save_overlay_2d(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    pred: np.ndarray | None = None,
    boxes: list[tuple[float, float, float, float]] | None = None,
    out_path: str | Path = "overlay.png",
    title: str | None = None,
) -> Path:
    """Save a 2D image with an optional mask/prediction contour overlay
    and/or bounding boxes (``(x_min, y_min, x_max, y_max)`` in pixels)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
    if mask is not None and mask.max() > 0:
        ax.contour(mask, levels=[0.5], colors="lime", linewidths=1.5)
    if pred is not None and pred.max() > 0:
        ax.contour(pred, levels=[0.5], colors="red", linewidths=1.2, linestyles="dashed")
    if boxes:
        from matplotlib.patches import Rectangle

        for x_min, y_min, x_max, y_max in boxes:
            ax.add_patch(
                Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    fill=False,
                    edgecolor="red",
                    linewidth=1.5,
                )
            )
    ax.set_title(title or "")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
