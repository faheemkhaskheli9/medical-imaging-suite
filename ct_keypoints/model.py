"""3D CT keypoint (landmark) detection via heatmap regression.

``3d-medical-keypoint-detection`` had implemented NIfTI/DICOM I/O and
spacing/HU preprocessing (ported to ``imaging_core``) but no model yet
(Phase 1 only). This module completes the loop: it reuses
``ct_segmentation``'s ``UNet3D`` architecture unchanged (heatmap regression
is the same "predict a same-shaped volume" shape of problem as binary
segmentation, just with an MSE loss against a Gaussian-blob target instead of
Dice/BCE against a binary mask) and a synthetic single-landmark generator, so
the feature is runnable end-to-end from the UI.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ct_segmentation.model import CHANNELS, UNet3D

VOLUME_SHAPE = (32, 32, 32)
CHECKPOINT_NAME = "ct_keypoints_heatmap3d.pt"
HEATMAP_SIGMA = 2.0


def _gaussian_heatmap(shape, center, sigma: float = HEATMAP_SIGMA) -> np.ndarray:
    zz, yy, xx = np.meshgrid(*(np.arange(s) for s in shape), indexing="ij")
    cz, cy, cx = center
    dist2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
    return np.exp(-dist2 / (2 * sigma**2)).astype(np.float32)


def generate_volume_and_heatmap(
    shape=VOLUME_SHAPE, noise_std: float = 0.05, background_level: float = 0.3, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """Synthetic (volume, heatmap target, landmark coordinate): noisy
    background with a single bright landmark point at a known location."""
    rng = np.random.default_rng(seed)
    margin = 4
    center = tuple(int(rng.integers(margin, s - margin)) for s in shape)
    volume = rng.normal(loc=background_level, scale=noise_std, size=shape).astype(np.float32)
    heatmap = _gaussian_heatmap(shape, center)
    volume = np.clip(volume + 0.5 * heatmap, 0.0, 1.0).astype(np.float32)
    return volume, heatmap, center


class SyntheticKeypointDataset(Dataset):
    def __init__(self, num_samples: int, shape=VOLUME_SHAPE, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.samples = [
            generate_volume_and_heatmap(shape=shape, seed=int(rng.integers(0, 2**31 - 1))) for _ in range(num_samples)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        volume, heatmap, _center = self.samples[idx]
        return (
            torch.from_numpy(volume).unsqueeze(0),
            torch.from_numpy(heatmap).unsqueeze(0),
        )


def _atomic_save(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(model.state_dict(), tmp)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def train_on_synthetic_data(epochs: int = 6, device: str = "cpu") -> UNet3D:
    model = UNet3D(channels=CHANNELS).to(device)
    loader = DataLoader(SyntheticKeypointDataset(40, seed=0), batch_size=4, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for volumes, heatmaps in loader:
            volumes, heatmaps = volumes.to(device), heatmaps.to(device)
            optimizer.zero_grad()
            loss = criterion(model(volumes), heatmaps)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def get_or_train_model(checkpoint_dir: Path, device: str = "cpu", epochs: int = 6) -> UNet3D:
    checkpoint_path = Path(checkpoint_dir) / CHECKPOINT_NAME
    if checkpoint_path.exists():
        model = UNet3D(channels=CHANNELS).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        return model
    model = train_on_synthetic_data(epochs=epochs, device=device)
    _atomic_save(model, checkpoint_path)
    return model


def predict_keypoint(model: UNet3D, volume: np.ndarray, device: str = "cpu") -> tuple[tuple[int, int, int], np.ndarray, float]:
    """Returns (predicted (z, y, x) coordinate, predicted heatmap, peak confidence)."""
    with torch.no_grad():
        tensor = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        heatmap = torch.sigmoid(model(tensor)).squeeze(0).squeeze(0).cpu().numpy()
    coord = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return (int(coord[0]), int(coord[1]), int(coord[2])), heatmap, float(heatmap.max())
