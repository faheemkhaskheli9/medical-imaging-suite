"""3D segmentation for breast MRI -- reuses ``ct_segmentation``'s UNet3D
architecture (same shape of problem: binary voxel segmentation), trained on
a synthetic spherical-tumor generator ported from this repo's original
``dataset_download.py::_materialize_synthetic``. That original repo only
reached Phase 1 (dataset acquisition + preprocessing, no model) -- this is
new code to make the feature runnable end-to-end from the UI, kept small and
CPU-friendly and validated only on synthetic data, same as
``ct_segmentation``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ct_segmentation.model import CHANNELS, DiceBCELoss, UNet3D

VOLUME_SHAPE = (32, 32, 24)
CHECKPOINT_NAME = "breast_tumor_unet3d.pt"


def _sphere_mask(shape: tuple[int, int, int], center, radius: float) -> np.ndarray:
    grids = np.ogrid[tuple(slice(0, s) for s in shape)]
    dist2 = sum((g - c) ** 2 for g, c in zip(grids, center))
    return (dist2 <= radius**2).astype(np.uint8)


def generate_volume_and_mask(shape=VOLUME_SHAPE, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic (MRI-like volume, tumor mask): Gaussian tissue background
    plus one bright sphere, normalized to [0, 1]."""
    rng = np.random.default_rng(seed)
    base = rng.normal(120.0, 15.0, size=shape)
    center = [rng.integers(s // 4, 3 * s // 4) for s in shape]
    radius = float(rng.uniform(min(shape) * 0.12, min(shape) * 0.22))
    mask = _sphere_mask(shape, center, radius)
    image = base + mask * rng.uniform(60.0, 110.0)
    image = np.clip(image, 0, None)
    normalized = (image / max(image.max(), 1e-6)).astype(np.float32)
    return normalized, mask


class SyntheticTumorDataset(Dataset):
    def __init__(self, num_samples: int, shape=VOLUME_SHAPE, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.samples = [
            generate_volume_and_mask(shape=shape, seed=int(rng.integers(0, 2**31 - 1))) for _ in range(num_samples)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        volume, mask = self.samples[idx]
        return (
            torch.from_numpy(volume).unsqueeze(0),
            torch.from_numpy(mask.astype(np.float32)).unsqueeze(0),
        )


def _atomic_save(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(model.state_dict(), tmp)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def train_on_synthetic_data(epochs: int = 5, device: str = "cpu") -> UNet3D:
    model = UNet3D(channels=CHANNELS).to(device)
    loader = DataLoader(SyntheticTumorDataset(40, seed=0), batch_size=4, shuffle=True)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for volumes, masks in loader:
            volumes, masks = volumes.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(volumes), masks)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def get_or_train_model(checkpoint_dir: Path, device: str = "cpu", epochs: int = 5) -> UNet3D:
    checkpoint_path = Path(checkpoint_dir) / CHECKPOINT_NAME
    if checkpoint_path.exists():
        model = UNet3D(channels=CHANNELS).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        return model
    model = train_on_synthetic_data(epochs=epochs, device=device)
    _atomic_save(model, checkpoint_path)
    return model


def predict_mask(model: UNet3D, volume: np.ndarray, device: str = "cpu", threshold: float = 0.5) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        probs = torch.sigmoid(model(tensor))
        return (probs > threshold).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
