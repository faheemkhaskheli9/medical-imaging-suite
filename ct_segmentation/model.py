"""3D U-Net for CT segmentation -- ported from ``3d-ct-segmentation``.

The model, loss, and synthetic-data generator below are near-verbatim ports
of that repo's ``src/models/unet3d.py`` + ``src/losses.py`` +
``src/data/synthetic.py`` (see its README section 11: on synthetic
ellipsoid-in-noise data it reaches val_dice = 0.976 in 5 epochs, CPU, in
seconds). No real public CT dataset was trained on in that repo or here --
``get_or_train_model`` trains this exact small architecture on synthetic
data the first time it's needed and caches the checkpoint, so the web demo
always has *some* trained model to run inference with, honestly labeled as
untested on real anatomy (see ``task.py``'s description and the result
summary).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

VOLUME_SHAPE = (32, 32, 32)
CHANNELS = (8, 16, 32, 64)
CHECKPOINT_NAME = "ct_segmentation_unet3d.pt"


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        diffs = [s - x_dim for s, x_dim in zip(skip.shape[2:], x.shape[2:])]
        if any(diffs):
            pad = []
            for d in reversed(diffs):
                pad.extend([d // 2, d - d // 2])
            x = nn.functional.pad(x, pad)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """Input (N, in_channels, D, H, W) -> output (N, out_channels, D, H, W) raw logits."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, channels=CHANNELS):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.in_conv = DoubleConv3D(in_channels, c1)
        self.down1 = Down3D(c1, c2)
        self.down2 = Down3D(c2, c3)
        self.down3 = Down3D(c3, c4)
        self.up1 = Up3D(c4, c3, c3)
        self.up2 = Up3D(c3, c2, c2)
        self.up3 = Up3D(c2, c1, c1)
        self.out_conv = nn.Conv3d(c1, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.out_conv(x)


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def _dice(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).flatten(1)
        target = target.flatten(1)
        intersection = (probs * target).sum(dim=1)
        union = probs.sum(dim=1) + target.sum(dim=1)
        return 1.0 - ((2 * intersection + self.eps) / (union + self.eps)).mean()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self._dice(logits, target) + self.bce_weight * self.bce(logits, target)


def dice_score(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float().flatten(1)
    target = (target > 0.5).float().flatten(1)
    intersection = (preds * target).sum(dim=1)
    union = preds.sum(dim=1) + target.sum(dim=1)
    both_empty = union == 0
    dice = (2 * intersection + eps) / (union + eps)
    return torch.where(both_empty, torch.ones_like(dice), dice).mean().item()


def generate_volume_and_mask(
    shape=VOLUME_SHAPE,
    num_lesions: int = 1,
    min_radius_frac: float = 0.12,
    max_radius_frac: float = 0.28,
    noise_std: float = 0.06,
    background_level: float = 0.35,
    lesion_contrast: float = 0.4,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic (volume, mask): Gaussian-noise background + random ellipsoid
    lesions. No real CT dataset is bundled -- see module docstring."""
    rng = np.random.default_rng(seed)
    d, h, w = shape
    volume = rng.normal(loc=background_level, scale=noise_std, size=shape).astype(np.float32)
    mask = np.zeros(shape, dtype=np.uint8)
    zz, yy, xx = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
    min_dim = min(shape)
    for _ in range(num_lesions):
        radius = rng.uniform(min_radius_frac, max_radius_frac) * min_dim
        margin = radius + 1
        cz = rng.uniform(margin, d - margin) if d > 2 * margin else d / 2
        cy = rng.uniform(margin, h - margin) if h > 2 * margin else h / 2
        cx = rng.uniform(margin, w - margin) if w > 2 * margin else w / 2
        rz, ry, rx = (radius * rng.uniform(0.8, 1.2) for _ in range(3))
        dist = ((zz - cz) / rz) ** 2 + ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
        lesion_region = dist <= 1.0
        mask[lesion_region] = 1
        volume[lesion_region] += lesion_contrast
    return np.clip(volume, 0.0, 1.0).astype(np.float32), mask


class SyntheticSegmentationDataset(Dataset):
    def __init__(self, num_samples: int, shape=VOLUME_SHAPE, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.samples = [
            generate_volume_and_mask(shape=shape, seed=int(rng.integers(0, 2**31 - 1)))
            for _ in range(num_samples)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        volume, mask = self.samples[idx]
        return (
            torch.from_numpy(volume).unsqueeze(0),
            torch.from_numpy(mask.astype(np.float32)).unsqueeze(0),
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


def train_on_synthetic_data(epochs: int = 5, device: str = "cpu") -> UNet3D:
    model = UNet3D(channels=CHANNELS).to(device)
    train_ds = SyntheticSegmentationDataset(40, seed=0)
    loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(epochs):
        for volumes, masks in loader:
            volumes, masks = volumes.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(volumes)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def get_or_train_model(checkpoint_dir: Path, device: str = "cpu", epochs: int = 5) -> UNet3D:
    """Load the cached demo checkpoint, or train + cache one if this is the
    first call. Training and the checkpoint write are cheap (CPU, seconds)
    by design -- see module docstring."""
    checkpoint_path = Path(checkpoint_dir) / CHECKPOINT_NAME
    model = UNet3D(channels=CHANNELS).to(device)
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        return model

    model = train_on_synthetic_data(epochs=epochs, device=device)
    _atomic_save(model, checkpoint_path)
    return model


def predict_mask(model: UNet3D, volume: np.ndarray, device: str = "cpu", threshold: float = 0.5) -> np.ndarray:
    """Run inference on one preprocessed volume (shape == VOLUME_SHAPE), return a uint8 mask."""
    with torch.no_grad():
        tensor = torch.from_numpy(volume.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        logits = model(tensor)
        probs = torch.sigmoid(logits)
        mask = (probs > threshold).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
    return mask
