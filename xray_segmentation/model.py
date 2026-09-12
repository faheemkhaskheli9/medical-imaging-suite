"""2D U-Net for X-ray segmentation.

``xray-image-segmentation``'s ``src/`` was still empty (Phase 1 not started)
-- this is new code, mirroring ``ct_segmentation``'s architecture/approach
one dimension down: a small 2D U-Net, Dice+BCE loss, trained on a synthetic
bright-lesion-on-noise generator so the feature is runnable end-to-end from
the UI, same honesty caveat (not validated on real anatomy).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

IMAGE_SIZE = (128, 128)
CHANNELS = (16, 32, 64, 128)
CHECKPOINT_NAME = "xray_segmentation_unet2d.pt"


class DoubleConv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv2D(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up2D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv2D(in_channels // 2 + skip_channels, out_channels)

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


class UNet2D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, channels=CHANNELS):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.in_conv = DoubleConv2D(in_channels, c1)
        self.down1 = Down2D(c1, c2)
        self.down2 = Down2D(c2, c3)
        self.down3 = Down2D(c3, c4)
        self.up1 = Up2D(c4, c3, c3)
        self.up2 = Up2D(c3, c2, c2)
        self.up3 = Up2D(c2, c1, c1)
        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=1)

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


def generate_image_and_mask(shape=IMAGE_SIZE, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic (X-ray-like image, lesion mask): noisy grayscale background
    with one bright elliptical lesion."""
    rng = np.random.default_rng(seed)
    h, w = shape
    image = rng.normal(loc=0.35, scale=0.06, size=shape).astype(np.float32)
    mask = np.zeros(shape, dtype=np.uint8)

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    min_dim = min(shape)
    radius = rng.uniform(0.1, 0.25) * min_dim
    margin = radius + 1
    cy = rng.uniform(margin, h - margin) if h > 2 * margin else h / 2
    cx = rng.uniform(margin, w - margin) if w > 2 * margin else w / 2
    ry, rx = radius * rng.uniform(0.8, 1.2), radius * rng.uniform(0.8, 1.2)
    dist = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
    region = dist <= 1.0
    mask[region] = 1
    image[region] += 0.4

    return np.clip(image, 0.0, 1.0).astype(np.float32), mask


class SyntheticXraySegmentationDataset(Dataset):
    def __init__(self, num_samples: int, shape=IMAGE_SIZE, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.samples = [
            generate_image_and_mask(shape=shape, seed=int(rng.integers(0, 2**31 - 1))) for _ in range(num_samples)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image, mask = self.samples[idx]
        return (
            torch.from_numpy(image).unsqueeze(0),
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


def train_on_synthetic_data(epochs: int = 5, device: str = "cpu") -> UNet2D:
    model = UNet2D(channels=CHANNELS).to(device)
    loader = DataLoader(SyntheticXraySegmentationDataset(60, seed=0), batch_size=8, shuffle=True)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def get_or_train_model(checkpoint_dir: Path, device: str = "cpu", epochs: int = 5) -> UNet2D:
    checkpoint_path = Path(checkpoint_dir) / CHECKPOINT_NAME
    if checkpoint_path.exists():
        model = UNet2D(channels=CHANNELS).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        return model
    model = train_on_synthetic_data(epochs=epochs, device=device)
    _atomic_save(model, checkpoint_path)
    return model


def predict_mask(model: UNet2D, image: np.ndarray, device: str = "cpu", threshold: float = 0.5) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        probs = torch.sigmoid(model(tensor))
        return (probs > threshold).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
