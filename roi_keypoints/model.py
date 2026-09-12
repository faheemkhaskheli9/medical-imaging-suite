"""YOLO ROI detection -- ported/extended from ``medical-keypoint-detection``.

That repo implemented Phase 1 (CSV annotation -> YOLO label conversion, see
``roi_dataset.py``) and a config-driven training wrapper
(``src/training/roi_trainer.py``) but had not trained a model yet. This
module completes the loop for the web demo: it builds a small synthetic
ROI dataset (a bright rectangle on a noisy background, the same idea as that
repo's ``examples/create_sample_dataset.py``), trains a from-scratch (not
internet-pretrained -- ``yolov8n.yaml``, not ``yolov8n.pt``, so no runtime
download dependency) tiny YOLO detector on it, and caches the weights.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from .roi_dataset import AnnotationRecord, to_yolo_label

IMG_SIZE = 128
CHECKPOINT_NAME = "roi_yolo_best.pt"
CLASS_NAMES = ("roi",)


def generate_synthetic_roi_image(size: int = IMG_SIZE, seed: int | None = None) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """One synthetic grayscale image with a single bright rectangular ROI,
    plus its pixel-space box (x_min, y_min, x_max, y_max)."""
    rng = np.random.default_rng(seed)
    image = rng.normal(60.0, 12.0, size=(size, size)).astype(np.float32)
    box_w = rng.uniform(0.18, 0.4) * size
    box_h = rng.uniform(0.18, 0.4) * size
    x_min = rng.uniform(0.05, 0.95 - box_w / size) * size
    y_min = rng.uniform(0.05, 0.95 - box_h / size) * size
    x_max, y_max = x_min + box_w, y_min + box_h
    image[int(y_min):int(y_max), int(x_min):int(x_max)] += rng.uniform(120.0, 170.0)
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image, (x_min, y_min, x_max, y_max)


def _write_yolo_dataset(root: Path, num_train: int = 20, num_val: int = 6, seed: int = 0) -> Path:
    for split, n, offset in (("train", num_train, 0), ("val", num_val, num_train)):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            image, box = generate_synthetic_roi_image(seed=seed + offset + i)
            image_id = f"{split}_{i:04d}"
            cv2.imwrite(str(root / "images" / split / f"{image_id}.png"), image)
            record = AnnotationRecord(
                image_id=image_id,
                class_id=0,
                x_min=box[0],
                y_min=box[1],
                x_max=box[2],
                y_max=box[3],
                image_width=IMG_SIZE,
                image_height=IMG_SIZE,
            )
            (root / "labels" / split / f"{image_id}.txt").write_text(to_yolo_label(record) + "\n", encoding="utf-8")

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "path: {}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['roi']\n".format(root.as_posix()),
        encoding="utf-8",
    )
    return data_yaml


def _atomic_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def train_on_synthetic_data(checkpoint_path: Path, epochs: int = 3, device: str = "cpu") -> None:
    """Trains a fresh YOLO detector and atomically copies its best weights to
    ``checkpoint_path``. The copy happens while the training run's temp
    directories are still alive -- they're deleted as soon as this function
    returns."""
    from ultralytics import YOLO  # noqa: PLC0415 - heavy import, kept lazy

    with TemporaryDirectory(prefix="roi_yolo_dataset_") as dataset_dir, TemporaryDirectory(prefix="roi_yolo_run_") as run_dir:
        data_yaml = _write_yolo_dataset(Path(dataset_dir))
        # yolov8n.yaml builds the architecture from scratch (no internet
        # fetch of pretrained weights) -- appropriate for a small synthetic
        # demo dataset, offline-safe.
        model = YOLO("yolov8n.yaml")
        model.train(
            data=str(data_yaml),
            imgsz=IMG_SIZE,
            epochs=epochs,
            batch=8,
            device=device,
            project=run_dir,
            name="roi_train",
            verbose=False,
            plots=False,
        )
        best_weights = Path(run_dir) / "roi_train" / "weights" / "best.pt"
        if not best_weights.exists():
            raise RuntimeError("YOLO training did not produce weights/best.pt")
        _atomic_copy(best_weights, checkpoint_path)


def get_or_train_model(checkpoint_dir: Path, device: str = "cpu", epochs: int = 3):
    from ultralytics import YOLO  # noqa: PLC0415 - heavy import, kept lazy

    checkpoint_path = Path(checkpoint_dir) / CHECKPOINT_NAME
    if not checkpoint_path.exists():
        train_on_synthetic_data(checkpoint_path, epochs=epochs, device=device)
    return YOLO(str(checkpoint_path))


def predict_boxes(model, image_path: Path, conf: float = 0.25) -> list[tuple[float, float, float, float, float]]:
    """Returns a list of (x_min, y_min, x_max, y_max, confidence) in pixels."""
    results = model.predict(source=str(image_path), conf=conf, verbose=False)
    boxes: list[tuple[float, float, float, float, float]] = []
    for result in results:
        for box, confidence in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
            x_min, y_min, x_max, y_max = box
            boxes.append((x_min, y_min, x_max, y_max, float(confidence)))
    return boxes
