"""ROI dataset ingestion and YOLO-format conversion (issue #1, Phase 1).

Reads pixel-space bounding-box annotations (image_id, x_min, y_min, x_max,
y_max, class_id, image_width, image_height — the schema shared by most
public chest X-ray ROI datasets, e.g. VinDr-CXR), converts them to
Ultralytics YOLO label format, and produces a deterministic train/val/test
split by image id.
"""
from __future__ import annotations

import csv
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class AnnotationRecord:
    image_id: str
    class_id: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    image_width: float
    image_height: float

    def __post_init__(self):
        if not (0 <= self.x_min < self.x_max <= self.image_width):
            raise ValueError(f"Invalid x bounds for {self.image_id}: {self.x_min}..{self.x_max} (width {self.image_width})")
        if not (0 <= self.y_min < self.y_max <= self.image_height):
            raise ValueError(f"Invalid y bounds for {self.image_id}: {self.y_min}..{self.y_max} (height {self.image_height})")


def load_annotations(csv_path: str | Path) -> List[AnnotationRecord]:
    """Loads pixel-space ROI annotations from a CSV with columns:
    image_id, class_id, x_min, y_min, x_max, y_max, image_width, image_height
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {csv_path}")

    records = []
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            records.append(
                AnnotationRecord(
                    image_id=row["image_id"],
                    class_id=int(row["class_id"]),
                    x_min=float(row["x_min"]),
                    y_min=float(row["y_min"]),
                    x_max=float(row["x_max"]),
                    y_max=float(row["y_max"]),
                    image_width=float(row["image_width"]),
                    image_height=float(row["image_height"]),
                )
            )
    return records


def to_yolo_label(record: AnnotationRecord) -> str:
    """Converts one pixel-space box to a YOLO label line:
    `class_id center_x center_y width height`, all normalized to [0, 1].
    """
    box_width = record.x_max - record.x_min
    box_height = record.y_max - record.y_min
    center_x = (record.x_min + box_width / 2) / record.image_width
    center_y = (record.y_min + box_height / 2) / record.image_height
    norm_width = box_width / record.image_width
    norm_height = box_height / record.image_height
    return f"{record.class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"


def write_yolo_labels(records: List[AnnotationRecord], labels_dir: str | Path) -> None:
    """Writes one `<image_id>.txt` per image (multiple boxes -> multiple lines)."""
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    by_image: Dict[str, List[AnnotationRecord]] = {}
    for record in records:
        by_image.setdefault(record.image_id, []).append(record)

    for image_id, image_records in by_image.items():
        lines = [to_yolo_label(r) for r in image_records]
        (labels_dir / f"{image_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def split_image_ids(
    image_ids: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Deterministic train/val/test split by image id.

    Deterministic across runs/machines: ids are sorted first (removes any
    incoming-order dependence), then shuffled with a seeded RNG local to
    this call (doesn't disturb global `random` state elsewhere).
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    unique_ids = sorted(set(image_ids))
    rng = random.Random(seed)
    shuffled = unique_ids.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * train_ratio)
    n_val = round(n * val_ratio)

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train:n_train + n_val]
    test_ids = shuffled[n_train + n_val:]
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def stable_hash(image_id: str) -> str:
    """Used by tests/tooling to sanity-check id stability across environments."""
    return hashlib.sha256(image_id.encode("utf-8")).hexdigest()
