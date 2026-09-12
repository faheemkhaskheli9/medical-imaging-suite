from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from django.test import SimpleTestCase, override_settings

from imaging_core.registry import get_task

from . import model, roi_dataset


class RoiDatasetTests(SimpleTestCase):
    def test_to_yolo_label_normalizes_box_to_unit_range(self):
        record = roi_dataset.AnnotationRecord(
            image_id="img1", class_id=0, x_min=10, y_min=20, x_max=50, y_max=60,
            image_width=100, image_height=100,
        )
        label = roi_dataset.to_yolo_label(record)
        class_id, cx, cy, w, h = (float(x) for x in label.split())
        self.assertEqual(class_id, 0)
        for value in (cx, cy, w, h):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_invalid_box_bounds_raise(self):
        with self.assertRaises(ValueError):
            roi_dataset.AnnotationRecord(
                image_id="img1", class_id=0, x_min=50, y_min=0, x_max=10, y_max=10,
                image_width=100, image_height=100,
            )

    def test_split_is_deterministic_across_calls(self):
        ids = [f"img{i}" for i in range(20)]
        split_a = roi_dataset.split_image_ids(ids, seed=42)
        split_b = roi_dataset.split_image_ids(ids, seed=42)
        self.assertEqual(split_a, split_b)


class SyntheticDataTests(SimpleTestCase):
    def test_generated_image_box_is_within_bounds(self):
        image, box = model.generate_synthetic_roi_image(size=64, seed=0)
        self.assertEqual(image.shape, (64, 64))
        x_min, y_min, x_max, y_max = box
        self.assertTrue(0 <= x_min < x_max <= 64)
        self.assertTrue(0 <= y_min < y_max <= 64)


class TaskEndToEndTests(SimpleTestCase):
    def test_run_on_a_real_image_upload_produces_detections_field(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "chest.png"
            image = np.random.default_rng(6).integers(0, 255, size=(64, 64), dtype=np.uint8)
            cv2.imwrite(str(image_path), image)

            output_dir = tmp_path / "out"
            output_dir.mkdir()

            with override_settings(CHECKPOINT_DIR=tmp_path / "checkpoints"):
                result = get_task("roi_keypoints").run(image_path, output_dir, {})

            self.assertIn("num_detections", result.metrics)
            self.assertIn("detections", result.metrics)
            self.assertTrue((output_dir / result.visualization_paths[0]).exists())
