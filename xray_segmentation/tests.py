from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from django.test import SimpleTestCase, override_settings

from imaging_core.registry import get_task

from . import model


class SyntheticDataTests(SimpleTestCase):
    def test_generate_image_and_mask_has_expected_shape_and_range(self):
        image, mask = model.generate_image_and_mask(shape=(32, 32), seed=0)
        self.assertEqual(image.shape, (32, 32))
        self.assertEqual(mask.shape, (32, 32))
        self.assertGreaterEqual(image.min(), 0.0)
        self.assertLessEqual(image.max(), 1.0)
        self.assertGreater(mask.sum(), 0)


class TrainAndPredictTests(SimpleTestCase):
    def test_tiny_train_then_predict_round_trip(self):
        with TemporaryDirectory() as tmp:
            trained = model.get_or_train_model(Path(tmp), device="cpu", epochs=1)
            image, _mask = model.generate_image_and_mask(seed=1)
            predicted = model.predict_mask(trained, image, device="cpu")
            self.assertEqual(predicted.shape, model.IMAGE_SIZE)


class TaskEndToEndTests(SimpleTestCase):
    def test_run_on_a_real_image_upload_produces_metrics(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_path = tmp_path / "xray.png"
            image = np.random.default_rng(5).integers(0, 255, size=(50, 50), dtype=np.uint8)
            cv2.imwrite(str(image_path), image)

            output_dir = tmp_path / "out"
            output_dir.mkdir()

            with override_settings(CHECKPOINT_DIR=tmp_path / "checkpoints"):
                result = get_task("xray_segmentation").run(image_path, output_dir, {})

            self.assertIn("predicted_foreground_fraction", result.metrics)
            self.assertTrue((output_dir / result.visualization_paths[0]).exists())
