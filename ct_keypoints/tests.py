from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
from django.test import SimpleTestCase, override_settings

from imaging_core.registry import get_task

from . import model


class SyntheticDataTests(SimpleTestCase):
    def test_generate_volume_and_heatmap_peaks_at_the_landmark(self):
        volume, heatmap, center = model.generate_volume_and_heatmap(shape=(16, 16, 16), seed=0)
        self.assertEqual(volume.shape, (16, 16, 16))
        peak = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        self.assertEqual(peak, center)


class TrainAndPredictTests(SimpleTestCase):
    def test_tiny_train_then_predict_round_trip(self):
        with TemporaryDirectory() as tmp:
            trained = model.get_or_train_model(Path(tmp), device="cpu", epochs=1)
            volume, _heatmap, _center = model.generate_volume_and_heatmap(seed=1)
            coord, heatmap, confidence = model.predict_keypoint(trained, volume, device="cpu")
            self.assertEqual(len(coord), 3)
            self.assertEqual(heatmap.shape, model.VOLUME_SHAPE)
            self.assertGreaterEqual(confidence, 0.0)


class TaskEndToEndTests(SimpleTestCase):
    def test_run_on_a_real_nifti_upload_produces_a_keypoint(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nifti_path = tmp_path / "in.nii.gz"
            array = np.random.default_rng(4).normal(0.4, 0.1, size=(18, 18, 18)).astype(np.float32)
            nib.save(nib.Nifti1Image(array, np.eye(4)), nifti_path)

            output_dir = tmp_path / "out"
            output_dir.mkdir()

            with override_settings(CHECKPOINT_DIR=tmp_path / "checkpoints"):
                result = get_task("ct_keypoints").run(nifti_path, output_dir, {})

            self.assertIn("predicted_keypoint_zyx", result.metrics)
            self.assertEqual(len(result.metrics["predicted_keypoint_zyx"]), 3)
            self.assertTrue((output_dir / result.visualization_paths[0]).exists())
