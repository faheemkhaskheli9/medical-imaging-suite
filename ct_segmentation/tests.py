from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
from django.test import SimpleTestCase, override_settings

from imaging_core.registry import get_task

from . import model


class SyntheticDataTests(SimpleTestCase):
    def test_generate_volume_and_mask_has_expected_shape_and_range(self):
        volume, mask = model.generate_volume_and_mask(shape=(16, 16, 16), seed=0)
        self.assertEqual(volume.shape, (16, 16, 16))
        self.assertEqual(mask.shape, (16, 16, 16))
        self.assertGreaterEqual(volume.min(), 0.0)
        self.assertLessEqual(volume.max(), 1.0)
        self.assertTrue(set(np.unique(mask)).issubset({0, 1}))
        self.assertGreater(mask.sum(), 0)  # at least one lesion voxel


class TrainAndPredictTests(SimpleTestCase):
    def test_tiny_train_then_predict_round_trip(self):
        with TemporaryDirectory() as tmp:
            trained = model.get_or_train_model(Path(tmp), device="cpu", epochs=1)
            volume, _mask = model.generate_volume_and_mask(seed=1)
            predicted = model.predict_mask(trained, volume, device="cpu")
            self.assertEqual(predicted.shape, model.VOLUME_SHAPE)
            self.assertTrue(set(np.unique(predicted)).issubset({0, 1}))

    def test_checkpoint_is_reused_on_second_call(self):
        with TemporaryDirectory() as tmp:
            model.get_or_train_model(Path(tmp), device="cpu", epochs=1)
            checkpoint_path = Path(tmp) / model.CHECKPOINT_NAME
            self.assertTrue(checkpoint_path.exists())
            mtime_first = checkpoint_path.stat().st_mtime
            model.get_or_train_model(Path(tmp), device="cpu", epochs=1)
            self.assertEqual(checkpoint_path.stat().st_mtime, mtime_first)  # not retrained


class TaskEndToEndTests(SimpleTestCase):
    def test_run_on_a_real_nifti_upload_produces_metrics_and_visualization(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nifti_path = tmp_path / "in.nii.gz"
            array = np.random.default_rng(2).normal(0.4, 0.1, size=(20, 20, 20)).astype(np.float32)
            nib.save(nib.Nifti1Image(array, np.eye(4)), nifti_path)

            output_dir = tmp_path / "out"
            output_dir.mkdir()

            with override_settings(CHECKPOINT_DIR=tmp_path / "checkpoints"):
                task = get_task("ct_segmentation")
                result = task.run(nifti_path, output_dir, {})

            self.assertIn("predicted_foreground_fraction", result.metrics)
            self.assertEqual(len(result.visualization_paths), 1)
            self.assertTrue((output_dir / result.visualization_paths[0]).exists())
