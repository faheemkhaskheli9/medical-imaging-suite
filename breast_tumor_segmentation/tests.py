from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
from django.test import SimpleTestCase, override_settings

from imaging_core.registry import get_task

from . import dataset_download, model


class SyntheticDataTests(SimpleTestCase):
    def test_generate_volume_and_mask_has_expected_shape(self):
        volume, mask = model.generate_volume_and_mask(shape=(16, 16, 12), seed=0)
        self.assertEqual(volume.shape, (16, 16, 12))
        self.assertEqual(mask.shape, (16, 16, 12))
        self.assertGreater(mask.sum(), 0)


class TrainAndPredictTests(SimpleTestCase):
    def test_tiny_train_then_predict_round_trip(self):
        with TemporaryDirectory() as tmp:
            trained = model.get_or_train_model(Path(tmp), device="cpu", epochs=1)
            volume, _mask = model.generate_volume_and_mask(seed=1)
            predicted = model.predict_mask(trained, volume, device="cpu")
            self.assertEqual(predicted.shape, model.VOLUME_SHAPE)


class DatasetManifestTests(SimpleTestCase):
    def test_synthetic_manifest_materializes_expected_case_count(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "dataset"
            manifest = dataset_download.Manifest(
                name="demo", version="1", mode="synthetic", root=root,
                synthetic={"num_cases": 3, "shape": (8, 8, 8), "seed": 0},
            )
            report = dataset_download.download_dataset(manifest)
            self.assertEqual(report.num_cases, 3)
            self.assertTrue(dataset_download.verify_dataset(root))

    def test_verify_dataset_false_for_incomplete_run(self):
        with TemporaryDirectory() as tmp:
            self.assertFalse(dataset_download.verify_dataset(Path(tmp) / "nowhere"))


class TaskEndToEndTests(SimpleTestCase):
    def test_run_on_a_real_nifti_upload_produces_metrics(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nifti_path = tmp_path / "in.nii.gz"
            array = np.random.default_rng(3).normal(0.4, 0.1, size=(18, 18, 14)).astype(np.float32)
            nib.save(nib.Nifti1Image(array, np.eye(4)), nifti_path)

            output_dir = tmp_path / "out"
            output_dir.mkdir()

            with override_settings(CHECKPOINT_DIR=tmp_path / "checkpoints"):
                result = get_task("breast_tumor_segmentation").run(nifti_path, output_dir, {})

            self.assertIn("predicted_tumor_fraction", result.metrics)
            self.assertTrue((output_dir / result.visualization_paths[0]).exists())
