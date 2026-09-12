from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from django.test import SimpleTestCase

from . import io as imaging_io
from . import preprocessing


class NiftiRoundTripTests(SimpleTestCase):
    def test_save_then_load_nifti_round_trips_array(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "vol.nii.gz"
            array = np.random.default_rng(0).normal(size=(6, 7, 8)).astype(np.float32)
            imaging_io.save_nifti(array, np.eye(4), path)

            volume = imaging_io.load_volume(path)
            np.testing.assert_allclose(volume.array, array, atol=1e-4)
            self.assertEqual(volume.shape, (6, 7, 8))

    def test_load_volume_missing_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            imaging_io.load_volume("does/not/exist.nii.gz")

    def test_load_volume_unsupported_suffix_raises_loader_error(self):
        with TemporaryDirectory() as tmp:
            bogus = Path(tmp) / "not_a_volume.txt"
            bogus.write_text("nope", encoding="utf-8")
            with self.assertRaises(imaging_io.LoaderError):
                imaging_io.load_volume(bogus)


class UploadZipTests(SimpleTestCase):
    def test_load_volume_upload_rejects_non_zip_non_nifti(self):
        with TemporaryDirectory() as tmp:
            bogus = Path(tmp) / "scan.doc"
            bogus.write_text("nope", encoding="utf-8")
            with self.assertRaises(imaging_io.LoaderError):
                imaging_io.load_volume_upload(bogus)

    def test_load_volume_upload_rejects_corrupt_zip(self):
        with TemporaryDirectory() as tmp:
            fake_zip = Path(tmp) / "series.zip"
            fake_zip.write_bytes(b"not actually a zip file")
            with self.assertRaises(imaging_io.LoaderError):
                imaging_io.load_volume_upload(fake_zip)


class PreprocessingTests(SimpleTestCase):
    def test_hu_window_normalize_clips_and_scales_to_unit_range(self):
        array = np.array([-2000.0, -1000.0, -300.0, 400.0, 2000.0])
        normalized = preprocessing.normalize_intensity(array, method="hu_window", hu_window=(-1000.0, 400.0))
        self.assertAlmostEqual(float(normalized.min()), 0.0)
        self.assertAlmostEqual(float(normalized.max()), 1.0)

    def test_invalid_hu_window_raises(self):
        with self.assertRaises(preprocessing.PreprocessError):
            preprocessing.normalize_intensity(np.zeros(4), method="hu_window", hu_window=(400.0, -1000.0))

    def test_crop_or_pad_produces_exact_target_shape(self):
        array = np.ones((10, 4, 20))
        out = preprocessing.crop_or_pad(array, (6, 8, 6))
        self.assertEqual(out.shape, (6, 8, 6))

    def test_resize_to_shape_produces_exact_target_shape(self):
        array = np.random.default_rng(1).normal(size=(17, 9, 23)).astype(np.float32)
        out = preprocessing.resize_to_shape(array, (32, 32, 32))
        self.assertEqual(out.shape, (32, 32, 32))

    def test_unknown_normalization_method_raises(self):
        with self.assertRaises(preprocessing.PreprocessError):
            preprocessing.normalize_intensity(np.zeros(4), method="not-a-real-method")
