"""Shared image/volume I/O: NIfTI, DICOM series, and 2D images.

Merged from three separate near-duplicate implementations across the
original repos (``3d-ct-segmentation/src/data/io.py``,
``3d-medical-keypoint-detection/src/ct_keypoints/io.py``,
``3d-breast-tumor-segmentation/src/breast_seg/preprocessing.py``'s NIfTI
loading) into one module every feature app now shares -- per the "harden
siblings the same way" rule: the DICOM-series loader, HU rescale, and
largest-series selection those three repos each needed only one of, now
apply uniformly.

Array/spacing convention: ``spacing`` is given in the same axis order as
``array`` (millimetres); the 4x4 ``affine`` maps voxel indices to physical
coordinates.
"""
from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

__all__ = [
    "LoaderError",
    "Volume",
    "load_volume",
    "load_volume_upload",
    "save_nifti",
    "load_image_2d",
    "save_image_2d",
]

_NIFTI_SUFFIXES = (".nii", ".nii.gz")


class LoaderError(RuntimeError):
    """A volume/image could not be loaded or is internally inconsistent."""


@dataclass
class Volume:
    array: np.ndarray
    affine: np.ndarray
    spacing: tuple[float, float, float]
    source_path: Path
    modality: str = "CT"
    axis_order: str = "ijk"

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.array.shape)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        sp = ", ".join(f"{s:.3f}" for s in self.spacing)
        return (
            f"Volume(shape={self.shape}, dtype={self.array.dtype}, "
            f"spacing=({sp}) mm, order={self.axis_order}, src={self.source_path.name})"
        )


def _has_nifti_suffix(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(_NIFTI_SUFFIXES)


def load_volume(path: str | os.PathLike[str], *, dtype: str | None = None) -> Volume:
    """Load a 3D volume from ``path``: a NIfTI file, a single ``.dcm`` file, or
    a directory of DICOM slices."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"volume path does not exist: {path}")

    if path.is_dir():
        volume = _load_dicom_series(path)
    elif _has_nifti_suffix(path):
        volume = _load_nifti(path)
    elif path.suffix.lower() == ".dcm":
        volume = _load_dicom_series(path.parent)
        volume.source_path = path
    else:
        raise LoaderError(
            f"unsupported volume path {path.name!r}: expected a .nii/.nii.gz file, "
            "a .dcm file, or a directory of DICOM slices"
        )

    if dtype is not None:
        volume.array = volume.array.astype(dtype)
    return volume


def load_volume_upload(upload_path: str | os.PathLike[str]) -> Volume:
    """Load a volume from a web-uploaded file: a NIfTI file, or a .zip of a
    DICOM series (the shape a browser upload takes for a multi-file series --
    a raw directory can't be uploaded as one HTML form field).
    """
    upload_path = Path(upload_path)
    if _has_nifti_suffix(upload_path):
        return load_volume(upload_path)
    if upload_path.suffix.lower() == ".zip":
        with TemporaryDirectory(prefix="dicom_upload_") as tmp:
            tmp_dir = Path(tmp)
            try:
                with zipfile.ZipFile(upload_path) as zf:
                    zf.extractall(tmp_dir)
            except zipfile.BadZipFile as exc:
                raise LoaderError(f"uploaded file is not a valid .zip: {exc}") from exc
            # A series zip may nest the .dcm files one level deep; search
            # for the directory that actually holds them.
            series_dir = _find_dicom_dir(tmp_dir)
            return _load_dicom_series(series_dir)
    raise LoaderError(
        f"unsupported upload {upload_path.name!r}: expected .nii/.nii.gz or a .zip of DICOM files"
    )


def _find_dicom_dir(root: Path) -> Path:
    if any(p.suffix.lower() == ".dcm" for p in root.iterdir() if p.is_file()):
        return root
    for sub in sorted(p for p in root.rglob("*") if p.is_dir()):
        if any(p.suffix.lower() == ".dcm" for p in sub.iterdir() if p.is_file()):
            return sub
    # Some series ship extensionless files; fall back to "any directory with
    # files" so pydicom's own sniffing in _load_dicom_series gets a chance.
    for sub in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        if any(p.is_file() for p in sub.iterdir()):
            return sub
    raise LoaderError("uploaded .zip contained no files")


def save_nifti(data: np.ndarray, affine: np.ndarray, path: str | os.PathLike[str]) -> None:
    import nibabel as nib  # noqa: PLC0415 - heavy import, kept lazy

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)
    nib.save(img, str(path))


def _load_nifti(path: Path) -> Volume:
    import nibabel as nib  # noqa: PLC0415 - heavy import, kept lazy

    try:
        img = nib.load(str(path))
    except Exception as exc:  # nibabel raises several concrete types
        raise LoaderError(f"could not read NIfTI {path}: {exc}") from exc

    array = np.asanyarray(img.dataobj)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    if len(zooms) < 3:
        zooms = (*zooms, *(1.0,) * (3 - len(zooms)))
    return Volume(
        array=array,
        affine=np.asarray(img.affine, dtype=np.float64),
        spacing=zooms,  # type: ignore[arg-type]
        source_path=path,
        axis_order="ijk",
    )


def _load_dicom_series(directory: Path) -> Volume:
    import pydicom  # noqa: PLC0415 - heavy import, kept lazy

    if not directory.is_dir():
        raise LoaderError(f"DICOM series path is not a directory: {directory}")

    datasets = []
    for candidate in sorted(directory.iterdir()):
        if candidate.is_dir():
            continue
        try:
            ds = pydicom.dcmread(str(candidate))
        except Exception:
            continue  # not a DICOM file -- skip quietly
        if "PixelData" in ds:
            datasets.append(ds)

    if not datasets:
        raise LoaderError(f"no readable DICOM slices with pixel data in {directory}")

    # Keep only the largest series so a folder holding scout/localizer images
    # alongside the main scan still loads the right volume.
    by_series: dict[str, list] = {}
    for ds in datasets:
        by_series.setdefault(str(getattr(ds, "SeriesInstanceUID", "unknown")), []).append(ds)
    series = max(by_series.values(), key=len)

    shapes = {(int(ds.Rows), int(ds.Columns)) for ds in series}
    if len(shapes) != 1:
        raise LoaderError(f"DICOM slices in {directory} have inconsistent sizes: {shapes}")

    def _sort_key(ds):
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) == 3:
            return float(ipp[2])
        return float(getattr(ds, "InstanceNumber", 0))

    series.sort(key=_sort_key)

    slope = float(getattr(series[0], "RescaleSlope", 1.0))
    intercept = float(getattr(series[0], "RescaleIntercept", 0.0))
    stack = np.stack([ds.pixel_array.astype(np.float32) for ds in series], axis=0)
    hu = stack * slope + intercept
    hu = hu.astype(np.int16)

    row_spacing, col_spacing = (float(v) for v in getattr(series[0], "PixelSpacing", (1.0, 1.0)))
    slice_spacing = _dicom_slice_spacing(series)
    affine = _dicom_affine(series[0], row_spacing, col_spacing, slice_spacing)

    return Volume(
        array=hu,
        affine=affine,
        spacing=(slice_spacing, row_spacing, col_spacing),
        source_path=directory,
        modality=str(getattr(series[0], "Modality", "CT")),
        axis_order="kji",
    )


def _dicom_slice_spacing(series: list) -> float:
    positions = [
        getattr(ds, "ImagePositionPatient", None) for ds in series if getattr(ds, "ImagePositionPatient", None) is not None
    ]
    if len(positions) >= 2:
        diffs = np.diff([float(p[2]) for p in positions])
        diffs = diffs[np.abs(diffs) > 1e-6]
        if diffs.size:
            return float(np.median(np.abs(diffs)))
    return float(getattr(series[0], "SliceThickness", 1.0) or 1.0)


def _dicom_affine(ds, row_spacing: float, col_spacing: float, slice_spacing: float) -> np.ndarray:
    orientation = getattr(ds, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0])
    row_cos = np.asarray(orientation[:3], dtype=np.float64)
    col_cos = np.asarray(orientation[3:6], dtype=np.float64)
    normal = np.cross(row_cos, col_cos)
    origin = np.asarray(getattr(ds, "ImagePositionPatient", [0.0, 0.0, 0.0]), dtype=np.float64)

    affine = np.eye(4)
    affine[:3, 0] = normal * slice_spacing
    affine[:3, 1] = col_cos * row_spacing
    affine[:3, 2] = row_cos * col_spacing
    affine[:3, 3] = origin
    return affine


# --- 2D images ---------------------------------------------------------------


def load_image_2d(path: str | os.PathLike[str]) -> np.ndarray:
    """Load a 2D image as a float32 grayscale array in [0, 1]."""
    import cv2  # noqa: PLC0415 - heavy import, kept lazy

    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise LoaderError(f"could not read image (unsupported format or corrupt file): {path}")
    return (image.astype(np.float32)) / 255.0


def save_image_2d(array: np.ndarray, path: str | os.PathLike[str]) -> None:
    import cv2  # noqa: PLC0415 - heavy import, kept lazy

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(array, 0.0, 1.0)
    cv2.imwrite(str(path), (clipped * 255.0).astype(np.uint8))
