"""Reproducible, idempotent dataset acquisition for breast-MRI segmentation.

Two manifest-driven modes (``configs/data.yaml``):

* ``synthetic`` -- deterministically generates a small set of NIfTI
  image/mask pairs locally. No network, CC0, always available for CI and
  the CPU demo.
* ``http`` -- downloads listed archives by URL, verifies each against a
  declared SHA-256, and optionally extracts them.

Robustness (per the project's review rules):

* every file is streamed to a ``*.part`` temp in the same directory and
  ``os.replace``\\d into place only after its checksum verifies -- an
  interrupted download never leaves a truncated file that looks complete;
* completion is marked by an explicit ``.complete`` JSON written last, so a
  killed run is detected and redone rather than skipped;
* an HTTP 200 whose body is not a valid archive is rejected before it is
  cached or extracted;
* zip/tar members that escape the target directory ("zip slip") are
  refused;
* the politeness ``sleep`` only runs after a real network request.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

__all__ = [
    "DatasetError",
    "DownloadReport",
    "Manifest",
    "download_dataset",
    "load_manifest",
    "verify_dataset",
]

DEFAULT_MANIFEST_PATH = Path("configs/data.yaml")
_COMPLETE_MARKER = ".complete"
_CHUNK = 1 << 20


class DatasetError(RuntimeError):
    """Dataset could not be acquired or failed validation."""


@dataclass
class Manifest:
    name: str
    version: str
    mode: str
    root: Path
    license: str = "unspecified"
    source_url: str = "unspecified"
    synthetic: dict[str, Any] = field(default_factory=dict)
    files: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Manifest:
        try:
            ds = raw["dataset"]
            mode = raw.get("mode", "synthetic")
            manifest = cls(
                name=str(ds["name"]),
                version=str(ds["version"]),
                mode=mode,
                root=Path(ds["root"]),
                license=str(ds.get("license", "unspecified")),
                source_url=str(ds.get("source_url", "unspecified")),
                synthetic=dict(raw.get("synthetic", {})),
                files=list(raw.get("files", [])),
            )
        except (KeyError, TypeError) as exc:
            raise DatasetError(f"malformed dataset manifest: missing/invalid {exc}") from exc

        if manifest.mode not in {"synthetic", "http"}:
            raise DatasetError(
                f"unknown dataset mode {manifest.mode!r} (expected 'synthetic' or 'http')"
            )
        if manifest.mode == "http" and not manifest.files:
            raise DatasetError("http mode requires a non-empty 'files:' list")
        return manifest


@dataclass
class DownloadReport:
    root: Path
    mode: str
    num_cases: int
    files_downloaded: int
    files_skipped: int
    bytes_written: int
    was_complete: bool


def load_manifest(path: str | os.PathLike[str] | None = None) -> Manifest:
    """Load the dataset manifest.

    A missing *default* path is an error too (there is no sane built-in
    fallback for "which dataset"), but an explicitly-passed path that does
    not exist fails loudly rather than silently using something else.
    """
    explicit = path is not None
    resolved = Path(path) if explicit else DEFAULT_MANIFEST_PATH
    if not resolved.exists():
        raise DatasetError(
            f"dataset manifest not found: {resolved}"
            + ("" if explicit else " (pass --config or create configs/data.yaml)")
        )
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise DatasetError(f"could not parse {resolved}: {exc}") from exc
    if not isinstance(raw, dict):
        raise DatasetError(f"{resolved} must contain a YAML mapping")
    return Manifest.from_dict(raw)


# --------------------------------------------------------------------- synthetic
def _sphere_mask(shape: tuple[int, int, int], center, radius: float) -> np.ndarray:
    grids = np.ogrid[tuple(slice(0, s) for s in shape)]
    dist2 = sum((g - c) ** 2 for g, c in zip(grids, center))
    return (dist2 <= radius**2).astype(np.uint8)


def _materialize_synthetic(manifest: Manifest, dest_root: Path) -> int:
    import nibabel as nib  # noqa: PLC0415 - heavy, only needed for generation

    cfg = manifest.synthetic
    num_cases = int(cfg.get("num_cases", 6))
    shape = tuple(int(x) for x in cfg.get("shape", (32, 32, 24)))
    if len(shape) != 3 or any(s < 8 for s in shape):
        raise DatasetError(f"synthetic.shape must be 3 dims >= 8, got {shape}")
    seed = int(cfg.get("seed", 42))
    rng = np.random.default_rng(seed)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])

    for case_idx in range(num_cases):
        case_dir = dest_root / f"case_{case_idx:04d}"
        case_dir.mkdir(parents=True, exist_ok=True)

        base = rng.normal(120.0, 15.0, size=shape)
        center = [rng.integers(s // 4, 3 * s // 4) for s in shape]
        radius = float(rng.uniform(min(shape) * 0.12, min(shape) * 0.22))
        mask = _sphere_mask(shape, center, radius)
        image = base + mask * rng.uniform(60.0, 110.0)
        image = np.clip(image, 0, None).astype(np.int16)

        nib.save(nib.Nifti1Image(image, affine), case_dir / "image.nii.gz")
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), case_dir / "mask.nii.gz")

    return num_cases


# ------------------------------------------------------------------------- http
def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_one(
    spec: dict[str, Any],
    dest_dir: Path,
    *,
    session: Any,
    delay: float,
    force: bool,
) -> tuple[Path, int, bool]:
    url = spec["url"]
    expected = str(spec["sha256"]).lower()
    dest = dest_dir / spec.get("dest", Path(url).name)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force and _sha256_of(dest) == expected:
        return dest, 0, True

    tmp = dest.with_name(dest.name + f".{os.getpid()}.part")
    digest = hashlib.sha256()
    written = 0
    try:
        response = session.get(url, stream=True, timeout=60)
        try:
            status = getattr(response, "status_code", 200)
            if status != 200:
                raise DatasetError(f"GET {url} returned HTTP {status}")
            with tmp.open("wb") as out:
                for chunk in response.iter_content(_CHUNK):
                    if not chunk:
                        continue
                    out.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
        finally:
            if delay > 0:
                time.sleep(delay)
            closer = getattr(response, "close", None)
            if callable(closer):
                closer()

        actual = digest.hexdigest()
        if actual != expected:
            raise DatasetError(
                f"checksum mismatch for {url}\n  expected {expected}\n  got      {actual}"
            )
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    if spec.get("extract"):
        _extract_archive(dest, dest_dir)
    return dest, written, False


def _extract_archive(archive: Path, dest_dir: Path) -> None:
    staging = dest_dir / f".extract.{os.getpid()}.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                _guard_members(zf.namelist(), staging)
                zf.extractall(staging)
        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive) as tf:
                names = [m.name for m in tf.getmembers()]
                _guard_members(names, staging)
                tf.extractall(staging)  # noqa: S202 - members pre-validated above
        else:
            raise DatasetError(f"{archive} is not a valid zip or tar archive")

        for item in staging.iterdir():
            target = dest_dir / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            os.replace(item, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _guard_members(names: list[str], base: Path) -> None:
    base_resolved = base.resolve()
    for name in names:
        resolved = (base / name).resolve()
        if base_resolved != resolved and base_resolved not in resolved.parents:
            raise DatasetError(f"unsafe archive member escapes target dir: {name!r}")


# --------------------------------------------------------------------- orchestrate
def _marker_path(root: Path) -> Path:
    return root / _COMPLETE_MARKER


def verify_dataset(root: str | os.PathLike[str]) -> bool:
    """True iff ``root`` has a ``.complete`` marker and its listed cases exist."""
    root = Path(root)
    marker = _marker_path(root)
    if not marker.exists():
        return False
    try:
        info = json.loads(marker.read_text(encoding="utf-8"))
        cases = info.get("cases", [])
        return all((root / case).exists() for case in cases) and len(cases) > 0
    except (json.JSONDecodeError, OSError):
        return False


def download_dataset(
    manifest: Manifest,
    *,
    force: bool = False,
    session: Any | None = None,
    request_delay_seconds: float = 1.0,
) -> DownloadReport:
    """Materialize the dataset described by ``manifest`` under its ``root``."""
    root = manifest.root

    if not force and verify_dataset(root):
        info = json.loads(_marker_path(root).read_text(encoding="utf-8"))
        return DownloadReport(
            root=root,
            mode=manifest.mode,
            num_cases=len(info.get("cases", [])),
            files_downloaded=0,
            files_skipped=len(info.get("cases", [])),
            bytes_written=0,
            was_complete=True,
        )

    downloaded = skipped = bytes_written = 0

    if manifest.mode == "synthetic":
        staging = root.with_name(root.name + f".{os.getpid()}.tmp")
        if staging.exists():
            shutil.rmtree(staging)
        try:
            num_cases = _materialize_synthetic(manifest, staging)
            if root.exists():
                shutil.rmtree(root)
            staging.rename(root)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        cases = sorted(p.name for p in root.iterdir() if p.is_dir())
    else:
        if session is None:
            import requests  # noqa: PLC0415 - only needed for real downloads

            session = requests
        root.mkdir(parents=True, exist_ok=True)
        for spec in manifest.files:
            _, wrote, was_skipped = _download_one(
                spec,
                root,
                session=session,
                delay=request_delay_seconds,
                force=force,
            )
            bytes_written += wrote
            downloaded += int(not was_skipped)
            skipped += int(was_skipped)
        cases = sorted(
            p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
        )
        num_cases = len(cases)

    marker = {
        "name": manifest.name,
        "version": manifest.version,
        "mode": manifest.mode,
        "cases": cases,
        "synthetic_seed": manifest.synthetic.get("seed"),
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _atomic_write_text(_marker_path(root), json.dumps(marker, indent=2))

    return DownloadReport(
        root=root,
        mode=manifest.mode,
        num_cases=num_cases,
        files_downloaded=downloaded,
        files_skipped=skipped,
        bytes_written=bytes_written,
        was_complete=False,
    )


def _atomic_write_text(target: Path, text: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
