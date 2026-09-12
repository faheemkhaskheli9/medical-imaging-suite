from __future__ import annotations

from pathlib import Path
from typing import Any

from django.conf import settings

from imaging_core.io import load_volume_upload
from imaging_core.preprocessing import normalize_intensity, resize_to_shape
from imaging_core.registry import BaseImagingTask, ImagingTaskError, InputKind, TaskResult, register_task
from imaging_core.visualize import save_orthogonal_slices

from .model import VOLUME_SHAPE, get_or_train_model, predict_mask


@register_task
class CtSegmentationTask(BaseImagingTask):
    key = "ct_segmentation"
    label = "3D CT Segmentation"
    description = (
        "3D U-Net binary segmentation on a CT volume (NIfTI or a .zip of a DICOM "
        "series). Demo model is trained only on synthetic ellipsoid-lesion "
        "volumes (CPU, seconds) -- not validated on real anatomy; treat its "
        "output as a pipeline demo, not a diagnostic result."
    )
    input_kind = InputKind.VOLUME_3D
    accepted_extensions = (".nii", ".nii.gz", ".zip")

    def run(self, input_path: Path, output_dir: Path, params: dict[str, Any]) -> TaskResult:
        try:
            volume = load_volume_upload(input_path)
        except Exception as exc:
            raise ImagingTaskError(f"Could not read the uploaded volume: {exc}") from exc

        normalized = normalize_intensity(volume.array, method="hu_window")
        resized = resize_to_shape(normalized, VOLUME_SHAPE)

        model = get_or_train_model(settings.CHECKPOINT_DIR, device=settings.IMAGING_DEVICE)
        mask = predict_mask(model, resized, device=settings.IMAGING_DEVICE)

        viz_path = save_orthogonal_slices(
            resized, mask=None, pred=mask, out_path=output_dir / "prediction.png", title="Predicted segmentation"
        )

        foreground_voxels = int(mask.sum())
        total_voxels = int(mask.size)
        return TaskResult(
            metrics={
                "input_shape": list(volume.shape),
                "model_input_shape": list(VOLUME_SHAPE),
                "predicted_foreground_voxels": foreground_voxels,
                "predicted_foreground_fraction": round(foreground_voxels / total_voxels, 4),
            },
            visualization_paths=[viz_path.name],
            summary=(
                "Inference-only demo run (no ground-truth mask was uploaded, so no "
                "Dice/IoU is reported). The volume was resized to the model's fixed "
                f"{VOLUME_SHAPE} input, so fine detail from larger scans is lost."
            ),
        )
