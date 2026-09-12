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
class BreastTumorSegmentationTask(BaseImagingTask):
    key = "breast_tumor_segmentation"
    label = "3D Breast Tumor Segmentation"
    description = (
        "3D U-Net tumor segmentation on a breast MRI volume (NIfTI or a .zip of a "
        "DICOM series). Demo model is trained only on a synthetic sphere-tumor "
        "generator -- not validated on real anatomy. See dataset_download.py for a "
        "manifest-driven real-dataset acquisition path (synthetic/http modes)."
    )
    input_kind = InputKind.VOLUME_3D
    accepted_extensions = (".nii", ".nii.gz", ".zip")

    def run(self, input_path: Path, output_dir: Path, params: dict[str, Any]) -> TaskResult:
        try:
            volume = load_volume_upload(input_path)
        except Exception as exc:
            raise ImagingTaskError(f"Could not read the uploaded volume: {exc}") from exc

        normalized = normalize_intensity(volume.array, method="percentile")
        resized = resize_to_shape(normalized, VOLUME_SHAPE)

        model = get_or_train_model(settings.CHECKPOINT_DIR, device=settings.IMAGING_DEVICE)
        mask = predict_mask(model, resized, device=settings.IMAGING_DEVICE)

        viz_path = save_orthogonal_slices(
            resized, mask=None, pred=mask, out_path=output_dir / "prediction.png", title="Predicted tumor mask"
        )

        foreground_voxels = int(mask.sum())
        return TaskResult(
            metrics={
                "input_shape": list(volume.shape),
                "model_input_shape": list(VOLUME_SHAPE),
                "predicted_tumor_voxels": foreground_voxels,
                "predicted_tumor_fraction": round(foreground_voxels / mask.size, 4),
            },
            visualization_paths=[viz_path.name],
            summary=(
                "Inference-only demo run on a synthetic-data-trained model; no "
                "ground-truth mask was uploaded, so no Dice/IoU is reported."
            ),
        )
