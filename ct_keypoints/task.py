from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any

from django.conf import settings

from imaging_core.io import load_volume_upload
from imaging_core.preprocessing import normalize_intensity, resize_to_shape
from imaging_core.registry import BaseImagingTask, ImagingTaskError, InputKind, TaskResult, register_task
from imaging_core.visualize import save_keypoint_slices

from .model import VOLUME_SHAPE, get_or_train_model, predict_keypoint


@register_task
class CtKeypointsTask(BaseImagingTask):
    key = "ct_keypoints"
    label = "3D CT Keypoint Detection"
    description = (
        "3D anatomical landmark localization on a CT volume (NIfTI or a .zip of a "
        "DICOM series) via heatmap regression. Demo model is trained only on a "
        "synthetic single-landmark generator -- not validated on real anatomy."
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
        coord, _heatmap, confidence = predict_keypoint(model, resized, device=settings.IMAGING_DEVICE)

        viz_path = save_keypoint_slices(
            resized,
            keypoints=None,
            pred_keypoints=np.array([coord]),
            out_path=output_dir / "keypoint.png",
            title="Predicted landmark",
        )

        return TaskResult(
            metrics={
                "input_shape": list(volume.shape),
                "model_input_shape": list(VOLUME_SHAPE),
                "predicted_keypoint_zyx": list(coord),
                "peak_confidence": round(confidence, 4),
            },
            visualization_paths=[viz_path.name],
            summary=(
                "Inference-only demo run (no ground-truth landmark was provided, so "
                "no Euclidean-distance error is reported)."
            ),
        )
