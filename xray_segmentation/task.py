from __future__ import annotations

import cv2
from pathlib import Path
from typing import Any

from django.conf import settings

from imaging_core.registry import BaseImagingTask, ImagingTaskError, InputKind, TaskResult, register_task
from imaging_core.visualize import save_overlay_2d

from .model import IMAGE_SIZE, get_or_train_model, predict_mask


@register_task
class XraySegmentationTask(BaseImagingTask):
    key = "xray_segmentation"
    label = "2D X-Ray Segmentation"
    description = (
        "2D U-Net segmentation on an X-ray image. Demo model is trained only on "
        "a synthetic bright-lesion-on-noise generator (no real X-ray dataset was "
        "acquired for this feature yet) -- not validated on real anatomy."
    )
    input_kind = InputKind.IMAGE_2D
    accepted_extensions = (".png", ".jpg", ".jpeg")

    def run(self, input_path: Path, output_dir: Path, params: dict[str, Any]) -> TaskResult:
        image = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ImagingTaskError(f"Could not read the uploaded image: {input_path.name}")

        resized = cv2.resize(image, IMAGE_SIZE).astype(float) / 255.0

        model = get_or_train_model(settings.CHECKPOINT_DIR, device=settings.IMAGING_DEVICE)
        mask = predict_mask(model, resized, device=settings.IMAGING_DEVICE)

        viz_path = save_overlay_2d(
            resized, pred=mask, out_path=output_dir / "prediction.png", title="Predicted segmentation"
        )

        foreground_pixels = int(mask.sum())
        return TaskResult(
            metrics={
                "original_shape": list(image.shape),
                "model_input_size": list(IMAGE_SIZE),
                "predicted_foreground_pixels": foreground_pixels,
                "predicted_foreground_fraction": round(foreground_pixels / mask.size, 4),
            },
            visualization_paths=[viz_path.name],
            summary="Inference-only demo run on a synthetic-data-trained model; no ground-truth mask was uploaded.",
        )
