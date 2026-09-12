from __future__ import annotations

import cv2
from pathlib import Path
from typing import Any

from django.conf import settings

from imaging_core.registry import BaseImagingTask, ImagingTaskError, InputKind, TaskResult, register_task
from imaging_core.visualize import save_overlay_2d

from .model import IMG_SIZE, get_or_train_model, predict_boxes


@register_task
class RoiKeypointsTask(BaseImagingTask):
    key = "roi_keypoints"
    label = "2D ROI / Keypoint Detection"
    description = (
        "YOLO region-of-interest detection on a 2D medical image (e.g. a chest "
        "X-ray), the first stage of a detect-then-localize keypoint pipeline. "
        "Demo model trains from scratch (no pretrained-weight download) on a "
        "small synthetic bright-rectangle dataset -- see roi_dataset.py for the "
        "real CSV-annotation -> YOLO-label conversion path (e.g. VinDr-CXR)."
    )
    input_kind = InputKind.IMAGE_2D
    accepted_extensions = (".png", ".jpg", ".jpeg")

    def run(self, input_path: Path, output_dir: Path, params: dict[str, Any]) -> TaskResult:
        image = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ImagingTaskError(f"Could not read the uploaded image: {input_path.name}")

        resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        resized_path = output_dir / "input_resized.png"
        cv2.imwrite(str(resized_path), resized)

        model = get_or_train_model(settings.CHECKPOINT_DIR, device=settings.IMAGING_DEVICE)
        boxes = predict_boxes(model, resized_path)

        viz_path = save_overlay_2d(
            resized.astype(float) / 255.0,
            boxes=[b[:4] for b in boxes],
            out_path=output_dir / "detections.png",
            title=f"{len(boxes)} ROI(s) detected",
        )

        return TaskResult(
            metrics={
                "original_shape": list(image.shape),
                "model_input_size": IMG_SIZE,
                "num_detections": len(boxes),
                "detections": [
                    {"x_min": round(b[0], 1), "y_min": round(b[1], 1), "x_max": round(b[2], 1), "y_max": round(b[3], 1), "confidence": round(b[4], 3)}
                    for b in boxes
                ],
            },
            visualization_paths=[viz_path.name],
            summary=f"{len(boxes)} region(s) of interest detected (demo model, synthetic training data).",
        )
