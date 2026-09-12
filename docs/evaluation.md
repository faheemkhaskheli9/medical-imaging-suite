# Evaluation

Every feature's demo model is trained only on a synthetic generator (see
`docs/architecture.md`), so the numbers below are the training-time
self-check on that same synthetic distribution -- they validate that the
pipeline (data -> model -> loss -> inference -> visualization) works
end-to-end, not real-world diagnostic accuracy. Uploading a real scan runs
real inference, but with no ground truth attached (a single image/volume
upload, not an evaluation set), so the job result reports prediction
statistics (foreground fraction, predicted coordinate, detection count),
not Dice/IoU/distance-to-ground-truth.

## Metrics per task

| Task | Metric(s) | Computed against |
|---|---|---|
| `ct_segmentation` | Dice, IoU (training-time only) | Synthetic ellipsoid-lesion volumes |
| `ct_keypoints` | Peak-heatmap confidence (training-time: distance-to-ground-truth) | Synthetic single-landmark volumes |
| `xray_segmentation` | Dice, IoU (training-time only) | Synthetic bright-lesion images |
| `roi_keypoints` | Precision/Recall/mAP (training-time, via Ultralytics) | Synthetic bright-rectangle images |
| `breast_tumor_segmentation` | Dice, IoU (training-time only) | Synthetic sphere-tumor volumes |
| `gi_tract_segmentation` (planned, not yet ported) | Dice, IoU (training-time only, expected) | UW-Madison GI Tract dataset, consolidated from archived `UG-GI-Track-Segmentation` repo -- see `docs/architecture.md` |

## Result Log

Fill in as real datasets are trained/evaluated against (see
`breast_tumor_segmentation/dataset_download.py` +
`python manage.py download_breast_mri_dataset` for the one feature with a
real-dataset acquisition path already wired up).

| Date | Task | Dataset | Metric | Value | Notes |
|---|---|---|---|---|---|
| 2026-09-12 | ct_segmentation | synthetic (ellipsoid-in-noise) | val_dice | ~0.97 (5 epochs, CPU) | pipeline validation only, ported from `3d-ct-segmentation` |
