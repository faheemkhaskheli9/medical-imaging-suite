# Medical Imaging Suite

> Medical Imaging portfolio project — independent open-source implementation.
> This is an original, from-scratch build. It is not affiliated with, and does not
> contain any code, prompts, data, or business logic from, any employer or client.

![status](https://img.shields.io/badge/status-mvp-yellow)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

## 1. Problem

Five related medical-imaging tasks -- CT segmentation, CT keypoint
detection, X-ray segmentation, 2D ROI/keypoint detection, and breast-MRI
tumor segmentation -- each started as its own scaffold-only repo with no
shared UI and no way for someone to just *try* the pipeline without cloning
code and running scripts. This project consolidates all five into one
Django web app: pick a task from a dashboard, upload a scan or image, get a
result back, with real background-job processing behind it.

**Planned 6th feature (not yet ported):** GI-tract segmentation (FPN +
EfficientNet-B3 backbone), consolidated in from the archived
[UG-GI-Track-Segmentation](../../portfolio-archived-repos/UG-GI-Track-Segmentation/)
repo. That repo was a standalone Kaggle notebook, not a Django app, so unlike
the 5 features above it still needs a real `gi_tract_segmentation` app
(model, views, templates, migrations, tests) built to match this suite's
existing pattern before it shows up in the task dashboard. Tracked in
`docs/architecture.md` and `docs/evaluation.md`.

## 2. Architecture

```text
Browser -> Django view -> Job row (queued) -> Celery task ->
  imaging_core (shared I/O/preprocessing) -> feature app's model ->
  Job.result (metrics + PNG) -> Browser (polls until done)
```

See `docs/architecture.md` for the full component breakdown and design
rationale (registry pattern, why every feature trains its own small model,
atomicity of job output).

## 3. Technology Stack

- Python, Django 5.2, Celery + Redis (background jobs), gunicorn
- PyTorch (CPU), a small U-Net (2D and 3D) per segmentation task, Ultralytics
  YOLO for ROI detection
- nibabel + pydicom (NIfTI/DICOM I/O), OpenCV + Pillow (2D images), SciPy
  (resampling), Matplotlib (visualization, headless Agg backend)
- Server-rendered templates, one token-based stylesheet, no JS build step
- sqlite by default; Postgres via `DATABASE_URL` for a real deployment

## 4. Feature List

- **3D CT Segmentation** -- 3D U-Net binary segmentation on a CT volume
  (NIfTI or a `.zip` of a DICOM series).
- **3D CT Keypoint Detection** -- heatmap-regression landmark localization
  on a CT volume.
- **2D X-Ray Segmentation** -- 2D U-Net segmentation on an X-ray image.
- **2D ROI / Keypoint Detection** -- YOLO region-of-interest detection on a
  2D medical image (first stage of a detect-then-localize pipeline).
- **3D Breast Tumor Segmentation** -- 3D U-Net tumor segmentation on a
  breast MRI volume.
- Shared: user accounts, per-user job history, background job processing,
  atomic result writes, orthogonal-slice / overlay visualizations.

Every demo model is trained only on synthetic data the first time it's
needed (see `docs/architecture.md`/`docs/evaluation.md`) -- each task page
and result summary says so explicitly. This is an inference-demo pipeline,
not a diagnostic tool.

## 5. Implementation Plan

1. Phase 1: Django scaffold + `imaging_core` (I/O, preprocessing, registry,
   visualization) + `jobs` app (model, Celery task, views/templates) --
   **done**.
2. Phase 2: `ct_segmentation` ported end-to-end (most complete source
   repo) -- **done**.
3. Phase 3: `roi_keypoints` (YOLO) and `breast_tumor_segmentation` ported
   -- **done**.
4. Phase 4: `xray_segmentation` and `ct_keypoints` built as new MVP
   features (their source repos had no trained model yet) -- **done**.
5. Phase 5: `docker-compose.yml`, CI, docs, `PORTFOLIO_INDEX.md` update --
   **done**.
6. Future: swap synthetic-data checkpoints for real-dataset-trained ones;
   real DICOM-series browser upload UX polish; admin-configurable model
   variants per task.

## 6. Repository Structure

```text
medical-imaging-suite/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── .env.example
├── manage.py
├── config/                # settings, urls, celery app
├── imaging_core/          # shared I/O, preprocessing, registry, visualize
├── jobs/                  # Job model, Celery task, dashboard/job views
├── accounts/               # signup view
├── ct_segmentation/        # feature app: model.py + task.py
├── ct_keypoints/
├── xray_segmentation/
├── roi_keypoints/
├── breast_tumor_segmentation/
├── templates/ + static/imaging/style.css
├── docker/                 # Dockerfile, docker-compose.yml
├── docs/
│   ├── architecture.md
│   └── evaluation.md
├── tests/  (per-app tests.py; this dir holds cross-cutting fixtures if any)
├── configs/                 # e.g. breast_tumor_data.yaml manifest
└── .github/workflows/ci.yml
```

## 7. Setup

```bash
git clone <this-repo-url>
cd medical-imaging-suite
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
python manage.py migrate
python manage.py createsuperuser   # optional, for /admin/
python manage.py runserver
```

Then open http://127.0.0.1:8000/, sign up, and pick a task.

## 8. Dataset

**Synthetic by default**, per feature -- see `docs/architecture.md`. No
real public medical dataset is bundled with this repo. Uploading a real
scan/image runs real inference through a synthetic-data-trained model (see
each task's on-page description); no ground truth is uploaded alongside it,
so results report prediction statistics, not accuracy metrics.

`breast_tumor_segmentation` additionally ships a manifest-driven, real
dataset acquisition path (`dataset_download.py`, ported from
`3d-breast-tumor-segmentation`):

```bash
python manage.py download_breast_mri_dataset --config configs/breast_tumor_data.yaml
```

## 9. Training / Execution

Each feature app's `model.py` exposes `train_on_synthetic_data(...)` and
`get_or_train_model(checkpoint_dir, ...)` directly (importable from a shell
or a script) if you want to retrain outside the web flow. The web flow
itself trains-and-caches on first use automatically -- no separate command
needed to "prime" a feature before using it from the UI.

## 10. Evaluation

See `docs/evaluation.md` for the per-task metric definitions and a Result
Log to fill in as real datasets are evaluated against.

## 11. Results

Every feature runs end-to-end on CPU: upload -> background job -> result
with metrics + a visualization PNG. See `docs/evaluation.md` for the
training-time synthetic-data numbers ported from `3d-ct-segmentation`
(val_dice ~0.97 in 5 epochs) and the honest caveat that repeats across
every task: not validated on real anatomy.

## 12. API

No REST API is exposed in this MVP -- everything is server-rendered Django
views (`jobs/urls.py`). A `Job` and its `result` JSON are visible at
`/jobs/<id>/`; `/jobs/<id>/status.json` is a small polling endpoint the job
page itself uses.

## 13. Docker

```bash
cd docker
docker compose up --build
```

Brings up `redis`, `web` (gunicorn, runs migrations on start), and `worker`
(Celery). See `docker-compose.yml` for environment variables (set
`DJANGO_SECRET_KEY` for anything beyond local testing).

## 14. Tests

```bash
pytest
```

39 tests as of this MVP: `imaging_core` I/O/preprocessing unit tests, one
registry contract test, `jobs` app tests (job creation, atomic
success/failure output, view access control, upload validation), and a
synthetic-data-generation + tiny-train + task-end-to-end test per feature
app.

## 15. Limitations

- This is a from-scratch, independent recreation built for portfolio
  purposes, consolidating five smaller scaffold repos (`3d-ct-segmentation`,
  `3d-medical-keypoint-detection`, `xray-image-segmentation`,
  `medical-keypoint-detection`, `3d-breast-tumor-segmentation`), which
  remain as their own standalone repos, unmodified.
- No feature's model has been trained/evaluated on a real public dataset in
  this repo -- every demo checkpoint is trained on synthetic data (see
  `docs/architecture.md`). Treat every result as a pipeline demo, never a
  diagnostic output.
- Large real scans are resized down to each model's small, fixed,
  CPU-friendly input shape (e.g. 32x32x32) on upload, so fine anatomical
  detail from a real full-resolution scan is lost.
- `roi_keypoints`' YOLO detector trains from a from-scratch architecture
  (`yolov8n.yaml`, not internet-pretrained weights) on very little synthetic
  data, so detection quality on a real image is weak -- this favors an
  offline-safe demo over a stronger, pretrained-weight-dependent one.
- sqlite is the default DB; for a real multi-container deployment, set
  `DATABASE_URL` to Postgres (sqlite has no real concurrent-writer story
  across the `web` and `worker` containers).

## 16. Future Work

- Train each feature's checkpoint on a real public dataset (the datasets
  named in the original five repos' READMEs: Medical Segmentation Decathlon
  / TotalSegmentator, VinDr-CXR, breast-MRI sets) and swap in the resulting
  checkpoint.
- Admin-configurable model variants per task (the registry already supports
  registering more than one task per "kind" of problem).
- Track open items as GitHub Issues.

## 17. Disclosure

This repository is an **independent open-source recreation inspired by the kind of
production systems I have worked on professionally**. It contains no employer or
client source code, prompts, datasets, credentials, architecture diagrams, or
business logic. All code, data, and documentation here are original or built on
publicly available datasets and open-source tools.

---
_Last updated: 2026-09-12_
