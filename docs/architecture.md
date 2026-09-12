# Architecture

## Pipeline

```text
Browser --> Django views/templates --> Job (row created, status=queued)
                                          |
                                    Celery task dispatch
                                    (jobs.tasks.run_imaging_job)
                                          |
                          imaging_core.registry.get_task(job.task_key)
                                          |
                imaging_core.io / imaging_core.preprocessing (shared)
                                          |
                     feature app's model.py (torch model, lazily
                     trained on synthetic data and cached on first use)
                                          |
                    Job.result (metrics JSON) + PNG(s) under
                    MEDIA_ROOT/jobs/<id>/output/ (atomic rename)
                                          |
                                Browser polls job status page
```

## Component list

- **config/** -- Django settings (all environment-derived values resolved
  once here), URL root, Celery app definition.
- **imaging_core/** -- shared, framework-agnostic code with no Django
  models: volume/image I/O (NIfTI, DICOM series, DICOM-series-as-zip
  upload, 2D images), preprocessing (resample, HU/zscore/minmax/percentile
  normalization, crop/pad, whole-volume resize), the task registry (`
  BaseImagingTask` ABC + `@register_task`), and shared visualization
  (orthogonal-slice PNGs, keypoint-marker PNGs, 2D overlay PNGs).
- **jobs/** -- the one cross-cutting Django app: `Job` model (the
  unconditional audit row for every submission), the one Celery task that
  runs any registered imaging task (`jobs/tasks.py::run_imaging_job`), and
  the dashboard/task-detail/job-detail/job-list views + templates.
- **Five feature apps** (`ct_segmentation`, `ct_keypoints`,
  `xray_segmentation`, `roi_keypoints`, `breast_tumor_segmentation`) --
  each owns one `model.py` (architecture + synthetic-data training +
  inference) and one `task.py` (a `BaseImagingTask` subclass wiring that
  model to `imaging_core`). Each self-registers via `apps.py::ready()`
  importing `task.py`, so `jobs` never imports a feature app directly --
  adding a 6th feature is "new app, subclass, register, add to
  `INSTALLED_APPS`."
- **accounts/** -- signup view (Django's built-in `LoginView`/`LogoutView`
  handle the rest).
- **templates/ + static/imaging/style.css** -- one token-based stylesheet,
  server-rendered templates, no JS build step (one small inline poll script
  on the job-status page).

## Why a registry instead of one view per feature

Mirrors the `BaseStrategy` / `@register_strategy("key")` pattern from a
sibling Django project: the `Job.task_key` field stores a string, never a
Python import path, so a job row can never point at arbitrary code, and the
dashboard/task-detail views are generic over whatever is registered rather
than needing a new `if/elif` per feature.

## Why every feature trains its own small model on synthetic data

None of the five source repos this project consolidates had a real public
dataset trained/evaluated end-to-end (see each original repo's README
Limitations section). Rather than serve nothing, each feature app trains a
small, CPU-fast model on a synthetic generator the first time it's used and
caches the checkpoint (`var/checkpoints/`, gitignored) -- so the web demo is
always runnable end-to-end, honestly labeled (every task's `description` and
result `summary` say so) as not validated on real anatomy. Swapping in a
real trained checkpoint later is a matter of replacing the cached file with
one trained via each app's `model.py` functions against a real dataset (see
`breast_tumor_segmentation/dataset_download.py` for the one feature that
already has a real-dataset acquisition path).

## Background jobs

No separate "sync" and "async" code path: `jobs/tasks.py::run_imaging_job`
is the only place a registered task's `run()` is invoked, whether Celery
executes it inline (`CELERY_TASK_ALWAYS_EAGER=True`, the default with no
`CELERY_BROKER_URL` configured -- local dev, tests, `docker run` without
compose) or on a separate worker process (`docker-compose.yml`, with Redis
as the broker).

## Atomicity

A task's `run()` writes to a fresh temp directory; only once it returns
successfully does `run_imaging_job` rename that directory into
`MEDIA_ROOT/jobs/<id>/output/` (same filesystem, so the rename is atomic).
A worker crash or unhandled exception mid-run never leaves the UI treating a
partial result as done -- the temp directory is discarded and `Job.status`
is set to `failed` with the exception message attached.
