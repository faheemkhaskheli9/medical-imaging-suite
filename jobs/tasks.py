"""The one Celery task that runs any registered imaging task.

Shared code path: this is the *only* place ``imaging_core.registry.get_task``
is invoked to actually process a job -- there is no separate "run inline"
code path to keep in sync (the management-command-vs-task duplication rule).
Whether this runs in-process (``CELERY_TASK_ALWAYS_EAGER=True``, the default
with no broker configured) or on a separate worker is entirely a settings
concern; this function behaves identically either way.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from celery import shared_task
from django.conf import settings
from django.utils import timezone

from imaging_core.registry import get_task


@shared_task(bind=True)
def run_imaging_job(self, job_id: int) -> None:
    from jobs.models import Job  # local import: avoids a hard import-order

    # dependency between the celery app and the jobs app at module-load time.

    job = Job.objects.get(pk=job_id)
    job.status = Job.Status.RUNNING
    job.started_at = timezone.now()
    job.save(update_fields=["status", "started_at", "updated_at"])

    media_root = Path(settings.MEDIA_ROOT)
    job_dir = media_root / "jobs" / str(job.pk)
    tmp_output_dir = job_dir / f".tmp_output_{self.request.id or 'sync'}"
    final_output_dir = job_dir / "output"

    try:
        task = get_task(job.task_key)
        tmp_output_dir.mkdir(parents=True, exist_ok=True)
        result = task.run(Path(job.input_file.path), tmp_output_dir, job.params or {})

        # Atomic publish: the tmp dir only replaces the final one once run()
        # has fully succeeded, so a crash mid-run (worker killed, OOM, an
        # unhandled exception partway through) never leaves the UI treating
        # a half-written result directory as "done".
        if final_output_dir.exists():
            shutil.rmtree(final_output_dir)
        tmp_output_dir.rename(final_output_dir)

        job.result = {
            "metrics": result.metrics,
            "visualization_paths": result.visualization_paths,
            "summary": result.summary,
        }
        job.output_dir = str(final_output_dir.relative_to(media_root)).replace("\\", "/")
        job.status = Job.Status.DONE
    except Exception as exc:  # noqa: BLE001 - any task failure becomes a Job
        # status, never a silently-stuck "running" row or a crashed worker.
        job.status = Job.Status.FAILED
        job.error = str(exc)
        shutil.rmtree(tmp_output_dir, ignore_errors=True)
    finally:
        job.finished_at = timezone.now()
        job.save(update_fields=["status", "result", "output_dir", "error", "finished_at", "updated_at"])
