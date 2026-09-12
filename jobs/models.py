from __future__ import annotations

from django.conf import settings
from django.db import models
from django.urls import reverse


def job_upload_path(instance: "Job", filename: str) -> str:
    # instance.pk is set by the time Django saves the FileField (the row is
    # inserted first via Job.objects.create(...) before the file is attached
    # in the view -- see jobs/views.py:submit_job), so every upload lives
    # under its own job's directory from the start.
    return f"jobs/{instance.pk}/input/{filename}"


class Job(models.Model):
    """One run of one registered imaging task -- the unconditional audit row
    for every submission, whether it ultimately succeeds or fails."""

    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        RUNNING = "running", "Running"
        DONE = "done", "Done"
        FAILED = "failed", "Failed"

    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="jobs")
    # References imaging_core.registry's string key, never a Python path.
    task_key = models.CharField(max_length=64, db_index=True)
    status = models.CharField(max_length=16, choices=Status.choices, default=Status.QUEUED, db_index=True)

    # blank=True at the model level only: the two-step create-then-attach
    # save in jobs/views.py needs an empty FileField to be valid for the
    # initial insert (job_upload_path needs instance.pk, which doesn't exist
    # until after that insert). The upload form itself requires a file.
    input_file = models.FileField(upload_to=job_upload_path, blank=True)
    params = models.JSONField(default=dict, blank=True)

    result = models.JSONField(null=True, blank=True)
    output_dir = models.CharField(max_length=255, blank=True)  # relative to MEDIA_ROOT, set on success
    error = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"Job #{self.pk} [{self.task_key}] ({self.status})"

    def get_absolute_url(self) -> str:
        return reverse("jobs:job_detail", args=[self.pk])

    @property
    def is_finished(self) -> bool:
        return self.status in (self.Status.DONE, self.Status.FAILED)

    @property
    def visualization_urls(self) -> list[str]:
        """Full, browser-usable URLs to this job's visualization PNGs, if any."""
        if not self.result or not self.output_dir:
            return []
        rel_paths = self.result.get("visualization_paths") or []
        return [f"{settings.MEDIA_URL}{self.output_dir}/{p}" for p in rel_paths]
