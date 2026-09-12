from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from django.conf import settings
from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from imaging_core.registry import (
    BaseImagingTask,
    ImagingTaskError,
    InputKind,
    TaskResult,
    _REGISTRY,
    all_tasks,
    get_task,
    register_task,
)

from .models import Job
from .tasks import run_imaging_job


class _StubTask(BaseImagingTask):
    """A fast, deterministic fake task registered only for these tests."""

    key = "test_stub"
    label = "Test Stub"
    description = "stub"
    input_kind = InputKind.IMAGE_2D
    accepted_extensions = (".txt",)

    def __init__(self, fail: bool = False):
        self.fail = fail

    def run(self, input_path: Path, output_dir: Path, params: dict) -> TaskResult:
        if self.fail:
            raise ImagingTaskError("stub task always fails")
        (output_dir / "note.txt").write_text("ok", encoding="utf-8")
        return TaskResult(metrics={"echo": params.get("echo", "none")}, visualization_paths=[], summary="stub done")


def _register_stub(fail: bool = False) -> None:
    _REGISTRY.pop("test_stub", None)

    class _Wrapped(_StubTask):
        def __init__(self):
            super().__init__(fail=fail)

    register_task(_Wrapped)


class RegistryTests(TestCase):
    def test_all_five_feature_tasks_are_registered(self):
        keys = {t.key for t in all_tasks()}
        self.assertEqual(
            keys,
            {
                "ct_segmentation",
                "ct_keypoints",
                "xray_segmentation",
                "roi_keypoints",
                "breast_tumor_segmentation",
            },
        )

    def test_duplicate_key_registration_raises(self):
        with self.assertRaises(ValueError):

            @register_task
            class _Dup(BaseImagingTask):
                key = "ct_segmentation"
                label = "dup"
                description = ""
                input_kind = InputKind.IMAGE_2D
                accepted_extensions = ()

                def run(self, input_path, output_dir, params):
                    raise NotImplementedError


class JobTaskExecutionTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="alice", password="pw12345")
        # Isolate MEDIA_ROOT per test: TestCase rolls back the DB (so Job pks
        # get reused across tests), but never touches the filesystem -- an
        # un-isolated MEDIA_ROOT would let one test's leftover jobs/<pk>/
        # directory make a later test's "no partial output" assertion see a
        # stale file from a different test run under the same pk.
        self._media_root = TemporaryDirectory()
        self._override = override_settings(MEDIA_ROOT=self._media_root.name)
        self._override.enable()
        self.addCleanup(self._override.disable)
        self.addCleanup(self._media_root.cleanup)

    def _make_job(self, task_key: str) -> Job:
        job = Job.objects.create(owner=self.user, task_key=task_key, params={"echo": "hi"})
        job.input_file.save("input.txt", SimpleUploadedFile("input.txt", b"hello"), save=True)
        return job

    def test_successful_run_marks_done_and_writes_atomic_output(self):
        _register_stub(fail=False)
        job = self._make_job("test_stub")

        run_imaging_job(job.pk)  # CELERY_TASK_ALWAYS_EAGER in tests -> runs inline

        job.refresh_from_db()
        self.assertEqual(job.status, Job.Status.DONE)
        self.assertEqual(job.result["metrics"], {"echo": "hi"})
        output_path = Path(settings.MEDIA_ROOT) / job.output_dir / "note.txt"
        self.assertTrue(output_path.exists())
        # No leftover .tmp_output_* directory from the atomic rename.
        job_dir = Path(settings.MEDIA_ROOT) / "jobs" / str(job.pk)
        leftovers = [p for p in job_dir.iterdir() if p.name.startswith(".tmp_output_")]
        self.assertEqual(leftovers, [])

    def test_failed_run_marks_failed_with_error_and_no_partial_output(self):
        _register_stub(fail=True)
        job = self._make_job("test_stub")

        run_imaging_job(job.pk)

        job.refresh_from_db()
        self.assertEqual(job.status, Job.Status.FAILED)
        self.assertIn("stub task always fails", job.error)
        self.assertEqual(job.output_dir, "")
        job_dir = Path(settings.MEDIA_ROOT) / "jobs" / str(job.pk)
        final_output = job_dir / "output"
        self.assertFalse(final_output.exists())


class ViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="bob", password="pw12345")
        self._media_root = TemporaryDirectory()
        self._override = override_settings(MEDIA_ROOT=self._media_root.name)
        self._override.enable()
        self.addCleanup(self._override.disable)
        self.addCleanup(self._media_root.cleanup)

    def test_dashboard_lists_all_registered_tasks(self):
        response = self.client.get(reverse("jobs:dashboard"))
        self.assertEqual(response.status_code, 200)
        for key in ("ct_segmentation", "xray_segmentation", "roi_keypoints"):
            self.assertContains(response, key)

    def test_task_detail_requires_login_to_submit(self):
        _register_stub(fail=False)
        response = self.client.post(
            reverse("jobs:task_detail", args=["test_stub"]),
            {"input_file": SimpleUploadedFile("input.txt", b"data")},
        )
        self.assertEqual(response.status_code, 302)
        self.assertIn("/accounts/login/", response.url)

    def test_authenticated_submit_creates_job_and_redirects_to_it(self):
        _register_stub(fail=False)
        self.client.login(username="bob", password="pw12345")
        response = self.client.post(
            reverse("jobs:task_detail", args=["test_stub"]),
            {"input_file": SimpleUploadedFile("input.txt", b"data")},
        )
        self.assertEqual(response.status_code, 302)
        job = Job.objects.get(owner=self.user, task_key="test_stub")
        self.assertEqual(job.status, Job.Status.DONE)  # eager mode runs synchronously
        self.assertRedirects(response, job.get_absolute_url())

    def test_rejects_unsupported_extension(self):
        _register_stub(fail=False)
        self.client.login(username="bob", password="pw12345")
        response = self.client.post(
            reverse("jobs:task_detail", args=["test_stub"]),
            {"input_file": SimpleUploadedFile("input.png", b"data")},
        )
        self.assertEqual(response.status_code, 200)  # form re-rendered with errors
        self.assertFalse(Job.objects.filter(owner=self.user).exists())

    def test_job_detail_is_scoped_to_owner(self):
        _register_stub(fail=False)
        other = User.objects.create_user(username="carol", password="pw12345")
        job = Job.objects.create(owner=other, task_key="test_stub", params={})
        self.client.login(username="bob", password="pw12345")
        response = self.client.get(job.get_absolute_url())
        self.assertEqual(response.status_code, 404)
