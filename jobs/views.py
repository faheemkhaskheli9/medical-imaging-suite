from __future__ import annotations

from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.http import Http404, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render

from imaging_core.registry import all_tasks, get_task, is_registered

from .forms import JobSubmitForm
from .models import Job
from .tasks import run_imaging_job

JOBS_PAGE_SIZE = 25


def dashboard(request):
    """The task catalog -- one card per registered imaging feature."""
    return render(request, "jobs/dashboard.html", {"tasks": all_tasks()})


def task_detail(request, task_key: str):
    if not is_registered(task_key):
        raise Http404(f"No imaging task registered as {task_key!r}")
    task = get_task(task_key)

    if request.method == "POST":
        if not request.user.is_authenticated:
            return redirect(f"/accounts/login/?next={request.path}")
        form = JobSubmitForm(request.POST, request.FILES, task=task)
        if form.is_valid():
            # Two-step save: job_upload_path needs instance.pk, which only
            # exists after the row is inserted once.
            job = Job.objects.create(owner=request.user, task_key=task.key, params={})
            job.input_file = form.cleaned_data["input_file"]
            job.save(update_fields=["input_file"])
            run_imaging_job.delay(job.pk)
            return redirect(job.get_absolute_url())
    else:
        form = JobSubmitForm(task=task)

    recent_jobs = []
    if request.user.is_authenticated:
        recent_jobs = Job.objects.filter(owner=request.user, task_key=task.key)[:5]

    return render(
        request,
        "jobs/task_detail.html",
        {"task": task, "form": form, "recent_jobs": recent_jobs},
    )


@login_required
def job_detail(request, pk: int):
    job = get_object_or_404(Job, pk=pk)
    if job.owner_id != request.user.id and not request.user.is_staff:
        raise Http404("Job not found")
    task = get_task(job.task_key) if is_registered(job.task_key) else None
    return render(request, "jobs/job_detail.html", {"job": job, "task": task})


@login_required
def job_status_json(request, pk: int):
    """Polled by job_detail.html's auto-refresh while a job is queued/running."""
    job = get_object_or_404(Job, pk=pk)
    if job.owner_id != request.user.id and not request.user.is_staff:
        raise Http404("Job not found")
    return JsonResponse({"status": job.status, "is_finished": job.is_finished})


@login_required
def job_list(request):
    show_all = request.user.is_staff and request.GET.get("all") == "1"
    jobs_qs = Job.objects.all() if show_all else Job.objects.filter(owner=request.user)
    jobs_qs = jobs_qs.select_related("owner")

    page = Paginator(jobs_qs, JOBS_PAGE_SIZE).get_page(request.GET.get("page"))
    query = request.GET.copy()
    query.pop("page", None)

    return render(
        request,
        "jobs/job_list.html",
        {"page": page, "query": query.urlencode(), "show_all": show_all},
    )
