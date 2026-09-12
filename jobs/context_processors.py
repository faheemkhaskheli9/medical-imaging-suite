from imaging_core.registry import all_tasks


def task_registry(request):
    """Exposes the full task catalog to every template (nav dropdown, etc.)
    without every view having to pass it explicitly."""
    return {"nav_imaging_tasks": all_tasks()}
