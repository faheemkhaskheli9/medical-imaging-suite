"""String-keyed registry for pluggable imaging tasks.

Mirrors the ``BaseStrategy`` / ``@register_strategy("key")`` pattern used
elsewhere in this author's Django projects: the ``Job`` model stores the
string ``task_key``, never a Python import path, so a job row can't point at
arbitrary code. Each feature app's ``apps.py::ready()`` imports its
``task.py`` so the ``@register_task`` decorator fires -- the registry is only
reliably populated after Django app startup, not at module-import time of
anything that imports ``imaging_core`` alone.

Adding a 6th feature is: new app, subclass ``BaseImagingTask``, decorate it,
add the app to ``INSTALLED_APPS``. Nothing in ``jobs`` changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class InputKind(str, Enum):
    """What kind of file a task's upload form should accept."""

    VOLUME_3D = "volume_3d"   # .nii/.nii.gz, or a .zip of a DICOM series
    IMAGE_2D = "image_2d"     # .png/.jpg/.jpeg


@dataclass
class TaskResult:
    """What a task's ``run()`` hands back to the ``jobs`` app.

    ``metrics`` must be JSON-serializable (it is stored verbatim on
    ``Job.result``). ``visualization_paths`` are paths *relative to the job's
    output directory* (the caller already knows that directory; keeping
    these relative keeps a task from needing to know MEDIA_ROOT).
    """

    metrics: dict[str, Any]
    visualization_paths: list[str] = field(default_factory=list)
    summary: str = ""


class ImagingTaskError(RuntimeError):
    """A task's input was invalid or it failed to produce a result."""


class BaseImagingTask(ABC):
    """Contract every feature app's task must implement.

    Subclasses are instantiated once at registration time and must be cheap
    to construct -- any heavy import (torch, ultralytics, ...) or model
    weight loading happens lazily inside ``run()``, not in ``__init__``, so
    importing this registry (e.g. for the dashboard view) never pays for it.
    """

    key: str
    label: str
    description: str
    input_kind: InputKind
    accepted_extensions: tuple[str, ...]

    _INPUT_KIND_LABELS = {
        InputKind.VOLUME_3D: "3D volume (NIfTI / DICOM .zip)",
        InputKind.IMAGE_2D: "2D image",
    }

    @property
    def input_kind_label(self) -> str:
        return self._INPUT_KIND_LABELS[self.input_kind]

    @abstractmethod
    def run(self, input_path: Path, output_dir: Path, params: dict[str, Any]) -> TaskResult:
        """Run the task on ``input_path``, writing any artifacts under
        ``output_dir`` (already created, empty), and return a ``TaskResult``.

        Must raise ``ImagingTaskError`` (or let a lower-level exception
        propagate) on failure rather than returning a partial/misleading
        result -- ``jobs.tasks`` converts any exception into
        ``Job.status = "failed"`` with the message attached.
        """


_REGISTRY: dict[str, BaseImagingTask] = {}


def register_task(task_cls: type[BaseImagingTask]) -> type[BaseImagingTask]:
    """Class decorator: instantiate ``task_cls`` and register it by ``.key``."""
    instance = task_cls()
    if not instance.key:
        raise ValueError(f"{task_cls.__name__}.key must be a non-empty string")
    if instance.key in _REGISTRY:
        raise ValueError(f"duplicate imaging task key: {instance.key!r}")
    _REGISTRY[instance.key] = instance
    return task_cls


def get_task(key: str) -> BaseImagingTask:
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(f"no imaging task registered for key {key!r}") from None


def all_tasks() -> list[BaseImagingTask]:
    """All registered tasks, sorted by label for stable UI ordering."""
    return sorted(_REGISTRY.values(), key=lambda t: t.label)


def is_registered(key: str) -> bool:
    return key in _REGISTRY
