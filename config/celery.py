"""Celery application for medical-imaging-suite.

Task modules self-register via each feature app's ``apps.py::ready()``
(mirrors the string-keyed registry pattern in ``imaging_core.registry``), and
Celery autodiscovers ``tasks.py`` in every installed app.
"""
from __future__ import annotations

import os

from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("medical_imaging_suite")
# All CELERY_* settings (including whether a broker is configured, and
# task_always_eager) live in config/settings.py -- see the "Background jobs"
# section there. Nothing here reads the environment directly, so behavior is
# entirely determined by settings, not by import order.
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
