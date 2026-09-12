"""
Django settings for medical-imaging-suite.

Every environment-derived value is resolved here, once, at settings-import
time -- feature apps and imaging_core never read `os.environ` themselves
(rule: import must be pure; hidden env reads elsewhere would make behavior
depend on import order instead of this one file).
"""
from __future__ import annotations

from pathlib import Path

from decouple import Csv, config

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = config("DJANGO_SECRET_KEY", default="dev-only-insecure-secret-key")
DEBUG = config("DJANGO_DEBUG", default=True, cast=bool)
ALLOWED_HOSTS = config("DJANGO_ALLOWED_HOSTS", default="localhost,127.0.0.1", cast=Csv())

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "widget_tweaks",
    # Shared job/registry infra
    "jobs",
    # Feature apps -- each registers one imaging_core task in apps.py::ready()
    "ct_segmentation",
    "ct_keypoints",
    "xray_segmentation",
    "roi_keypoints",
    "breast_tumor_segmentation",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "jobs.context_processors.task_registry",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

# --- Database ---------------------------------------------------------------
# Lenient default (sqlite, no setup) vs. strict explicit input: an explicitly
# set DATABASE_URL that Django can't parse/reach should fail loudly at
# startup, not silently fall back to sqlite.
_database_url = config("DATABASE_URL", default="")
if _database_url:
    import dj_database_url  # noqa: PLC0415 - only needed on this path

    DATABASES = {"default": dj_database_url.parse(_database_url, conn_max_age=600)}
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "jobs:dashboard"
LOGOUT_REDIRECT_URL = "jobs:dashboard"

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --- Uploads ------------------------------------------------------------
# Reject anything absurd before it touches disk; per-task forms narrow this
# further (e.g. only .nii/.nii.gz/.dcm/.zip for 3D tasks).
MAX_UPLOAD_SIZE_BYTES = config("MAX_UPLOAD_SIZE_BYTES", default=200 * 1024 * 1024, cast=int)
DATA_UPLOAD_MAX_MEMORY_SIZE = MAX_UPLOAD_SIZE_BYTES
FILE_UPLOAD_MAX_MEMORY_SIZE = 5 * 1024 * 1024  # stream anything bigger straight to disk

# --- Background jobs (Celery) -----------------------------------------------
# No broker configured -> tasks run synchronously in the web process (fine
# for local dev/tests, and for `docker run` without docker-compose). Set
# CELERY_BROKER_URL (docker-compose sets it to the `redis` service) to get
# real async processing via a separate worker.
CELERY_BROKER_URL = config("CELERY_BROKER_URL", default="")
CELERY_RESULT_BACKEND = CELERY_BROKER_URL or None
CELERY_TASK_ALWAYS_EAGER = not bool(CELERY_BROKER_URL)
CELERY_TASK_EAGER_PROPAGATES = True
CELERY_TASK_TRACK_STARTED = True
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"

# --- Torch / model settings --------------------------------------------------
# CPU-only by default across every feature app's model.py -- see each
# app's README section. Override per-deployment if a CUDA image is built.
IMAGING_DEVICE = config("IMAGING_DEVICE", default="cpu")

# Where each feature app caches its lazily-trained demo checkpoint (outside
# MEDIA_ROOT: these are model weights, not job output, and must not be
# served/browsable the way job results are).
CHECKPOINT_DIR = BASE_DIR / "var" / "checkpoints"
