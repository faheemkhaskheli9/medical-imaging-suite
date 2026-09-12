from django.apps import AppConfig


class CtSegmentationConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "ct_segmentation"

    def ready(self):
        from . import task  # noqa: F401 - import fires @register_task
