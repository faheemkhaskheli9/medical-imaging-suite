from django.apps import AppConfig


class BreastTumorSegmentationConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "breast_tumor_segmentation"

    def ready(self):
        from . import task  # noqa: F401 - import fires @register_task
