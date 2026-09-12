from django.apps import AppConfig


class XraySegmentationConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "xray_segmentation"

    def ready(self):
        from . import task  # noqa: F401 - import fires @register_task
