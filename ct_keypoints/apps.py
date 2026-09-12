from django.apps import AppConfig


class CtKeypointsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "ct_keypoints"

    def ready(self):
        from . import task  # noqa: F401 - import fires @register_task
