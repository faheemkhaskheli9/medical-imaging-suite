"""Manifest-driven breast-MRI dataset acquisition.

Wraps ``breast_tumor_segmentation.dataset_download`` (ported from
``3d-breast-tumor-segmentation``) as a management command so real dataset
acquisition -- separate from the web demo's synthetic-data-trained model --
can be run the same way any of this project's management commands are.

    python manage.py download_breast_mri_dataset --config configs/data.yaml
"""
from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from breast_tumor_segmentation.dataset_download import DatasetError, download_dataset, load_manifest


class Command(BaseCommand):
    help = "Acquire (or verify) the breast-MRI dataset described by a manifest YAML file."

    def add_arguments(self, parser):
        parser.add_argument(
            "--config", default=None, help="Path to a dataset manifest YAML (default: configs/data.yaml)"
        )
        parser.add_argument("--force", action="store_true", help="Re-acquire even if already complete")

    def handle(self, *args, **options):
        config_path = options["config"]
        try:
            manifest = load_manifest(Path(config_path) if config_path else None)
            report = download_dataset(manifest, force=options["force"])
        except DatasetError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(
            self.style.SUCCESS(
                f"Dataset {manifest.name!r} ({manifest.mode} mode): {report.num_cases} case(s) at {report.root} "
                f"({'already complete' if report.was_complete else 'freshly acquired'})"
            )
        )
