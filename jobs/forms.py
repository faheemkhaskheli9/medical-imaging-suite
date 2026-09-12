from __future__ import annotations

from django import forms
from django.conf import settings

from imaging_core.registry import BaseImagingTask


class JobSubmitForm(forms.Form):
    input_file = forms.FileField(help_text="")

    def __init__(self, *args, task: BaseImagingTask, **kwargs):
        self.task = task
        super().__init__(*args, **kwargs)
        self.fields["input_file"].help_text = (
            f"Accepted: {', '.join(task.accepted_extensions)}"
        )

    def clean_input_file(self):
        uploaded = self.cleaned_data["input_file"]
        name = uploaded.name.lower()
        if not any(name.endswith(ext) for ext in self.task.accepted_extensions):
            raise forms.ValidationError(
                f"Unsupported file type for {self.task.label}. "
                f"Expected one of: {', '.join(self.task.accepted_extensions)}"
            )
        if uploaded.size > settings.MAX_UPLOAD_SIZE_BYTES:
            max_mb = settings.MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)
            raise forms.ValidationError(f"File too large; max size is {max_mb:.0f} MB.")
        return uploaded
