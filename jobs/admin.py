from django.contrib import admin

from .models import Job


@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = ("id", "task_key", "owner", "status", "created_at", "finished_at")
    list_filter = ("task_key", "status")
    search_fields = ("owner__username", "task_key")
    readonly_fields = ("created_at", "updated_at", "started_at", "finished_at")
