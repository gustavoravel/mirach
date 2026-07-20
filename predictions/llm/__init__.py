"""NVIDIA NIM agentic layer for Mirach."""

from django.conf import settings

NIM_ENABLED = bool(getattr(settings, 'NVIDIA_NIM_API_KEY', '') or '')

__all__ = [
    'NIM_ENABLED',
]
