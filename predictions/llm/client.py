"""NVIDIA NIM OpenAI-compatible client factory."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from django.conf import settings


@lru_cache(maxsize=1)
def get_nim_client():
    """Return an OpenAI client pointed at NVIDIA NIM, or None if disabled."""
    api_key = getattr(settings, 'NVIDIA_NIM_API_KEY', '') or ''
    if not api_key:
        return None
    from openai import OpenAI

    base_url = getattr(
        settings,
        'NVIDIA_NIM_BASE_URL',
        'https://integrate.api.nvidia.com/v1',
    )
    return OpenAI(base_url=base_url, api_key=api_key)


def get_nim_model() -> str:
    return getattr(
        settings,
        'NVIDIA_NIM_MODEL',
        'nvidia/nemotron-3-ultra-550b-a55b',
    )


def is_nim_available() -> bool:
    return get_nim_client() is not None
