try:
    from .celery import app as celery_app  # type: ignore
    __all__ = ("celery_app",)
except Exception:
    # Allow project to import even if Celery not yet set up during initial migrations
    pass

