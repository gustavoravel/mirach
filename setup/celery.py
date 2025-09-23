import os

from celery import Celery


os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'setup.settings')

app = Celery('mirach')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
app.config_from_object('django.conf:settings', namespace='CELERY')

# Auto-discover tasks in installed apps
app.autodiscover_tasks()


@app.task(bind=True)
def debug_task(self):  # pragma: no cover
    return {'request_id': self.request.id}


