from __future__ import annotations

from celery import shared_task
from celery.utils.log import get_task_logger
from django.conf import settings


@shared_task(bind=True)
def run_prediction_task(self, prediction_id: int) -> dict:
    # Lazy import to avoid heavy deps at worker boot if not needed
    from .models import Prediction
    from .services import PredictionService
    from django.utils import timezone

    prediction = Prediction.objects.get(id=prediction_id)
    prediction.status = 'training'
    prediction.progress = 10
    prediction.save(update_fields=['status', 'progress'])

    service = PredictionService()
    # Route by priority (example hook; actual queue selection done by Celery routing)
    result = service.run_prediction(prediction)

    if result.get('success'):
        prediction.progress = 100
        prediction.estimated_completion = timezone.now()
        prediction.save(update_fields=['progress', 'estimated_completion'])
    return {
        'prediction_id': prediction_id,
        'status': 'completed',
        'result': result,
    }


