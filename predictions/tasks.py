from __future__ import annotations

from celery import shared_task


@shared_task(bind=True)
def run_prediction_task(self, prediction_id: int) -> dict:
    # Lazy import to avoid heavy deps at worker boot if not needed
    from .services import run_prediction_by_id

    result = run_prediction_by_id(prediction_id)
    return {
        'prediction_id': prediction_id,
        'status': 'completed',
        'result': result,
    }


