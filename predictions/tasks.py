from __future__ import annotations

from celery import shared_task
from celery.utils.log import get_task_logger
from django.conf import settings

logger = get_task_logger(__name__)


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
    result = service.run_prediction(prediction)

    if result.get('success'):
        prediction.refresh_from_db()
        prediction.progress = 90
        prediction.save(update_fields=['progress'])
        # Narrative insights (non-fatal)
        try:
            from .llm.narrative_agent import generate_insights
            insights = generate_insights(prediction)
            explain = dict(prediction.explainability or {})
            explain['ai_insights'] = insights
            prediction.explainability = explain
            prediction.save(update_fields=['explainability'])
        except Exception as exc:
            logger.warning("NarrativeAgent failed for prediction %s: %s", prediction_id, exc)
        prediction.progress = 100
        prediction.estimated_completion = timezone.now()
        prediction.save(update_fields=['progress', 'estimated_completion'])
    return {
        'prediction_id': prediction_id,
        'status': 'completed' if result.get('success') else 'failed',
        'result': result,
    }


@shared_task(bind=True)
def run_model_championship_task(self, dataset_id: int, candidates=None) -> dict:
    """Celery task: empirical model championship for a dataset."""
    from datasets.models import Dataset
    from .services import PredictionService
    from .championship import ModelChampionship

    dataset = Dataset.objects.get(pk=dataset_id)
    service = PredictionService()
    series, exog = service.prepare_data(dataset)
    champ = ModelChampionship()
    result = champ.run(series, exog=exog, candidates=candidates)
    return {
        'dataset_id': dataset_id,
        'success': True,
        'result': result,
    }


@shared_task(bind=True)
def run_auto_model_plan_task(self, dataset_id: int) -> dict:
    """Celery task: OrchestratorAgent produces ModelPlan for Auto mode."""
    from .llm.orchestrator_agent import plan_model

    plan = plan_model(dataset_id)
    return {
        'dataset_id': dataset_id,
        'success': True,
        'plan': plan,
    }
