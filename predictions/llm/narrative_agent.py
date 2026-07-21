"""Narrative agent: insights anchored on real metrics (never invents forecasts)."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .agent import chat_structured
from .client import is_nim_available
from .schemas import InsightReport

logger = logging.getLogger(__name__)


def _forecast_slope(values: List[float]) -> Optional[float]:
    if not values or len(values) < 2:
        return None
    y = np.asarray(values, dtype=float)
    x = np.arange(len(y), dtype=float)
    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return None


def _fallback_insights(context: Dict[str, Any]) -> Dict[str, Any]:
    metrics = context.get('metrics') or {}
    algo = context.get('algorithm') or 'modelo'
    horizon = context.get('horizon')
    slope = context.get('forecast_slope')
    beats = context.get('beats_baseline')
    noun = context.get('forecast_noun') or 'previsão'
    domain = context.get('domain')
    parts = [f"Previsão de {noun} com {algo}"]
    if domain:
        parts.append(f"domínio {domain}")
    if horizon:
        parts.append(f"horizonte {horizon}")
    if metrics.get('rmse') is not None:
        parts.append(f"RMSE={metrics['rmse']:.4g}")
    elif metrics.get('mae') is not None:
        parts.append(f"MAE={metrics['mae']:.4g}")
    else:
        parts.append("sem métricas de validação (conjunto de validação ausente)")
    summary = "; ".join(parts) + "."
    drivers = []
    if slope is not None:
        direction = 'alta' if slope > 0 else ('queda' if slope < 0 else 'estável')
        drivers.append(f"Tendência da {noun} aponta {direction} (slope≈{slope:.4g}).")
    risks = []
    if beats is False:
        risks.append("O modelo não superou o baseline naive no campeonato — interprete com cautela.")
    if not metrics:
        risks.append("Métricas de holdout indisponíveis.")
    next_steps = [
        f"Compare a {noun} com um baseline naive/seasonal naive.",
        "Revise features exógenas mapeadas e qualidade dos gaps temporais.",
    ]
    return InsightReport(
        summary=summary,
        drivers=drivers,
        risks=risks,
        next_steps=next_steps,
    ).model_dump()


def generate_insights(prediction) -> Dict[str, Any]:
    """
    Build InsightReport from completed prediction. Never asks LLM for numeric forecasts.
    """
    preds_data = prediction.predictions_data or {}
    forecast = preds_data.get('forecast') or []
    metrics = prediction.metrics or {}
    explain = prediction.explainability or {}
    fi = explain.get('feature_importances') or {}
    top_features = []
    if isinstance(fi, dict):
        top_features = [
            {'feature': k, 'importance': v}
            for k, v in sorted(fi.items(), key=lambda x: abs(float(x[1])), reverse=True)[:8]
        ]

    domain_meta = None
    try:
        from predictions.domains import resolve_dataset_domain
        domain_meta = resolve_dataset_domain(prediction.dataset)
    except Exception:
        domain_meta = None

    slope = _forecast_slope([float(x) for x in forecast]) if forecast else None
    mean_fc = float(np.mean(forecast)) if forecast else None

    context = {
        'algorithm': prediction.prediction_model.algorithm_type,
        'model_name': prediction.prediction_model.name,
        'horizon': prediction.prediction_horizon,
        'metrics': metrics,
        'forecast_mean': mean_fc,
        'forecast_slope': slope,
        'top_features': top_features,
        'domain': (domain_meta or {}).get('label'),
        'domain_code': (domain_meta or {}).get('code'),
        'target_vocabulary': (domain_meta or {}).get('target_vocabulary'),
        'forecast_noun': (domain_meta or {}).get('forecast_noun'),
        'domain_guidance': (domain_meta or {}).get('agent_guidance'),
        'beats_baseline': explain.get('beats_baseline'),
        'championship': explain.get('championship'),
        # Do NOT send full forecast series to LLM — only aggregates
        'n_forecast_points': len(forecast),
    }

    if not is_nim_available():
        return _fallback_insights(context)

    system = (
        "Você é um analista de séries temporais. Escreva insights em português (pt-BR) "
        "estritamente ancorados nos números fornecidos. "
        "NUNCA invente valores de previsão, métricas ou percentuais que não estejam no contexto. "
        "Personalize a linguagem ao domínio informado (ex.: demanda no varejo, tráfego web, "
        "produção industrial, volume logístico). "
        "Responda só em JSON no schema pedido."
    )
    user = (
        "Contexto da previsão (agregados reais calculados pelo sistema):\n"
        f"{json.dumps(context, ensure_ascii=False, default=str)}\n\n"
        "Produza summary, drivers, risks e next_steps úteis para o usuário de negócio, "
        f"usando o vocabulário de {(domain_meta or {}).get('target_vocabulary') or 'série temporal'}."
    )
    parsed = chat_structured(
        system=system,
        user=user,
        schema=InsightReport,
        temperature=0.4,
        max_tokens=2048,
    )
    if parsed is None:
        return _fallback_insights(context)
    return parsed.model_dump()
