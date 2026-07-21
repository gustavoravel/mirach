"""Orchestrator agent: plan model selection via championship tools + NIM."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .agent import tool_calling_loop
from .client import is_nim_available
from .schemas import ModelPlan

logger = logging.getLogger(__name__)


def _tool_schemas() -> List[Dict[str, Any]]:
    return [
        {
            'type': 'function',
            'function': {
                'name': 'get_data_profile',
                'description': 'Retorna o perfil compacto do dataset',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'dataset_id': {'type': 'integer'},
                    },
                    'required': ['dataset_id'],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'get_characteristic_flags',
                'description': 'Flags estatísticas: tendência, sazonalidade, estacionariedade',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'dataset_id': {'type': 'integer'},
                    },
                    'required': ['dataset_id'],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'run_championship',
                'description': 'Executa campeonato walk-forward e retorna ranking empírico',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'dataset_id': {'type': 'integer'},
                        'candidates': {
                            'type': 'array',
                            'items': {'type': 'string'},
                            'description': 'Lista opcional de algoritmos candidatos',
                        },
                    },
                    'required': ['dataset_id'],
                },
            },
        },
    ]


def plan_model(dataset_id: int) -> Dict[str, Any]:
    """
    Return ModelPlan dict. Falls back to championship winner without NIM.
    """
    from datasets.models import Dataset
    from predictions.services import PredictionService
    from predictions.championship import ModelChampionship

    dataset = Dataset.objects.get(pk=dataset_id)
    service = PredictionService()
    series, exog = service.prepare_data(dataset)
    champ = ModelChampionship()
    champ_result = champ.run(series, exog=exog)

    def get_data_profile(args: Dict[str, Any]) -> Any:
        ds = Dataset.objects.get(pk=int(args.get('dataset_id', dataset_id)))
        profile = ds.data_profile or {}
        return {
            'rows': profile.get('rows') or ds.total_rows,
            'columns': profile.get('columns') or ds.column_names,
            'inferred_freq': profile.get('inferred_freq'),
            'warnings': profile.get('warnings'),
            'domain': (ds.ai_interpretation or {}).get('domain_label')
                or (ds.ai_interpretation or {}).get('inferred_domain'),
            'domain_code': (ds.ai_interpretation or {}).get('domain_code'),
            'has_exog': exog is not None,
            'n_points': len(series),
        }

    def get_characteristic_flags(args: Dict[str, Any]) -> Any:
        return {
            'has_trend': service._detect_trend(series),
            'has_seasonality': service._detect_seasonality(series),
            'is_stationary': service._is_stationary(series),
            'length': len(series),
        }

    def run_championship(args: Dict[str, Any]) -> Any:
        cands = args.get('candidates')
        result = champ.run(series, exog=exog, candidates=cands)
        # Compact ranking for tokens
        ranking = [
            {
                'algorithm': r['algorithm'],
                'rmse': r['rmse'] if r['rmse'] != float('inf') else None,
                'mae': r['mae'] if r['mae'] != float('inf') else None,
                'smape': r.get('smape'),
                'error': r.get('error'),
            }
            for r in result.get('ranking', [])[:8]
        ]
        return {
            'best_model': result.get('best_model'),
            'beats_baseline': result.get('beats_baseline'),
            'ranking': ranking,
        }

    handlers = {
        'get_data_profile': get_data_profile,
        'get_characteristic_flags': get_characteristic_flags,
        'run_championship': run_championship,
    }

    fallback = ModelPlan(
        algorithm=champ_result.get('best_model') or 'arima',
        parameters={},
        transforms=[],
        rationale='Vencedor do campeonato walk-forward (fallback determinístico).',
        candidate_algorithms=[r['algorithm'] for r in champ_result.get('ranking', [])[:5]],
        beats_baseline=champ_result.get('beats_baseline'),
    ).model_dump()
    fallback['_championship'] = {
        'ranking': champ_result.get('ranking'),
        'beats_baseline': champ_result.get('beats_baseline'),
        'naive_rmse': champ_result.get('naive_rmse'),
    }

    if not is_nim_available():
        return fallback

    from predictions.domains import resolve_dataset_domain
    domain_meta = resolve_dataset_domain(dataset)

    system = (
        "Você é um orquestrador de modelagem de séries temporais. "
        "Use as tools para inspecionar o dataset e o campeonato. "
        "Escolha algorithm e parameters conservadores. "
        "Nunca invente métricas — use apenas resultados das tools. "
        "IMPORTANTE: todo texto voltado ao usuário (especialmente o campo rationale) "
        "deve ser escrito em português do Brasil (pt-BR). "
        "Não use inglês no rationale. "
        f"Domínio do dataset: {domain_meta['code']} ({domain_meta['label']}). "
        f"Orientação setorial: {domain_meta.get('agent_guidance', '')} "
        f"Vocabulário do alvo: {domain_meta.get('target_vocabulary', 'série alvo')}. "
        "Resposta final: JSON ModelPlan com language=\"pt-BR\"."
    )
    user = (
        f"Dataset id={dataset_id}, domínio={domain_meta['code']}. "
        "Planeje o melhor algoritmo para previsão. "
        "Chame run_championship se necessário e então emita o ModelPlan. "
        "Escreva o rationale em português (pt-BR), mencionando o contexto do domínio."
    )
    parsed = tool_calling_loop(
        system=system,
        user=user,
        tools=_tool_schemas(),
        tool_handlers=handlers,
        schema=ModelPlan,
        temperature=0.2,
        max_rounds=3,
    )
    if parsed is None:
        return fallback
    plan = parsed.model_dump()
    plan['language'] = 'pt-BR'
    plan['_championship'] = {
        'ranking': champ_result.get('ranking'),
        'beats_baseline': champ_result.get('beats_baseline'),
        'naive_rmse': champ_result.get('naive_rmse'),
    }
    if plan.get('beats_baseline') is None:
        plan['beats_baseline'] = champ_result.get('beats_baseline')
    return plan
