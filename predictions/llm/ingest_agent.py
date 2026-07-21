"""Ingest agent: interpret dataset profile via NVIDIA NIM."""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, Optional

import pandas as pd
from django.core.cache import cache

from datasets.utils import build_data_profile, detect_time_series_columns
from .agent import chat_structured
from .client import is_nim_available
from .schemas import DatasetInterpretation

logger = logging.getLogger(__name__)


def _fallback_interpretation(df: pd.DataFrame, profile: Dict[str, Any]) -> Dict[str, Any]:
    suggestions = detect_time_series_columns(df)
    timestamp = next((c for c, t in suggestions.items() if t == 'timestamp'), None)
    target = next((c for c, t in suggestions.items() if t == 'target'), None)
    features = [c for c, t in suggestions.items() if t == 'feature']
    ignore = [c for c, t in suggestions.items() if t == 'ignore']

    from predictions.domains import DOMAINS, DOMAIN_CUSTOM, enrich_interpretation_with_domain

    # Heuristic domain from column name hints
    cols_blob = ' '.join(str(c).lower() for c in (df.columns or []))
    guessed = DOMAIN_CUSTOM
    best_hits = 0
    for code, meta in DOMAINS.items():
        if code == DOMAIN_CUSTOM:
            continue
        hints = (meta.get('target_hints') or []) + (meta.get('related_hints') or [])
        hits = sum(1 for h in hints if h in cols_blob)
        if hits > best_hits:
            best_hits = hits
            guessed = code

    result = DatasetInterpretation(
        timestamp_column=timestamp,
        target_column=target,
        feature_columns=features,
        ignore_columns=ignore,
        dayfirst=bool(profile.get('dayfirst', True)),
        inferred_frequency=profile.get('inferred_freq'),
        inferred_domain=guessed,
        domain_code=guessed,
        issues=list(profile.get('warnings') or []),
        confidence=0.4,
    ).model_dump()
    return enrich_interpretation_with_domain(result)


def _compact_profile_for_llm(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Send aggregates/samples only — never the full file."""
    return {
        'rows': profile.get('rows'),
        'columns': profile.get('columns'),
        'dtypes': profile.get('dtypes'),
        'null_pct': profile.get('null_pct'),
        'sample_rows': profile.get('sample_rows'),
        'dayfirst_heuristic': profile.get('dayfirst'),
        'inferred_freq': profile.get('inferred_freq'),
        'warnings': profile.get('warnings'),
        'target_stats': profile.get('target_stats'),
    }


def suggest_column_mappings(dataset) -> Optional[Dict[str, Any]]:
    """
    Return DatasetInterpretation as dict. Uses NIM when available,
    otherwise heuristic fallback. Cached by profile hash.
    """
    try:
        ext = (dataset.file_type or '').lower()
        with dataset.file.open('rb') as fh:
            if ext in ['.xlsx', '.xls'] or (dataset.file.name or '').lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(fh)
            else:
                from datasets.utils import _read_csv_bytes
                df, _ = _read_csv_bytes(fh.read())
    except Exception as exc:
        logger.warning("IngestAgent could not read dataset %s: %s", dataset.pk, exc)
        return None

    profile = dataset.data_profile if dataset.data_profile else build_data_profile(df)
    if not dataset.data_profile:
        try:
            dataset.data_profile = profile
            dataset.save(update_fields=['data_profile'])
        except Exception:
            pass

    compact = _compact_profile_for_llm(profile)
    cache_key = 'llm:profile:' + hashlib.sha256(
        json.dumps(compact, sort_keys=True, default=str).encode('utf-8')
    ).hexdigest()[:32]
    cached = cache.get(cache_key)
    if cached:
        from predictions.domains import enrich_interpretation_with_domain
        return enrich_interpretation_with_domain(cached)

    if not is_nim_available():
        result = _fallback_interpretation(df, profile)
        cache.set(cache_key, result, 3600)
        return result

    system = (
        "Você é um especialista em séries temporais. Analise o perfil compacto "
        "de um dataset e sugira mapeamento de colunas. "
        "Nunca invente dados numéricos. Responda só em JSON. "
        "Todo texto explicativo (reason, issues) deve estar em português (pt-BR). "
        "Para o domínio, escolha UM código entre: RETAIL, INVENTORY_PLANNING, "
        "WORK_FORCE, WEB_TRAFFIC, METRICS, MANUFACTURING, LOGISTICS, CUSTOM "
        "(inspirado nos domínios do Amazon Forecast)."
    )
    user = (
        "Perfil do dataset:\n"
        f"{json.dumps(compact, ensure_ascii=False, default=str)}\n\n"
        "Identifique timestamp, target, features e colunas a ignorar. "
        "Indique date_format (ex: %d/%m/%Y), dayfirst, inferred_frequency, "
        "inferred_domain (código canônico do domínio) e issues. "
        "Escreva reasons e issues em português."
    )
    parsed = chat_structured(
        system=system,
        user=user,
        schema=DatasetInterpretation,
        temperature=0.2,
        max_tokens=2048,
    )
    if parsed is None:
        result = _fallback_interpretation(df, profile)
    else:
        result = parsed.model_dump()
        # Persist dayfirst into profile for prepare_data
        try:
            profile = dict(dataset.data_profile or profile)
            profile['dayfirst'] = result.get('dayfirst', profile.get('dayfirst', True))
            if result.get('inferred_frequency'):
                profile['inferred_freq'] = result['inferred_frequency']
            dataset.data_profile = profile
            dataset.save(update_fields=['data_profile'])
        except Exception:
            pass

    from predictions.domains import enrich_interpretation_with_domain
    result = enrich_interpretation_with_domain(result)

    cache.set(cache_key, result, 3600)
    return result
