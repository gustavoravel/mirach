"""
Dataset domains inspired by Amazon Forecast predefined domains.

See: https://docs.aws.amazon.com/forecast/latest/dg/howitworks-domains-ds-types.html

A domain does not change the forecasting engine itself — it guides:
- column/role expectations (target vs related series vs metadata)
- sectoral hyperparameter presets in the wizard
- agent copy (orchestrator rationale, narrative insights)
- results UI vocabulary (demand, inventory, traffic, etc.)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Canonical domain codes (AWS-aligned + product extensions)
DOMAIN_RETAIL = 'RETAIL'
DOMAIN_INVENTORY = 'INVENTORY_PLANNING'
DOMAIN_WORKFORCE = 'WORK_FORCE'
DOMAIN_WEB_TRAFFIC = 'WEB_TRAFFIC'
DOMAIN_METRICS = 'METRICS'
DOMAIN_MANUFACTURING = 'MANUFACTURING'
DOMAIN_LOGISTICS = 'LOGISTICS'
DOMAIN_CUSTOM = 'CUSTOM'

DOMAIN_CHOICES = [
    (DOMAIN_RETAIL, 'Varejo'),
    (DOMAIN_INVENTORY, 'Planejamento de estoque'),
    (DOMAIN_WORKFORCE, 'Força de trabalho'),
    (DOMAIN_WEB_TRAFFIC, 'Tráfego web'),
    (DOMAIN_METRICS, 'Métricas / KPIs'),
    (DOMAIN_MANUFACTURING, 'Manufatura'),
    (DOMAIN_LOGISTICS, 'Logística'),
    (DOMAIN_CUSTOM, 'Personalizado'),
]

# Free-text / legacy aliases → canonical code
_ALIAS_MAP = {
    'retail': DOMAIN_RETAIL,
    'varejo': DOMAIN_RETAIL,
    'demanda': DOMAIN_RETAIL,
    'sales': DOMAIN_RETAIL,
    'vendas': DOMAIN_RETAIL,
    'inventory': DOMAIN_INVENTORY,
    'inventory_planning': DOMAIN_INVENTORY,
    'estoque': DOMAIN_INVENTORY,
    'supply': DOMAIN_INVENTORY,
    'supply_chain': DOMAIN_INVENTORY,
    'workforce': DOMAIN_WORKFORCE,
    'work_force': DOMAIN_WORKFORCE,
    'força de trabalho': DOMAIN_WORKFORCE,
    'forca de trabalho': DOMAIN_WORKFORCE,
    'pessoal': DOMAIN_WORKFORCE,
    'hr': DOMAIN_WORKFORCE,
    'web': DOMAIN_WEB_TRAFFIC,
    'web_traffic': DOMAIN_WEB_TRAFFIC,
    'tráfego': DOMAIN_WEB_TRAFFIC,
    'trafego': DOMAIN_WEB_TRAFFIC,
    'traffic': DOMAIN_WEB_TRAFFIC,
    'metrics': DOMAIN_METRICS,
    'métricas': DOMAIN_METRICS,
    'metricas': DOMAIN_METRICS,
    'kpi': DOMAIN_METRICS,
    'revenue': DOMAIN_METRICS,
    'receita': DOMAIN_METRICS,
    'manufacturing': DOMAIN_MANUFACTURING,
    'manufatura': DOMAIN_MANUFACTURING,
    'produção': DOMAIN_MANUFACTURING,
    'producao': DOMAIN_MANUFACTURING,
    'indústria': DOMAIN_MANUFACTURING,
    'industria': DOMAIN_MANUFACTURING,
    'logistics': DOMAIN_LOGISTICS,
    'logística': DOMAIN_LOGISTICS,
    'logistica': DOMAIN_LOGISTICS,
    'transporte': DOMAIN_LOGISTICS,
    'custom': DOMAIN_CUSTOM,
    'personalizado': DOMAIN_CUSTOM,
    'outro': DOMAIN_CUSTOM,
    'other': DOMAIN_CUSTOM,
}


DOMAINS: Dict[str, Dict[str, Any]] = {
    DOMAIN_RETAIL: {
        'code': DOMAIN_RETAIL,
        'label': 'Varejo',
        'description': (
            'Previsão de demanda/vendas por produto ou loja. '
            'Alvo típico: demand/sales; relacionados: preço, estoque, promoções.'
        ),
        'target_hints': ['demand', 'sales', 'weekly_sales', 'vendas', 'demanda', 'qty', 'quantity'],
        'related_hints': ['price', 'preco', 'promo', 'inventory', 'estoque', 'webpage_hits'],
        'metadata_hints': ['brand', 'marca', 'category', 'categoria', 'color', 'cor', 'sku'],
        'target_vocabulary': 'demanda / vendas',
        'forecast_noun': 'demanda',
        'agent_guidance': (
            'Foque em sazonalidade semanal/mensal, promoções e elasticidade de preço. '
            'Explique picos de demanda e risco de ruptura de estoque.'
        ),
        'preset': {
            'algorithm': 'ets',
            'horizon': 12,
            'parameters': {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 7},
            'label': 'Varejo (ETS sazonal semanal)',
        },
        'result_tips': [
            'Compare a previsão com promoções planejadas no horizonte.',
            'Monitore SKUs com IC amplo — risco de ruptura ou overstock.',
        ],
    },
    DOMAIN_INVENTORY: {
        'code': DOMAIN_INVENTORY,
        'label': 'Planejamento de estoque',
        'description': (
            'Planejamento de abastecimento e níveis de estoque. '
            'Alvo típico: demanda ou consumo; relacionados: lead time, estoque em mãos.'
        ),
        'target_hints': ['demand', 'consumption', 'consumo', 'usage', 'reorder'],
        'related_hints': ['inventory_onhand', 'estoque', 'lead_time', 'safety_stock'],
        'metadata_hints': ['warehouse', 'armazem', 'sku', 'supplier', 'fornecedor'],
        'target_vocabulary': 'consumo / reposição',
        'forecast_noun': 'necessidade de reposição',
        'agent_guidance': (
            'Enfatize lead time, estoque de segurança e risco de ruptura. '
            'Sugira revisar cobertura de estoque no horizonte previsto.'
        ),
        'preset': {
            'algorithm': 'arima',
            'horizon': 12,
            'parameters': {
                'auto_order': False,
                'order': [1, 1, 1],
                'seasonal_order': [1, 1, 1, 12],
            },
            'label': 'Estoque (ARIMA sazonal mensal)',
        },
        'result_tips': [
            'Use o IC superior para dimensionar estoque de segurança.',
            'Alinhe o horizonte ao lead time de fornecimento.',
        ],
    },
    DOMAIN_WORKFORCE: {
        'code': DOMAIN_WORKFORCE,
        'label': 'Força de trabalho',
        'description': (
            'Dimensionamento de equipe e carga de trabalho. '
            'Alvo típico: headcount, horas ou tickets atendidos.'
        ),
        'target_hints': ['headcount', 'hours', 'horas', 'staff', 'tickets', 'calls', 'chamados'],
        'related_hints': ['overtime', 'absenteeism', 'absenteismo', 'shifts'],
        'metadata_hints': ['role', 'cargo', 'team', 'equipe', 'site', 'unidade'],
        'target_vocabulary': 'demanda de mão de obra',
        'forecast_noun': 'necessidade de pessoal',
        'agent_guidance': (
            'Relacione a previsão a escalas, turnos e sazonalidade operacional. '
            'Aponte riscos de subdimensionamento em picos.'
        ),
        'preset': {
            'algorithm': 'ets',
            'horizon': 8,
            'parameters': {'trend': 'add', 'seasonal': 'add', 'seasonal_periods': 7},
            'label': 'Força de trabalho (ETS semanal)',
        },
        'result_tips': [
            'Converta a previsão em FTEs ou turnos necessários.',
            'Revise feriados e campanhas que alterem a carga.',
        ],
    },
    DOMAIN_WEB_TRAFFIC: {
        'code': DOMAIN_WEB_TRAFFIC,
        'label': 'Tráfego web',
        'description': (
            'Estimativa de visitas, sessões ou pageviews. '
            'Relacionados: campanhas, eventos, conversões.'
        ),
        'target_hints': ['visits', 'sessions', 'pageviews', 'visitas', 'sessoes', 'traffic'],
        'related_hints': ['campaign', 'campanha', 'ad_spend', 'bounce_rate', 'conversions'],
        'metadata_hints': ['channel', 'canal', 'device', 'dispositivo', 'page', 'pagina'],
        'target_vocabulary': 'tráfego / sessões',
        'forecast_noun': 'tráfego',
        'agent_guidance': (
            'Considere sazonalidade diária/semanal e efeitos de campanhas. '
            'Alerta para outliers de viralização ou quedas de disponibilidade.'
        ),
        'preset': {
            'algorithm': 'prophet',
            'horizon': 14,
            'parameters': {
                'daily_seasonality': True,
                'weekly_seasonality': True,
                'yearly_seasonality': False,
            },
            'label': 'Tráfego web (Prophet sazonal)',
        },
        'result_tips': [
            'Cruze a previsão com o calendário de campanhas.',
            'Use o IC para dimensionar capacidade de infraestrutura.',
        ],
    },
    DOMAIN_METRICS: {
        'code': DOMAIN_METRICS,
        'label': 'Métricas / KPIs',
        'description': (
            'Previsão de indicadores de negócio (receita, margem, cash flow). '
            'Útil para metas e planejamento financeiro.'
        ),
        'target_hints': ['revenue', 'receita', 'margin', 'margem', 'cashflow', 'kpi', 'metric'],
        'related_hints': ['customers', 'clientes', 'orders', 'pedidos', 'fx_rate'],
        'metadata_hints': ['business_unit', 'unidade', 'category', 'region', 'regiao'],
        'target_vocabulary': 'métrica / KPI',
        'forecast_noun': 'indicador',
        'agent_guidance': (
            'Traduza a previsão em linguagem de negócio e metas. '
            'Destaque tendência e risco de desvio vs. baseline.'
        ),
        'preset': {
            'algorithm': 'arima',
            'horizon': 12,
            'parameters': {'use_auto_arima': True},
            'label': 'Métricas (Auto-ARIMA)',
        },
        'result_tips': [
            'Compare a previsão com a meta do período.',
            'Separe tendência estrutural de ruído de curto prazo.',
        ],
    },
    DOMAIN_MANUFACTURING: {
        'code': DOMAIN_MANUFACTURING,
        'label': 'Manufatura',
        'description': (
            'Produção industrial com sazonalidade anual e ciclos de capacidade. '
            'Alvo típico: unidades produzidas ou throughput.'
        ),
        'target_hints': ['production', 'producao', 'output', 'throughput', 'units', 'unidades'],
        'related_hints': ['capacity', 'capacidade', 'downtime', 'scrap', 'orders'],
        'metadata_hints': ['plant', 'planta', 'line', 'linha', 'sku', 'product'],
        'target_vocabulary': 'produção',
        'forecast_noun': 'volume de produção',
        'agent_guidance': (
            'Considere sazonalidade anual, manutenção e restrições de capacidade. '
            'Relate a previsão ao planejamento de MRP/capacidade.'
        ),
        'preset': {
            'algorithm': 'arima',
            'horizon': 12,
            'parameters': {
                'auto_order': False,
                'order': [1, 1, 1],
                'seasonal_order': [1, 1, 1, 12],
            },
            'label': 'Manufatura (ARIMA sazonal anual)',
        },
        'result_tips': [
            'Valide a previsão contra capacidade instalada.',
            'Antecipe paradas programadas no horizonte.',
        ],
    },
    DOMAIN_LOGISTICS: {
        'code': DOMAIN_LOGISTICS,
        'label': 'Logística',
        'description': (
            'Volumes de transporte, entregas ou ocupação de frota. '
            'Modelos ML capturam efeitos recentes e regressores operacionais.'
        ),
        'target_hints': ['shipments', 'entregas', 'volume', 'orders', 'freight', 'frete'],
        'related_hints': ['distance', 'fuel', 'combustivel', 'weather', 'holiday'],
        'metadata_hints': ['route', 'rota', 'hub', 'carrier', 'transportadora'],
        'target_vocabulary': 'volume logístico',
        'forecast_noun': 'volume de entregas',
        'agent_guidance': (
            'Foque em picos operacionais, lead times e impacto de feriados. '
            'Sugira alocação de frota e capacidade de hubs.'
        ),
        'preset': {
            'algorithm': 'lightgbm',
            'horizon': 8,
            'parameters': {
                'n_estimators': 300,
                'learning_rate': 0.05,
                'num_leaves': 31,
            },
            'label': 'Logística (LightGBM)',
        },
        'result_tips': [
            'Alinhe a previsão à capacidade de frota e hubs.',
            'Revise feriados e eventos regionais no horizonte.',
        ],
    },
    DOMAIN_CUSTOM: {
        'code': DOMAIN_CUSTOM,
        'label': 'Personalizado',
        'description': (
            'Domínio genérico quando os dados não se encaixam em um setor predefinido. '
            'Requer mapeamento explícito de timestamp e alvo.'
        ),
        'target_hints': ['value', 'valor', 'y', 'target', 'alvo'],
        'related_hints': [],
        'metadata_hints': [],
        'target_vocabulary': 'série alvo',
        'forecast_noun': 'previsão',
        'agent_guidance': (
            'Use linguagem neutra e foque em tendência, sazonalidade e qualidade dos dados. '
            'Evite jargão setorial específico.'
        ),
        'preset': {
            'algorithm': 'arima',
            'horizon': 12,
            'parameters': {'use_auto_arima': True},
            'label': 'Personalizado (Auto-ARIMA)',
        },
        'result_tips': [
            'Confirme se o horizonte e a frequência fazem sentido para o caso.',
            'Revise o mapeamento de colunas se as métricas parecerem inconsistentes.',
        ],
    },
}


def normalize_domain(value: Optional[str]) -> str:
    """Map free text / legacy labels to a canonical domain code."""
    if not value:
        return DOMAIN_CUSTOM
    raw = str(value).strip()
    if not raw:
        return DOMAIN_CUSTOM
    upper = raw.upper().replace(' ', '_').replace('-', '_')
    if upper in DOMAINS:
        return upper
    key = raw.lower().strip()
    if key in _ALIAS_MAP:
        return _ALIAS_MAP[key]
    # Fuzzy contains
    for alias, code in _ALIAS_MAP.items():
        if alias in key or key in alias:
            return code
    return DOMAIN_CUSTOM


def get_domain(code_or_label: Optional[str]) -> Dict[str, Any]:
    code = normalize_domain(code_or_label)
    return dict(DOMAINS.get(code, DOMAINS[DOMAIN_CUSTOM]))


def domain_choices() -> List[Dict[str, str]]:
    return [{'code': c, 'label': lbl} for c, lbl in DOMAIN_CHOICES]


def domain_presets() -> List[Dict[str, Any]]:
    """Presets for wizard select (one per domain with a meaningful template)."""
    out = []
    for code, meta in DOMAINS.items():
        preset = dict(meta.get('preset') or {})
        if not preset:
            continue
        out.append({
            'code': code,
            'label': preset.get('label') or meta['label'],
            'algorithm': preset.get('algorithm'),
            'horizon': preset.get('horizon', 12),
            'parameters': preset.get('parameters') or {},
            'description': meta.get('description', ''),
        })
    return out


def resolve_dataset_domain(dataset) -> Dict[str, Any]:
    """Resolve domain from dataset.ai_interpretation (code or free text)."""
    interp = dataset.ai_interpretation if isinstance(getattr(dataset, 'ai_interpretation', None), dict) else {}
    code = interp.get('domain_code') or interp.get('inferred_domain')
    return get_domain(code)


def enrich_interpretation_with_domain(interpretation: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize and attach domain_code + label on an interpretation dict."""
    data = dict(interpretation or {})
    domain = get_domain(data.get('domain_code') or data.get('inferred_domain'))
    data['domain_code'] = domain['code']
    data['inferred_domain'] = domain['label']  # human-readable for UI chips
    data['domain_label'] = domain['label']
    return data
