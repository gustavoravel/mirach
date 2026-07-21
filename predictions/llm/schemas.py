"""Pydantic schemas for NIM agent structured outputs."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ColumnSuggestion(BaseModel):
    column_name: str
    column_type: str = Field(
        description="One of: timestamp, target, feature, ignore"
    )
    reason: str = Field(default="", description="Motivo em português (pt-BR)")


class DatasetInterpretation(BaseModel):
    timestamp_column: Optional[str] = None
    target_column: Optional[str] = None
    feature_columns: List[str] = Field(default_factory=list)
    ignore_columns: List[str] = Field(default_factory=list)
    date_format: Optional[str] = None
    dayfirst: bool = True
    inferred_frequency: Optional[str] = None
    inferred_domain: Optional[str] = Field(
        default=None,
        description="Domínio inferido em português (ex: varejo, manufatura)",
    )
    issues: List[str] = Field(
        default_factory=list,
        description="Problemas ou alertas em português (pt-BR)",
    )
    column_suggestions: List[ColumnSuggestion] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ModelPlan(BaseModel):
    algorithm: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    transforms: List[str] = Field(
        default_factory=list,
        description="Optional transforms e.g. log, diff",
    )
    rationale: str = Field(
        default="",
        description="Explicação em português (pt-BR) do porquê deste algoritmo e parâmetros",
    )
    candidate_algorithms: List[str] = Field(default_factory=list)
    beats_baseline: Optional[bool] = None
    language: str = Field(default="pt-BR", description="Idioma do rationale (sempre pt-BR)")


class InsightReport(BaseModel):
    summary: str = Field(description="Resumo executivo em português (pt-BR)")
    drivers: List[str] = Field(
        default_factory=list,
        description="Drivers em português (pt-BR)",
    )
    risks: List[str] = Field(
        default_factory=list,
        description="Riscos em português (pt-BR)",
    )
    next_steps: List[str] = Field(
        default_factory=list,
        description="Próximos passos em português (pt-BR)",
    )
    language: str = "pt-BR"
