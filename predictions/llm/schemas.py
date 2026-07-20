"""Pydantic schemas for NIM agent structured outputs."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ColumnSuggestion(BaseModel):
    column_name: str
    column_type: str = Field(
        description="One of: timestamp, target, feature, ignore"
    )
    reason: str = ""


class DatasetInterpretation(BaseModel):
    timestamp_column: Optional[str] = None
    target_column: Optional[str] = None
    feature_columns: List[str] = Field(default_factory=list)
    ignore_columns: List[str] = Field(default_factory=list)
    date_format: Optional[str] = None
    dayfirst: bool = True
    inferred_frequency: Optional[str] = None
    inferred_domain: Optional[str] = None
    issues: List[str] = Field(default_factory=list)
    column_suggestions: List[ColumnSuggestion] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class ModelPlan(BaseModel):
    algorithm: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    transforms: List[str] = Field(
        default_factory=list,
        description="Optional transforms e.g. log, diff",
    )
    rationale: str = ""
    candidate_algorithms: List[str] = Field(default_factory=list)
    beats_baseline: Optional[bool] = None


class InsightReport(BaseModel):
    summary: str
    drivers: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    language: str = "pt-BR"
