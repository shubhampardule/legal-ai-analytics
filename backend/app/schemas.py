from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .config import DEFAULT_TOP_K


class BaseRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AnalysisInput(BaseRequestModel):
    text: str | None = Field(default=None)
    case_id: str | None = Field(default=None)


class SimilarCasesRequest(AnalysisInput):
    top_k: int = Field(default=DEFAULT_TOP_K)
    outcome: str | None = Field(default=None, description="accepted|rejected")
    year_from: int | None = Field(default=None, ge=1800, le=2100)
    year_to: int | None = Field(default=None, ge=1800, le=2100)


class AnalyzeRequest(SimilarCasesRequest):
    include_explanation: bool = Field(default=True)
    include_similar_cases: bool = Field(default=True)


class CaseChatRequest(AnalysisInput):
    question: str = Field(min_length=3, max_length=2000)
    top_k_context: int = Field(default=4, ge=1, le=8)


class SuccessEnvelope(BaseModel):
    status: str = "ok"
    request_id: str
    warnings: list[str] = Field(default_factory=list)
    data: dict[str, Any]


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorEnvelope(BaseModel):
    status: str = "error"
    request_id: str
    error: ErrorDetail
