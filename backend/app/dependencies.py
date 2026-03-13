from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from .config import DEBUG
from .services.case_lookup import CaseLookupService
from .services.explanation import ExplanationService
from .services.analysis_cache import AnalysisCacheService
from .services.chat_rag import ChatRagService
from .services.prediction import PredictionService
from .services.similarity import SimilarityService


ERROR_MESSAGES: dict[str, str] = {
    "invalid_input": "We couldn't process this case text. Please review your input and try again.",
    "validation_error": "Some request fields are invalid. Please review the form and try again.",
    "case_not_found": "We couldn't find that case ID. Please check it and try again.",
    "internal_error": "Something went wrong on our side. Please try again in a moment.",
}


def get_user_friendly_message(code: str, fallback: str | None = None) -> str:
    if fallback:
        return fallback
    return ERROR_MESSAGES.get(code, ERROR_MESSAGES["internal_error"])


class ApiException(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str | None = None,
        debug_detail: str | None = None,
    ) -> None:
        resolved_message = get_user_friendly_message(code, message)
        super().__init__(resolved_message)
        self.status_code = status_code
        self.code = code
        self.message = resolved_message
        self.debug_detail = debug_detail


@dataclass
class ApiServices:
    case_lookup: CaseLookupService
    prediction: PredictionService
    explanation: ExplanationService
    similarity: SimilarityService
    chat_rag: ChatRagService
    analysis_cache: AnalysisCacheService


@asynccontextmanager
async def lifespan(app):
    case_lookup = CaseLookupService()
    prediction = PredictionService()
    explanation = ExplanationService(prediction)
    similarity = SimilarityService()
    chat_rag = ChatRagService(prediction)
    analysis_cache = AnalysisCacheService()
    app.state.services = ApiServices(
        case_lookup=case_lookup,
        prediction=prediction,
        explanation=explanation,
        similarity=similarity,
        chat_rag=chat_rag,
        analysis_cache=analysis_cache,
    )
    yield


def get_services(request: Request) -> ApiServices:
    return request.app.state.services


def success_envelope(request: Request, data: dict[str, Any], warnings: list[str] | None = None):
    return {
        "status": "ok",
        "request_id": getattr(request.state, "request_id", ""),
        "warnings": warnings or [],
        "data": data,
    }


def error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str | None = None,
    debug_detail: str | None = None,
) -> JSONResponse:
    error_payload: dict[str, Any] = {
        "code": code,
        "message": get_user_friendly_message(code, message),
    }
    if DEBUG and debug_detail:
        error_payload["debug_detail"] = debug_detail

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "request_id": getattr(request.state, "request_id", ""),
            "error": error_payload,
        },
    )
