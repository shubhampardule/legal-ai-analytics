from __future__ import annotations

from copy import deepcopy
import logging
from time import perf_counter
from uuid import uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    ALLOWED_ORIGINS,
    ALLOW_ORIGIN_REGEX,
    API_PREFIX,
    API_VERSION,
    DEFAULT_TOP_K,
    EXPLANATION_METHOD_NAME,
    LONG_DOCUMENT_CHAR_THRESHOLD,
    MAX_INPUT_CHARS,
    MAX_TOP_K,
    PREDICTION_MAX_CHUNKS,
    PREDICTION_MAX_CHUNK_CHARS,
    PREDICTION_MODEL_NAME,
    SERVICE_NAME,
    SIMILARITY_DISCLAIMER,
    SIMILARITY_MODEL_NAME,
)
from .services.text_processing import extract_legal_entities
from .dependencies import (
    ApiException,
    ApiServices,
    error_response,
    get_services,
    get_user_friendly_message,
    lifespan,
    success_envelope,
)
from .logging_utils import configure_logging
from .schemas import AnalyzeRequest, AnalysisInput, CaseChatRequest, SimilarCasesRequest


configure_logging()
logger = logging.getLogger("legal_ai_api")


app = FastAPI(title=SERVICE_NAME, version=API_VERSION, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOW_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request.state.request_id = str(uuid4())
    started_at = perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = round((perf_counter() - started_at) * 1000, 3)
        logger.exception(
            "http_request_failed",
            extra={
                "request_id": request.state.request_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "duration_ms": elapsed_ms,
            },
        )
        raise

    elapsed_ms = round((perf_counter() - started_at) * 1000, 3)
    logger.info(
        "http_request_completed",
        extra={
            "request_id": request.state.request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": elapsed_ms,
        },
    )
    return response


@app.exception_handler(ApiException)
async def handle_api_exception(request: Request, exc: ApiException):
    logger.error(
        "api_exception",
        extra={
            "request_id": getattr(request.state, "request_id", ""),
            "method": request.method,
            "path": request.url.path,
            "status_code": exc.status_code,
            "error_code": exc.code,
            "error_message": exc.message,
        },
    )
    return error_response(
        request=request,
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        debug_detail=exc.debug_detail,
    )


@app.exception_handler(RequestValidationError)
async def handle_request_validation_error(request: Request, exc: RequestValidationError):
    first_error = exc.errors()[0] if exc.errors() else {}
    raw_message = first_error.get("msg", "Request validation failed.")
    user_message = get_user_friendly_message("validation_error")
    logger.warning(
        "request_validation_error",
        extra={
            "request_id": getattr(request.state, "request_id", ""),
            "method": request.method,
            "path": request.url.path,
            "status_code": 422,
            "error_code": "validation_error",
            "error_message": user_message,
            "validation_detail": raw_message,
            "errors": exc.errors(),
        },
    )
    return error_response(
        request=request,
        status_code=422,
        code="validation_error",
        message=user_message,
        debug_detail=raw_message,
    )


@app.exception_handler(Exception)
async def handle_unexpected_exception(request: Request, exc: Exception):
    user_message = get_user_friendly_message("internal_error")
    logger.exception(
        "unhandled_exception",
        extra={
            "request_id": getattr(request.state, "request_id", ""),
            "method": request.method,
            "path": request.url.path,
            "status_code": 500,
            "error_code": "internal_error",
            "error_message": user_message,
            "exception_message": str(exc),
        },
    )
    return error_response(
        request=request,
        status_code=500,
        code="internal_error",
        message=user_message,
        debug_detail=str(exc),
    )


def validate_input_source(text: str | None, case_id: str | None) -> None:
    if bool(text) == bool(case_id):
        raise ApiException(
            status_code=400,
            code="invalid_input",
            message="Exactly one of text or case_id must be provided.",
        )


def validate_top_k(top_k: int) -> None:
    if top_k < 1 or top_k > MAX_TOP_K:
        raise ApiException(
            status_code=400,
            code="invalid_input",
            message=f"top_k must be between 1 and {MAX_TOP_K}.",
        )


def validate_similarity_filters(
    outcome: str | None,
    year_from: int | None,
    year_to: int | None,
) -> None:
    if outcome is not None and outcome not in {"accepted", "rejected"}:
        raise ApiException(
            status_code=400,
            code="invalid_input",
            message="outcome must be one of: accepted, rejected.",
        )
    if year_from is not None and year_to is not None and year_from > year_to:
        raise ApiException(
            status_code=400,
            code="invalid_input",
            message="year_from must be less than or equal to year_to.",
        )


def resolve_input_context(payload: AnalysisInput, services: ApiServices):
    validate_input_source(payload.text, payload.case_id)

    if payload.case_id:
        try:
            case = services.case_lookup.get_case(payload.case_id)
        except KeyError as exc:
            raise ApiException(
                status_code=404,
                code="case_not_found",
                debug_detail=str(exc),
            ) from exc

        context = {
            "source": "case_id",
            "case_id": case["id"],
            "split": case["split"],
            "true_label": case["label_name"],
            "clean_char_length": case["clean_char_length"],
            "needs_chunking": case["needs_chunking"],
        }
        return case["clean_text"], context

    try:
        clean_text = services.prediction.clean_user_text(payload.text or "")
    except ValueError as exc:
        raise ApiException(
            status_code=400,
            code="invalid_input",
            debug_detail=str(exc),
        ) from exc

    context = {
        "source": "text",
        "case_id": None,
        "split": None,
        "true_label": None,
        "clean_char_length": len(clean_text),
        "needs_chunking": len(clean_text) > LONG_DOCUMENT_CHAR_THRESHOLD,
    }
    return clean_text, context


def extend_input_context_with_prediction(input_context: dict[str, object], prediction) -> dict[str, object]:
    enriched = dict(input_context)
    enriched["truncated_for_model"] = bool(prediction.truncated_for_model)
    return enriched


def collect_warnings(input_context: dict[str, object], prediction) -> list[str]:
    warnings: list[str] = []
    if prediction.truncated_for_model:
        warnings.append(
            "Prediction and explanation used a truncated text window "
            f"(up to {PREDICTION_MAX_CHUNKS} chunks × {PREDICTION_MAX_CHUNK_CHARS} chars)."
        )
    if bool(input_context.get("needs_chunking")):
        warnings.append(
            "Long document detected. Similarity search uses a head-tail embedding strategy for retrieval."
        )
    return warnings


def confidence_level_from_score(confidence_score: float) -> str:
    if confidence_score >= 0.8:
        return "High"
    if confidence_score >= 0.65:
        return "Moderate"
    return "Low"


def similar_case_quality_from_average(avg_similarity: float) -> str:
    if avg_similarity >= 0.85:
        return "Strong"
    if avg_similarity >= 0.7:
        return "Moderate"
    return "Weak"


def build_analysis_summary(
    prediction_payload: dict[str, object] | None,
    explanation_payload: dict[str, object] | None,
    retrieval_payload: dict[str, object] | None,
) -> dict[str, object]:
    if not prediction_payload:
        return {
            "verdict": "Prediction unavailable",
            "confidence_level": "Low",
            "evidence_count": 0,
            "similar_case_quality": "Weak",
            "sentence": "We couldn't complete prediction for this case right now. Please try again.",
        }

    predicted_label = str(prediction_payload.get("predicted_label", "rejected"))
    accepted_probability = float(prediction_payload.get("accepted_probability", 0.0))
    rejected_probability = float(prediction_payload.get("rejected_probability", 1.0 - accepted_probability))
    confidence_score = max(accepted_probability, rejected_probability)
    confidence_percent = int(round(confidence_score * 100))
    confidence_level = confidence_level_from_score(confidence_score)

    verdict = (
        "Likely to be accepted"
        if predicted_label == "accepted"
        else "Likely to be rejected"
    )

    supporting_count = 0
    opposing_count = 0
    if explanation_payload:
        sentence_evidence = explanation_payload.get("sentence_evidence")
        if isinstance(sentence_evidence, dict):
            supporting = sentence_evidence.get("supporting")
            opposing = sentence_evidence.get("opposing")
            if isinstance(supporting, list):
                supporting_count = len(supporting)
            if isinstance(opposing, list):
                opposing_count = len(opposing)
    evidence_count = supporting_count + opposing_count

    avg_similarity = 0.0
    if retrieval_payload:
        results = retrieval_payload.get("results")
        if isinstance(results, list) and results:
            similarity_values = [
                float(item.get("similarity_score", 0.0))
                for item in results
                if isinstance(item, dict)
            ]
            if similarity_values:
                avg_similarity = sum(similarity_values) / len(similarity_values)

    similar_case_quality = similar_case_quality_from_average(avg_similarity)
    sentence = (
        f"This case is {verdict.lower()} with {confidence_level.lower()} confidence "
        f"({confidence_percent}%). {evidence_count} evidence passages found."
    )

    return {
        "verdict": verdict,
        "confidence_level": confidence_level,
        "evidence_count": evidence_count,
        "similar_case_quality": similar_case_quality,
        "sentence": sentence,
    }


@app.get("/health")
async def health(request: Request, services: ApiServices = Depends(get_services)):
    data = {
        "service": SERVICE_NAME,
        "api_version": API_VERSION,
        "prediction_model_loaded": services.prediction is not None,
        "similarity_index_loaded": services.similarity is not None,
        "similarity_encoder_loaded": services.similarity is not None,
    }
    return success_envelope(request, data)


@app.get(f"{API_PREFIX}/meta")
async def meta(request: Request):
    data = {
        "service": SERVICE_NAME,
        "api_version": API_VERSION,
        "prediction_model": PREDICTION_MODEL_NAME,
        "explanation_method": EXPLANATION_METHOD_NAME,
        "similarity_model": SIMILARITY_MODEL_NAME,
        "max_input_chars": MAX_INPUT_CHARS,
        "max_top_k": MAX_TOP_K,
        "default_top_k": DEFAULT_TOP_K,
        "similarity_disclaimer": SIMILARITY_DISCLAIMER,
    }
    return success_envelope(request, data)


@app.post(f"{API_PREFIX}/predict")
async def predict(
    payload: AnalysisInput,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    clean_text, input_context = resolve_input_context(payload, services)

    prediction_started = perf_counter()
    prediction = services.prediction.predict_from_clean_text(clean_text)
    prediction_elapsed_ms = round((perf_counter() - prediction_started) * 1000, 3)
    logger.info(
        "prediction_completed",
        extra={
            "request_id": getattr(request.state, "request_id", ""),
            "method": request.method,
            "path": request.url.path,
            "duration_ms": prediction_elapsed_ms,
            "input_source": input_context.get("source"),
            "case_id": input_context.get("case_id"),
        },
    )

    warnings = collect_warnings(input_context, prediction)
    data = {
        "input": extend_input_context_with_prediction(input_context, prediction),
        "prediction": services.prediction.to_payload(prediction),
    }
    return success_envelope(request, data, warnings=warnings)


@app.post(f"{API_PREFIX}/explain")
async def explain(
    payload: AnalysisInput,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    clean_text, input_context = resolve_input_context(payload, services)

    prediction_started = perf_counter()
    prediction = services.prediction.predict_from_clean_text(clean_text)
    prediction_elapsed_ms = round((perf_counter() - prediction_started) * 1000, 3)
    logger.info(
        "prediction_completed",
        extra={
            "request_id": getattr(request.state, "request_id", ""),
            "method": request.method,
            "path": request.url.path,
            "duration_ms": prediction_elapsed_ms,
            "input_source": input_context.get("source"),
            "case_id": input_context.get("case_id"),
        },
    )

    warnings = collect_warnings(input_context, prediction)

    explanation_started = perf_counter()
    explanation = services.explanation.explain_prediction(
        prediction=prediction,
        clean_char_length=int(input_context["clean_char_length"]),
        needs_chunking=bool(input_context["needs_chunking"]),
    )
    explanation_elapsed_ms = round((perf_counter() - explanation_started) * 1000, 3)
    logger.info(
        "explanation_completed",
        extra={
            "request_id": getattr(request.state, "request_id", ""),
            "method": request.method,
            "path": request.url.path,
            "duration_ms": explanation_elapsed_ms,
            "input_source": input_context.get("source"),
            "case_id": input_context.get("case_id"),
        },
    )

    data = {
        "input": extend_input_context_with_prediction(input_context, prediction),
        "prediction": services.prediction.to_payload(prediction),
        "explanation": explanation,
    }
    return success_envelope(request, data, warnings=warnings)


@app.post(f"{API_PREFIX}/similar-cases")
async def similar_cases(
    payload: SimilarCasesRequest,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    validate_top_k(payload.top_k)
    validate_similarity_filters(payload.outcome, payload.year_from, payload.year_to)
    clean_text, input_context = resolve_input_context(payload, services)

    try:
        similarity_started = perf_counter()
        if payload.case_id:
            retrieval = services.similarity.search_by_case_id(
                payload.case_id,
                payload.top_k,
                outcome=payload.outcome,
                year_from=payload.year_from,
                year_to=payload.year_to,
            )
        else:
            retrieval = services.similarity.search_by_clean_text(
                clean_text,
                payload.top_k,
                outcome=payload.outcome,
                year_from=payload.year_from,
                year_to=payload.year_to,
            )
        similarity_elapsed_ms = round((perf_counter() - similarity_started) * 1000, 3)
        logger.info(
            "similarity_search_completed",
            extra={
                "request_id": getattr(request.state, "request_id", ""),
                "method": request.method,
                "path": request.url.path,
                "duration_ms": similarity_elapsed_ms,
                "top_k": payload.top_k,
                "input_source": input_context.get("source"),
                "case_id": input_context.get("case_id"),
            },
        )
    except KeyError as exc:
        raise ApiException(404, "case_not_found", debug_detail=str(exc)) from exc
    except ValueError as exc:
        raise ApiException(400, "invalid_input", debug_detail=str(exc)) from exc

    data = {
        "input": input_context,
        "retrieval": retrieval,
    }
    return success_envelope(request, data)


@app.post(f"{API_PREFIX}/analyze")
async def analyze(
    payload: AnalyzeRequest,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    validate_top_k(payload.top_k)
    validate_similarity_filters(payload.outcome, payload.year_from, payload.year_to)

    cache_key: tuple[str, int, bool, bool, str | None, int | None, int | None] | None = None
    if payload.case_id:
        cache_key = (
            payload.case_id,
            payload.top_k,
            payload.include_explanation,
            payload.include_similar_cases,
            payload.outcome,
            payload.year_from,
            payload.year_to,
        )
        cached = services.analysis_cache.get(cache_key)
        if cached is not None:
            logger.info(
                "analysis_cache_hit",
                extra={
                    "request_id": getattr(request.state, "request_id", ""),
                    "method": request.method,
                    "path": request.url.path,
                    "case_id": payload.case_id,
                    "top_k": payload.top_k,
                    "include_explanation": payload.include_explanation,
                    "include_similar_cases": payload.include_similar_cases,
                    "outcome": payload.outcome,
                    "year_from": payload.year_from,
                    "year_to": payload.year_to,
                },
            )
            return success_envelope(
                request,
                deepcopy(cached["data"]),
                warnings=list(cached["warnings"]),
            )

        logger.info(
            "analysis_cache_miss",
            extra={
                "request_id": getattr(request.state, "request_id", ""),
                "method": request.method,
                "path": request.url.path,
                "case_id": payload.case_id,
                "top_k": payload.top_k,
                "include_explanation": payload.include_explanation,
                "include_similar_cases": payload.include_similar_cases,
                "outcome": payload.outcome,
                "year_from": payload.year_from,
                "year_to": payload.year_to,
            },
        )

    clean_text, input_context = resolve_input_context(payload, services)
    warnings: list[str] = []
    section_status = {
        "prediction": "skipped",
        "explanation": "skipped",
        "similarity": "skipped",
    }
    section_errors: dict[str, str] = {}

    prediction = None
    entities = extract_legal_entities(clean_text)
    data = {
        "input": dict(input_context),
        "prediction": None,
        "explanation": None,
        "retrieval": None,
        "summary": None,
        "entities": entities,
        "section_status": section_status,
    }

    prediction_started = perf_counter()
    try:
        prediction = services.prediction.predict_from_clean_text(clean_text)
        prediction_elapsed_ms = round((perf_counter() - prediction_started) * 1000, 3)
        logger.info(
            "prediction_completed",
            extra={
                "request_id": getattr(request.state, "request_id", ""),
                "method": request.method,
                "path": request.url.path,
                "duration_ms": prediction_elapsed_ms,
                "input_source": input_context.get("source"),
                "case_id": input_context.get("case_id"),
            },
        )
        section_status["prediction"] = "ok"
        data["input"] = extend_input_context_with_prediction(input_context, prediction)
        data["prediction"] = services.prediction.to_payload(prediction)
        warnings.extend(collect_warnings(input_context, prediction))
    except Exception as exc:
        prediction_elapsed_ms = round((perf_counter() - prediction_started) * 1000, 3)
        message = "Prediction is temporarily unavailable. Please try again."
        section_status["prediction"] = "error"
        section_errors["prediction"] = message
        warnings.append(message)
        logger.exception(
            "prediction_failed",
            extra={
                "request_id": getattr(request.state, "request_id", ""),
                "method": request.method,
                "path": request.url.path,
                "duration_ms": prediction_elapsed_ms,
                "input_source": input_context.get("source"),
                "case_id": input_context.get("case_id"),
                "exception_message": str(exc),
            },
        )

    if payload.include_explanation:
        if prediction is None:
            section_status["explanation"] = "skipped"
            warnings.append("Explanation was skipped because prediction failed.")
        else:
            explanation_started = perf_counter()
            try:
                data["explanation"] = services.explanation.explain_prediction(
                    prediction=prediction,
                    clean_char_length=int(input_context["clean_char_length"]),
                    needs_chunking=bool(input_context["needs_chunking"]),
                )
                explanation_elapsed_ms = round((perf_counter() - explanation_started) * 1000, 3)
                logger.info(
                    "explanation_completed",
                    extra={
                        "request_id": getattr(request.state, "request_id", ""),
                        "method": request.method,
                        "path": request.url.path,
                        "duration_ms": explanation_elapsed_ms,
                        "input_source": input_context.get("source"),
                        "case_id": input_context.get("case_id"),
                    },
                )
                section_status["explanation"] = "ok"
            except Exception as exc:
                explanation_elapsed_ms = round((perf_counter() - explanation_started) * 1000, 3)
                message = "Explanation is temporarily unavailable."
                section_status["explanation"] = "error"
                section_errors["explanation"] = message
                warnings.append(message)
                logger.exception(
                    "explanation_failed",
                    extra={
                        "request_id": getattr(request.state, "request_id", ""),
                        "method": request.method,
                        "path": request.url.path,
                        "duration_ms": explanation_elapsed_ms,
                        "input_source": input_context.get("source"),
                        "case_id": input_context.get("case_id"),
                        "exception_message": str(exc),
                    },
                )

    if payload.include_similar_cases:
        similarity_started = perf_counter()
        try:
            if payload.case_id:
                retrieval = services.similarity.search_by_case_id(
                    payload.case_id,
                    payload.top_k,
                    outcome=payload.outcome,
                    year_from=payload.year_from,
                    year_to=payload.year_to,
                )
            else:
                retrieval = services.similarity.search_by_clean_text(
                    clean_text,
                    payload.top_k,
                    outcome=payload.outcome,
                    year_from=payload.year_from,
                    year_to=payload.year_to,
                )
            similarity_elapsed_ms = round((perf_counter() - similarity_started) * 1000, 3)
            logger.info(
                "similarity_search_completed",
                extra={
                    "request_id": getattr(request.state, "request_id", ""),
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": similarity_elapsed_ms,
                    "top_k": payload.top_k,
                    "input_source": input_context.get("source"),
                    "case_id": input_context.get("case_id"),
                },
            )
            section_status["similarity"] = "ok"
            data["retrieval"] = retrieval
        except Exception as exc:
            similarity_elapsed_ms = round((perf_counter() - similarity_started) * 1000, 3)
            message = (
                "Prediction and explanation succeeded, but similar case search is temporarily unavailable."
                if section_status["prediction"] == "ok"
                else "Similar case search is temporarily unavailable."
            )
            section_status["similarity"] = "error"
            section_errors["similarity"] = message
            warnings.append(message)
            logger.exception(
                "similarity_search_failed",
                extra={
                    "request_id": getattr(request.state, "request_id", ""),
                    "method": request.method,
                    "path": request.url.path,
                    "duration_ms": similarity_elapsed_ms,
                    "top_k": payload.top_k,
                    "input_source": input_context.get("source"),
                    "case_id": input_context.get("case_id"),
                    "exception_message": str(exc),
                },
            )

    if section_errors:
        data["section_errors"] = section_errors

    data["summary"] = build_analysis_summary(
        prediction_payload=data["prediction"],
        explanation_payload=data["explanation"],
        retrieval_payload=data["retrieval"],
    )

    if cache_key is not None and section_status["prediction"] == "ok":
        services.analysis_cache.set(cache_key, data=data, warnings=warnings)
        logger.info(
            "analysis_cache_store",
            extra={
                "request_id": getattr(request.state, "request_id", ""),
                "method": request.method,
                "path": request.url.path,
                "case_id": payload.case_id,
                "top_k": payload.top_k,
                "include_explanation": payload.include_explanation,
                "include_similar_cases": payload.include_similar_cases,
                "outcome": payload.outcome,
                "year_from": payload.year_from,
                "year_to": payload.year_to,
            },
        )

    return success_envelope(request, data, warnings=warnings)


@app.post(f"{API_PREFIX}/chat-case")
async def chat_case(
    payload: CaseChatRequest,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    clean_case_text, input_context = resolve_input_context(payload, services)

    rag_started = perf_counter()
    try:
        rag_result = services.chat_rag.answer_question(
            clean_case_text=clean_case_text,
            question=payload.question,
            top_k_context=payload.top_k_context,
        )
    except ValueError as exc:
        raise ApiException(
            status_code=400,
            code="invalid_input",
            debug_detail=str(exc),
        ) from exc

    rag_elapsed_ms = round((perf_counter() - rag_started) * 1000, 3)
    logger.info(
        "chat_case_completed",
        extra={
            "request_id": getattr(request.state, "request_id", ""),
            "method": request.method,
            "path": request.url.path,
            "duration_ms": rag_elapsed_ms,
            "input_source": input_context.get("source"),
            "case_id": input_context.get("case_id"),
            "top_k_context": payload.top_k_context,
        },
    )

    data = {
        "input": input_context,
        "question": payload.question,
        "rag": rag_result,
    }
    return success_envelope(request, data)


@app.get(API_PREFIX + "/cases/{case_id}")
async def get_case(
    case_id: str,
    request: Request,
    services: ApiServices = Depends(get_services),
):
    try:
        case = services.case_lookup.get_case(case_id)
        # Assuming case may be a dict. If it's a Pydantic model it would need .dict()
        if hasattr(case, "dict"):
            case = case.dict()
        elif hasattr(case, "model_dump"):
            case = case.model_dump()
        return success_envelope(request, {"case": case})
    except KeyError as exc:
        raise ApiException(404, "case_not_found", debug_detail=str(exc)) from exc



@app.get(API_PREFIX + "/cases")
async def list_cases(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    query: str = None,
    outcome: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    services: ApiServices = Depends(get_services),
):
    try:
        limit = min(max(1, limit), 200)
        offset = max(0, offset)
        validate_similarity_filters(outcome, year_from, year_to)
        cases_data = services.case_lookup.list_cases(
            limit=limit,
            offset=offset,
            query=query,
            outcome=outcome,
            year_from=year_from,
            year_to=year_to,
        )
        return success_envelope(request, cases_data)
    except Exception as exc:
        raise ApiException(500, "internal_error", debug_detail=str(exc)) from exc

