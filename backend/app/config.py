from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv(name: str) -> list[str]:
    raw_value = os.getenv(name, "")
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value.strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value.strip())
    except ValueError:
        return default

SERVICE_NAME = "legal-ai-api"
API_VERSION = "v1"
API_PREFIX = "/api/v1"
DEBUG = _env_bool("DEBUG", default=False)
LOCAL_ORIGIN_REGEX = r"https?://(localhost|127\.0\.0\.1)(:\d+)?$"
ALLOWED_ORIGINS = _env_csv("ALLOWED_ORIGINS")
ALLOW_ORIGIN_REGEX = None if ALLOWED_ORIGINS else LOCAL_ORIGIN_REGEX

PREDICTION_MODEL_NAME = "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33"
EXPLANATION_METHOD_NAME = "sentence_evidence_with_weighted_terms"
SIMILARITY_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_INPUT_CHARS = 400_000
LONG_DOCUMENT_CHAR_THRESHOLD = 100_000
PREDICTION_MAX_CHUNK_CHARS = 2200
PREDICTION_CHUNK_STRIDE_CHARS = 1600
PREDICTION_MAX_CHUNKS = 16
PREDICTION_BATCH_SIZE = 4
AI_JUDGE_EXTRACTION_ENABLED = _env_bool("AI_JUDGE_EXTRACTION_ENABLED", default=True)
AI_JUDGE_NER_MODEL = os.getenv("AI_JUDGE_NER_MODEL", "dslim/bert-base-NER")
AI_JUDGE_MIN_CONFIDENCE = _env_float("AI_JUDGE_MIN_CONFIDENCE", 0.72)
AI_JUDGE_MAX_CONTEXTS = max(10, _env_int("AI_JUDGE_MAX_CONTEXTS", 80))
SIMILARITY_RERANK_POOL_MULTIPLIER = max(2, _env_int("SIMILARITY_RERANK_POOL_MULTIPLIER", 4))
SIMILARITY_RERANK_MAX_CANDIDATES = max(10, _env_int("SIMILARITY_RERANK_MAX_CANDIDATES", 40))
SIMILARITY_RERANK_WEIGHT = float(os.getenv("SIMILARITY_RERANK_WEIGHT", "0.6"))
DEFAULT_TOP_K = 6
MAX_TOP_K = 30
DEFAULT_TOP_K_TERMS = 10
DEFAULT_TOP_K_SENTENCES = 3
ANALYSIS_CACHE_MAX_SIZE = max(1, _env_int("ANALYSIS_CACHE_MAX_SIZE", 256))
ANALYSIS_CACHE_TTL_SECONDS = max(1, _env_int("ANALYSIS_CACHE_TTL_SECONDS", 600))

SIMILARITY_DISCLAIMER = (
    "Similar cases for research. Similarity does not guarantee same outcome."
)

SIMILARITY_INDEX_DIR = PROJECT_ROOT / "artifacts" / "similarity_index"
RETRIEVAL_EMBEDDING_DIR = PROJECT_ROOT / "artifacts" / "retrieval_case_embeddings"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "ildc"

PROCESSED_SPLITS = ("train", "dev", "test")

LABEL_NAMES = {
    0: "rejected",
    1: "accepted",
}
