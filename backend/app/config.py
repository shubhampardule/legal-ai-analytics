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

SERVICE_NAME = "legal-ai-api"
API_VERSION = "v1"
API_PREFIX = "/api/v1"
DEBUG = _env_bool("DEBUG", default=False)
LOCAL_ORIGIN_REGEX = r"https?://(localhost|127\.0\.0\.1)(:\d+)?$"
ALLOWED_ORIGINS = _env_csv("ALLOWED_ORIGINS")
ALLOW_ORIGIN_REGEX = None if ALLOWED_ORIGINS else LOCAL_ORIGIN_REGEX

PREDICTION_MODEL_NAME = "baseline_tfidf_logreg"
EXPLANATION_METHOD_NAME = "linear_feature_contribution"
SIMILARITY_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_INPUT_CHARS = 400_000
MODEL_MAX_TEXT_CHARS = 100_000
LONG_DOCUMENT_CHAR_THRESHOLD = 100_000
DEFAULT_TOP_K = 6
MAX_TOP_K = 30
DEFAULT_TOP_K_TERMS = 10
DEFAULT_TOP_K_SENTENCES = 3
ANALYSIS_CACHE_MAX_SIZE = max(1, _env_int("ANALYSIS_CACHE_MAX_SIZE", 256))
ANALYSIS_CACHE_TTL_SECONDS = max(1, _env_int("ANALYSIS_CACHE_TTL_SECONDS", 600))

SIMILARITY_DISCLAIMER = (
    "Similar cases for research. Similarity does not guarantee same outcome."
)

BASELINE_DIR = PROJECT_ROOT / "artifacts" / "baseline" / "tfidf_logreg"
SIMILARITY_INDEX_DIR = PROJECT_ROOT / "artifacts" / "similarity_index"
RETRIEVAL_EMBEDDING_DIR = PROJECT_ROOT / "artifacts" / "retrieval_case_embeddings"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "ildc"

PROCESSED_SPLITS = ("train", "dev", "test")

LABEL_NAMES = {
    0: "rejected",
    1: "accepted",
}
