from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import pipeline


PROCESSED_DIR = Path("data/processed/ildc")
SPLITS_DIR = Path("data/splits/ildc")
ARTIFACT_DIR = Path("artifacts/advanced/deberta_zero_shot")
MODEL_NAME = "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33"
HYPOTHESIS_TEMPLATE = "The legal outcome of this case is {}."
CANDIDATE_LABELS = ["accepted", "rejected"]
REQUIRE_CUDA = True
MAX_INPUT_CHARS = 120_000
MAX_CHUNK_CHARS = 2200
CHUNK_STRIDE_CHARS = 1600
MAX_CHUNKS = 16
BATCH_SIZE = 4


@dataclass
class SplitMetrics:
    split: str
    rows: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: list[list[int]]


def load_expected_ids(split: str) -> set[str]:
    path = SPLITS_DIR / f"{split}_ids.txt"
    with path.open("r", encoding="utf-8") as handle:
        return {line.strip() for line in handle if line.strip()}


def prepare_text(series: pd.Series, max_chars: int) -> pd.Series:
    return series.fillna("").str.slice(0, max_chars)


def load_split(split: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{split}.parquet"
    df = pd.read_parquet(path, columns=["id", "label", "clean_text", "needs_chunking"])
    expected_ids = load_expected_ids(split)
    actual_ids = set(df["id"].tolist())
    if actual_ids != expected_ids:
        raise ValueError(f"Saved split IDs do not match parquet contents for split={split}")
    df = df.copy()
    df["model_text"] = prepare_text(df["clean_text"], MAX_INPUT_CHARS)
    return df


def make_chunks(text: str) -> tuple[list[str], bool]:
    cleaned = (text or "").strip()
    if len(cleaned) <= MAX_CHUNK_CHARS:
        return [cleaned], False

    chunks: list[str] = []
    start = 0
    while start < len(cleaned) and len(chunks) < MAX_CHUNKS:
        end = min(start + MAX_CHUNK_CHARS, len(cleaned))
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start += CHUNK_STRIDE_CHARS

    truncated = (start + MAX_CHUNK_CHARS) < len(cleaned)
    if not chunks:
        return [cleaned[:MAX_CHUNK_CHARS]], len(cleaned) > MAX_CHUNK_CHARS
    return chunks, truncated


def run_zero_shot(classifier, texts: list[str]) -> list[dict[str, float]]:
    if not texts:
        return []

    outputs = classifier(
        texts,
        candidate_labels=CANDIDATE_LABELS,
        multi_label=False,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
        batch_size=BATCH_SIZE,
    )
    if isinstance(outputs, dict):
        outputs = [outputs]

    normalized_scores: list[dict[str, float]] = []
    for output in outputs:
        labels = [str(label).strip().lower() for label in output.get("labels", [])]
        scores = [float(score) for score in output.get("scores", [])]
        label_to_score = {label: score for label, score in zip(labels, scores)}
        accepted = float(label_to_score.get("accepted", 0.0))
        rejected = float(label_to_score.get("rejected", 0.0))
        denom = max(accepted + rejected, 1e-12)
        normalized_scores.append(
            {
                "accepted": accepted / denom,
                "rejected": rejected / denom,
            }
        )
    return normalized_scores


def infer_probabilities(classifier, df: pd.DataFrame) -> tuple[list[float], list[bool], list[int]]:
    probabilities: list[float] = []
    truncated_flags: list[bool] = []
    chunk_counts: list[int] = []

    for text in df["model_text"].tolist():
        chunks, truncated = make_chunks(text)
        chunk_scores = run_zero_shot(classifier, chunks)
        accepted_probability = sum(score["accepted"] for score in chunk_scores) / max(len(chunk_scores), 1)
        probabilities.append(float(accepted_probability))
        truncated_flags.append(bool(truncated))
        chunk_counts.append(int(len(chunks)))

    return probabilities, truncated_flags, chunk_counts


def compute_metrics(split: str, y_true, y_pred, y_prob) -> SplitMetrics:
    return SplitMetrics(
        split=split,
        rows=int(len(y_true)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
    )


def save_predictions(split: str, df: pd.DataFrame, y_pred, y_prob, truncated_flags, chunk_counts) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACT_DIR / f"{split}_predictions.parquet"
    pred_df = pd.DataFrame(
        {
            "id": df["id"],
            "label": df["label"],
            "predicted_label": y_pred,
            "accepted_probability": y_prob,
            "needs_chunking": df["needs_chunking"],
            "truncated_for_model": truncated_flags,
            "chunk_count": chunk_counts,
        }
    )
    pred_df.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_split("train")
    dev_df = load_split("dev")
    test_df = load_split("test")

    if REQUIRE_CUDA and not torch.cuda.is_available():
        raise RuntimeError("GPU is required for this script, but CUDA is not available.")

    device_index = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        device=device_index,
    )

    started = time.time()
    train_prob, train_truncated, train_chunk_counts = infer_probabilities(classifier, train_df)
    dev_prob, dev_truncated, dev_chunk_counts = infer_probabilities(classifier, dev_df)
    test_prob, test_truncated, test_chunk_counts = infer_probabilities(classifier, test_df)
    inference_seconds = round(time.time() - started, 2)

    train_pred = [int(prob >= 0.5) for prob in train_prob]
    dev_pred = [int(prob >= 0.5) for prob in dev_prob]
    test_pred = [int(prob >= 0.5) for prob in test_prob]

    dev_metrics = compute_metrics("dev", dev_df["label"], dev_pred, dev_prob)
    test_metrics = compute_metrics("test", test_df["label"], test_pred, test_prob)

    train_pred_path = save_predictions("train", train_df, train_pred, train_prob, train_truncated, train_chunk_counts)
    dev_pred_path = save_predictions("dev", dev_df, dev_pred, dev_prob, dev_truncated, dev_chunk_counts)
    test_pred_path = save_predictions("test", test_df, test_pred, test_prob, test_truncated, test_chunk_counts)

    report = {
        "model_name": "deberta_v3_zero_shot",
        "hf_model": MODEL_NAME,
        "input_processed_dir": str(PROCESSED_DIR),
        "split_manifest": str(SPLITS_DIR / "split_manifest.json"),
        "inference_config": {
            "max_input_chars": MAX_INPUT_CHARS,
            "max_chunk_chars": MAX_CHUNK_CHARS,
            "chunk_stride_chars": CHUNK_STRIDE_CHARS,
            "max_chunks": MAX_CHUNKS,
            "batch_size": BATCH_SIZE,
        },
        "runtime": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "require_cuda": REQUIRE_CUDA,
            "inference_seconds": inference_seconds,
        },
        "dataset": {
            "train_rows": int(len(train_df)),
            "dev_rows": int(len(dev_df)),
            "test_rows": int(len(test_df)),
        },
        "metrics": {
            "dev": asdict(dev_metrics),
            "test": asdict(test_metrics),
        },
        "artifacts": {
            "train_predictions": str(train_pred_path),
            "dev_predictions": str(dev_pred_path),
            "test_predictions": str(test_pred_path),
        },
    }

    report_path = ARTIFACT_DIR / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
