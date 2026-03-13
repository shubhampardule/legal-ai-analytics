from __future__ import annotations

import json
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import AutoModel, AutoTokenizer


PROCESSED_DIR = Path("data/processed/ildc")
SPLITS_DIR = Path("data/splits/ildc")
ARTIFACT_DIR = Path("artifacts/advanced/minilm_embedding_logreg")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEGMENT_HEAD_CHARS = 4000
SEGMENT_TAIL_CHARS = 4000
TOKEN_MAX_LENGTH = 256
BATCH_SIZE = 32
CUDA_BATCH_SIZE = 128
RANDOM_STATE = 42
LOGREG_C_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0]
THRESHOLDS = [0.35, 0.4, 0.45, 0.5]
FORCE_RECOMPUTE_EMBEDDINGS = False


@dataclass
class SplitMetrics:
    split: str
    rows: int
    threshold: float
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


def load_split(split: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{split}.parquet"
    df = pd.read_parquet(path, columns=["id", "label", "clean_text", "needs_chunking"])
    expected_ids = load_expected_ids(split)
    actual_ids = set(df["id"].tolist())
    if actual_ids != expected_ids:
        raise ValueError(f"Saved split IDs do not match parquet contents for split={split}")
    return df


def make_segments(text: str) -> list[str]:
    text = text or ""
    if len(text) <= SEGMENT_HEAD_CHARS + SEGMENT_TAIL_CHARS:
        return [text]
    head = text[:SEGMENT_HEAD_CHARS]
    tail = text[-SEGMENT_TAIL_CHARS:]
    if head == tail:
        return [head]
    return [head, tail]


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    denom = torch.clamp(mask.sum(dim=1), min=1e-9)
    return masked.sum(dim=1) / denom


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_batch_size(device: torch.device) -> int:
    if device.type == "cuda":
        return CUDA_BATCH_SIZE
    return BATCH_SIZE


def embedding_path(split: str) -> Path:
    return ARTIFACT_DIR / f"{split}_embeddings.npy"


def encode_documents(
    df: pd.DataFrame,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    segment_texts: list[str] = []
    segment_doc_ids: list[int] = []

    for doc_idx, text in enumerate(df["clean_text"].tolist()):
        segments = make_segments(text)
        segment_texts.extend(segments)
        segment_doc_ids.extend([doc_idx] * len(segments))

    doc_embeddings: list[list[np.ndarray]] = [[] for _ in range(len(df))]
    model.eval()
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode():
        for start in range(0, len(segment_texts), batch_size):
            batch_texts = segment_texts[start : start + batch_size]
            batch_doc_ids = segment_doc_ids[start : start + batch_size]
            batch = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=TOKEN_MAX_LENGTH,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            with autocast_context:
                outputs = model(**batch)
            pooled = mean_pool(outputs.last_hidden_state, batch["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            pooled_np = pooled.cpu().numpy()
            for doc_idx, emb in zip(batch_doc_ids, pooled_np):
                doc_embeddings[doc_idx].append(emb)

    aggregated = np.zeros((len(df), model.config.hidden_size), dtype=np.float32)
    for doc_idx, emb_list in enumerate(doc_embeddings):
        stacked = np.vstack(emb_list)
        mean_emb = stacked.mean(axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb = mean_emb / norm
        aggregated[doc_idx] = mean_emb.astype(np.float32)

    return aggregated


def load_or_encode_embeddings(
    split: str,
    df: pd.DataFrame,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    out_path = embedding_path(split)
    expected_shape = (len(df), model.config.hidden_size)

    if out_path.exists() and not FORCE_RECOMPUTE_EMBEDDINGS:
        cached = np.load(out_path)
        if cached.shape == expected_shape:
            print(
                f"[resume] Using cached {split} embeddings from {out_path} with shape {cached.shape}.",
                flush=True,
            )
            return cached.astype(np.float32, copy=False)
        print(
            f"[resume] Ignoring cached {split} embeddings because shape {cached.shape} "
            f"!= expected {expected_shape}.",
            flush=True,
        )

    print(
        f"[encode] Starting {split} embeddings on {device} for {len(df)} documents "
        f"(batch_size={batch_size}).",
        flush=True,
    )
    started = time.time()
    embeddings = encode_documents(df, tokenizer, model, device, batch_size)
    np.save(out_path, embeddings)
    elapsed = round(time.time() - started, 2)
    print(
        f"[encode] Saved {split} embeddings to {out_path} in {elapsed}s.",
        flush=True,
    )
    return embeddings


def evaluate_predictions(split: str, y_true, y_prob, threshold: float) -> SplitMetrics:
    y_pred = (y_prob >= threshold).astype(int)
    return SplitMetrics(
        split=split,
        rows=int(len(y_true)),
        threshold=float(threshold),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
    )


def save_predictions(split: str, df: pd.DataFrame, y_prob, threshold: float) -> Path:
    out_path = ARTIFACT_DIR / f"{split}_predictions.parquet"
    y_pred = (y_prob >= threshold).astype(int)
    pred_df = pd.DataFrame(
        {
            "id": df["id"],
            "label": df["label"],
            "predicted_label": y_pred,
            "accepted_probability": y_prob,
            "needs_chunking": df["needs_chunking"],
        }
    )
    pred_df.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_split("train")
    dev_df = load_split("dev")
    test_df = load_split("test")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    device = resolve_device()
    batch_size = resolve_batch_size(device)
    model.to(device)
    print(
        json.dumps(
            {
                "device": str(device),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "batch_size": batch_size,
                "force_recompute_embeddings": FORCE_RECOMPUTE_EMBEDDINGS,
            },
            indent=2,
        ),
        flush=True,
    )

    started = time.time()
    train_embeddings = load_or_encode_embeddings(
        "train", train_df, tokenizer, model, device, batch_size
    )
    dev_embeddings = load_or_encode_embeddings(
        "dev", dev_df, tokenizer, model, device, batch_size
    )
    test_embeddings = load_or_encode_embeddings(
        "test", test_df, tokenizer, model, device, batch_size
    )
    embedding_seconds = round(time.time() - started, 2)

    y_train = train_df["label"].to_numpy()
    y_dev = dev_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    search_results = []
    best = None
    best_key = None

    for c_value in LOGREG_C_VALUES:
        classifier = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )
        classifier.fit(train_embeddings, y_train)
        dev_prob = classifier.predict_proba(dev_embeddings)[:, 1]

        for threshold in THRESHOLDS:
            metrics = evaluate_predictions("dev", y_dev, dev_prob, threshold)
            result = {
                "C": c_value,
                "threshold": threshold,
                "dev_metrics": asdict(metrics),
            }
            search_results.append(result)
            key = (metrics.f1, metrics.recall, metrics.accuracy)
            if best is None or key > best_key:
                best = {
                    "classifier": classifier,
                    "dev_prob": dev_prob,
                    "threshold": threshold,
                    "C": c_value,
                    "dev_metrics": metrics,
                }
                best_key = key

    classifier: LogisticRegression = best["classifier"]
    chosen_threshold = float(best["threshold"])
    dev_prob = best["dev_prob"]
    test_prob = classifier.predict_proba(test_embeddings)[:, 1]
    dev_metrics = evaluate_predictions("dev", y_dev, dev_prob, chosen_threshold)
    test_metrics = evaluate_predictions("test", y_test, test_prob, chosen_threshold)

    dev_pred_path = save_predictions("dev", dev_df, dev_prob, chosen_threshold)
    test_pred_path = save_predictions("test", test_df, test_prob, chosen_threshold)

    joblib.dump(classifier, ARTIFACT_DIR / "classifier.joblib")
    (ARTIFACT_DIR / "model_config.json").write_text(
        json.dumps(
            {
                "embedding_model": MODEL_NAME,
                "segment_head_chars": SEGMENT_HEAD_CHARS,
                "segment_tail_chars": SEGMENT_TAIL_CHARS,
                "token_max_length": TOKEN_MAX_LENGTH,
                "batch_size": batch_size,
                "device": str(device),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report = {
        "advanced_model_name": "minilm_head_tail_embedding_logreg",
        "embedding_model": MODEL_NAME,
        "input_processed_dir": str(PROCESSED_DIR),
        "split_manifest": str(SPLITS_DIR / "split_manifest.json"),
        "embedding_config": {
            "segment_head_chars": SEGMENT_HEAD_CHARS,
            "segment_tail_chars": SEGMENT_TAIL_CHARS,
            "token_max_length": TOKEN_MAX_LENGTH,
            "batch_size": batch_size,
            "embedding_dimension": int(train_embeddings.shape[1]),
        },
        "classifier_search": {
            "candidate_C_values": LOGREG_C_VALUES,
            "candidate_thresholds": THRESHOLDS,
            "best_C": best["C"],
            "best_threshold": chosen_threshold,
        },
        "runtime": {
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "embedding_seconds": embedding_seconds,
        },
        "metrics": {
            "dev": asdict(dev_metrics),
            "test": asdict(test_metrics),
        },
        "artifacts": {
            "classifier": str(ARTIFACT_DIR / "classifier.joblib"),
            "model_config": str(ARTIFACT_DIR / "model_config.json"),
            "train_embeddings": str(ARTIFACT_DIR / "train_embeddings.npy"),
            "dev_embeddings": str(ARTIFACT_DIR / "dev_embeddings.npy"),
            "test_embeddings": str(ARTIFACT_DIR / "test_embeddings.npy"),
            "dev_predictions": str(dev_pred_path),
            "test_predictions": str(test_pred_path),
            "search_results": str(ARTIFACT_DIR / "search_results.json"),
        },
    }

    (ARTIFACT_DIR / "search_results.json").write_text(
        json.dumps(search_results, indent=2),
        encoding="utf-8",
    )
    (ARTIFACT_DIR / "training_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
