from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROCESSED_DIR = Path("data/processed/ildc")
SPLITS_DIR = Path("data/splits/ildc")
ARTIFACT_DIR = Path("artifacts/baseline/tfidf_logreg")
MAX_TEXT_CHARS = 50_000
VECTORIZER_CONFIG = {
    "analyzer": "word",
    "ngram_range": (1, 1),
    "max_features": 50_000,
    "min_df": 3,
    "max_df": 0.98,
    "sublinear_tf": True,
    "strip_accents": "unicode",
    "lowercase": True,
}
MODEL_CONFIG = {
    "solver": "liblinear",
    "max_iter": 1000,
    "random_state": 42,
}


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
    df["model_text"] = prepare_text(df["clean_text"], MAX_TEXT_CHARS)
    return df


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


def save_predictions(split: str, df: pd.DataFrame, y_pred, y_prob) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARTIFACT_DIR / f"{split}_predictions.parquet"
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

    vectorizer = TfidfVectorizer(**VECTORIZER_CONFIG)

    started = time.time()
    x_train = vectorizer.fit_transform(train_df["model_text"])
    x_dev = vectorizer.transform(dev_df["model_text"])
    x_test = vectorizer.transform(test_df["model_text"])

    model = LogisticRegression(**MODEL_CONFIG)
    model.fit(x_train, train_df["label"])
    training_seconds = round(time.time() - started, 2)

    dev_prob = model.predict_proba(x_dev)[:, 1]
    dev_pred = (dev_prob >= 0.5).astype(int)
    test_prob = model.predict_proba(x_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)

    dev_metrics = compute_metrics("dev", dev_df["label"], dev_pred, dev_prob)
    test_metrics = compute_metrics("test", test_df["label"], test_pred, test_prob)

    dev_pred_path = save_predictions("dev", dev_df, dev_pred, dev_prob)
    test_pred_path = save_predictions("test", test_df, test_pred, test_prob)

    joblib.dump(vectorizer, ARTIFACT_DIR / "vectorizer.joblib")
    joblib.dump(model, ARTIFACT_DIR / "model.joblib")

    report = {
        "baseline_name": "tfidf_logistic_regression",
        "input_processed_dir": str(PROCESSED_DIR),
        "split_manifest": str(SPLITS_DIR / "split_manifest.json"),
        "max_text_chars": MAX_TEXT_CHARS,
        "vectorizer_config": VECTORIZER_CONFIG,
        "model_config": MODEL_CONFIG,
        "training": {
            "train_rows": int(len(train_df)),
            "dev_rows": int(len(dev_df)),
            "test_rows": int(len(test_df)),
            "x_train_shape": [int(x_train.shape[0]), int(x_train.shape[1])],
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
            "training_seconds": training_seconds,
        },
        "metrics": {
            "dev": asdict(dev_metrics),
            "test": asdict(test_metrics),
        },
        "artifacts": {
            "vectorizer": str(ARTIFACT_DIR / "vectorizer.joblib"),
            "model": str(ARTIFACT_DIR / "model.joblib"),
            "dev_predictions": str(dev_pred_path),
            "test_predictions": str(test_pred_path),
        },
    }

    report_path = ARTIFACT_DIR / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
