from __future__ import annotations

import json
from pathlib import Path


BASELINE_REPORT = Path("artifacts/baseline/tfidf_logreg/training_report.json")
ADVANCED_REPORT = Path("artifacts/advanced/minilm_embedding_logreg/training_report.json")
OUT_DIR = Path("artifacts/model_comparison")
OUT_PATH = OUT_DIR / "prediction_model_comparison.json"


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def predicted_positive_rate(confusion_matrix: list[list[int]]) -> float:
    true_negative, false_positive = confusion_matrix[0]
    false_negative, true_positive = confusion_matrix[1]
    total = true_negative + false_positive + false_negative + true_positive
    if total == 0:
        return 0.0
    return (false_positive + true_positive) / total


def round4(value: float) -> float:
    return round(float(value), 4)


def build_summary(name: str, report: dict) -> dict:
    dev = report["metrics"]["dev"]
    test = report["metrics"]["test"]
    return {
        "model_name": name,
        "dev": {
            "accuracy": round4(dev["accuracy"]),
            "precision": round4(dev["precision"]),
            "recall": round4(dev["recall"]),
            "f1": round4(dev["f1"]),
            "roc_auc": round4(dev["roc_auc"]),
            "predicted_positive_rate": round4(
                predicted_positive_rate(dev["confusion_matrix"])
            ),
        },
        "test": {
            "accuracy": round4(test["accuracy"]),
            "precision": round4(test["precision"]),
            "recall": round4(test["recall"]),
            "f1": round4(test["f1"]),
            "roc_auc": round4(test["roc_auc"]),
            "predicted_positive_rate": round4(
                predicted_positive_rate(test["confusion_matrix"])
            ),
        },
    }


def metric_deltas(baseline: dict, advanced: dict) -> dict:
    deltas = {}
    for split in ("dev", "test"):
        deltas[split] = {}
        for metric in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "predicted_positive_rate",
        ):
            deltas[split][metric] = round4(
                advanced[split][metric] - baseline[split][metric]
            )
    return deltas


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_report = load_report(BASELINE_REPORT)
    advanced_report = load_report(ADVANCED_REPORT)

    baseline = build_summary("baseline_tfidf_logreg", baseline_report)
    advanced = build_summary("advanced_minilm_embedding_logreg", advanced_report)
    deltas = metric_deltas(baseline, advanced)

    selected_model = "baseline_tfidf_logreg"
    selection_reason = [
        "Baseline is selected as the main prediction model.",
        "It has much stronger test precision, accuracy, and ROC-AUC than the advanced model.",
        "The advanced model improves recall and F1, but its predicted positive rate is unrealistically high.",
        "For the current project stage, the advanced model is better kept as an embedding source for later similarity retrieval rather than as the primary predictor.",
    ]

    comparison = {
        "baseline": baseline,
        "advanced": advanced,
        "advanced_minus_baseline": deltas,
        "selection": {
            "selected_model": selected_model,
            "reasoning": selection_reason,
        },
    }

    OUT_PATH.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
