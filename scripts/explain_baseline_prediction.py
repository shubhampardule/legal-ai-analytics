from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocess_ildc import clean_text
from train_baseline_tfidf_logreg import MAX_TEXT_CHARS


PROCESSED_DIR = Path("data/processed/ildc")
BASELINE_DIR = Path("artifacts/baseline/tfidf_logreg")
EXPLAINABILITY_DIR = Path("artifacts/explainability")

LABEL_NAMES = {0: "rejected", 1: "accepted"}
SENTENCE_RE = re.compile(r".+?(?:[.!?](?=\s+|$)|;(?=\s+)|$)", re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain a single prediction from the saved baseline model."
    )
    parser.add_argument(
        "--split",
        choices=["train", "dev", "test"],
        help="Processed dataset split to load the case from.",
    )
    parser.add_argument(
        "--case-id",
        help="Case ID from the selected split.",
    )
    parser.add_argument(
        "--raw-text-file",
        type=Path,
        help="Optional plain-text file to explain instead of a saved dataset case.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON file path for saving the explanation.",
    )
    parser.add_argument(
        "--top-k-terms",
        type=int,
        default=10,
        help="Number of top contributing terms to return for each class direction.",
    )
    parser.add_argument(
        "--top-k-sentences",
        type=int,
        default=3,
        help="Number of top supporting and opposing sentences to return.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    using_dataset_case = args.split is not None or args.case_id is not None
    using_raw_text = args.raw_text_file is not None

    if using_dataset_case and using_raw_text:
        raise ValueError("Use either --split/--case-id or --raw-text-file, not both.")

    if using_dataset_case:
        if not args.split or not args.case_id:
            raise ValueError("Both --split and --case-id are required for dataset explanations.")
        return

    if not using_raw_text:
        raise ValueError("Provide either --split and --case-id, or --raw-text-file.")


def load_case(split: str, case_id: str) -> dict[str, object]:
    df = pd.read_parquet(
        PROCESSED_DIR / f"{split}.parquet",
        columns=["id", "label", "clean_text", "clean_char_length", "needs_chunking"],
    )
    matches = df.loc[df["id"] == case_id]
    if matches.empty:
        raise KeyError(f"Case ID not found in split={split}: {case_id}")
    row = matches.iloc[0]
    return {
        "source": "dataset",
        "split": split,
        "case_id": row["id"],
        "true_label": int(row["label"]),
        "clean_text": row["clean_text"],
        "clean_char_length": int(row["clean_char_length"]),
        "needs_chunking": bool(row["needs_chunking"]),
    }


def load_raw_text(path: Path) -> dict[str, object]:
    raw_text = path.read_text(encoding="utf-8")
    cleaned = clean_text(raw_text)
    return {
        "source": "raw_text_file",
        "split": None,
        "case_id": path.stem,
        "true_label": None,
        "clean_text": cleaned,
        "clean_char_length": len(cleaned),
        "needs_chunking": len(cleaned) > MAX_TEXT_CHARS,
    }


def prepare_model_text(text: str) -> str:
    return (text or "")[:MAX_TEXT_CHARS]


def extract_sentences(text: str) -> list[dict[str, object]]:
    sentences: list[dict[str, object]] = []
    for match in SENTENCE_RE.finditer(text):
        raw = match.group(0)
        left_trim = len(raw) - len(raw.lstrip())
        right_trim = len(raw) - len(raw.rstrip())
        start_char = match.start() + left_trim
        end_char = match.end() - right_trim
        sentence_text = text[start_char:end_char].strip()
        if not sentence_text:
            continue
        if not any(char.isalnum() for char in sentence_text):
            continue
        sentences.append(
            {
                "text": sentence_text,
                "start_char": int(start_char),
                "end_char": int(end_char),
                "char_length": int(len(sentence_text)),
            }
        )
    if not sentences and text.strip():
        sentences.append(
            {
                "text": text.strip(),
                "start_char": 0,
                "end_char": len(text.strip()),
                "char_length": len(text.strip()),
            }
        )
    return sentences


def build_term_contributions(
    row_vector,
    vectorizer,
    coefficients: np.ndarray,
    top_k: int,
) -> dict[str, list[dict[str, float | str]]]:
    feature_names = vectorizer.get_feature_names_out()
    records = []
    for feature_idx, tfidf_value in zip(row_vector.indices, row_vector.data):
        coefficient = float(coefficients[feature_idx])
        contribution = float(tfidf_value * coefficient)
        records.append(
            {
                "term": feature_names[feature_idx],
                "tfidf_value": round(float(tfidf_value), 6),
                "coefficient": round(coefficient, 6),
                "contribution": round(contribution, 6),
            }
        )

    accepted_terms = [
        record for record in sorted(records, key=lambda item: item["contribution"], reverse=True)
        if record["contribution"] > 0
    ][:top_k]
    rejected_terms = [
        {
            **record,
            "contribution": round(abs(float(record["contribution"])), 6),
        }
        for record in sorted(records, key=lambda item: item["contribution"])
        if record["contribution"] < 0
    ][:top_k]

    return {
        "accepted": accepted_terms,
        "rejected": rejected_terms,
    }


def build_sentence_evidence(
    model_text: str,
    vectorizer,
    coefficients: np.ndarray,
    predicted_label: int,
    top_k: int,
) -> dict[str, list[dict[str, object]]]:
    sentences = extract_sentences(model_text)
    sentence_texts = [sentence["text"] for sentence in sentences]
    sentence_matrix = vectorizer.transform(sentence_texts)

    accepted_coefficients = np.clip(coefficients, a_min=0.0, a_max=None)
    rejected_coefficients = np.clip(-coefficients, a_min=0.0, a_max=None)

    accepted_scores = np.asarray(sentence_matrix @ accepted_coefficients).ravel()
    rejected_scores = np.asarray(sentence_matrix @ rejected_coefficients).ravel()

    enriched = []
    for index, sentence in enumerate(sentences):
        enriched.append(
            {
                **sentence,
                "accepted_evidence": round(float(accepted_scores[index]), 6),
                "rejected_evidence": round(float(rejected_scores[index]), 6),
            }
        )

    supporting_key = "accepted_evidence" if predicted_label == 1 else "rejected_evidence"
    opposing_key = "rejected_evidence" if predicted_label == 1 else "accepted_evidence"

    supporting = [
        sentence
        for sentence in sorted(enriched, key=lambda item: item[supporting_key], reverse=True)
        if sentence[supporting_key] > 0
    ][:top_k]
    opposing = [
        sentence
        for sentence in sorted(enriched, key=lambda item: item[opposing_key], reverse=True)
        if sentence[opposing_key] > 0
    ][:top_k]

    return {
        "supporting": supporting,
        "opposing": opposing,
        "sentence_count": len(sentences),
    }


def explain_case(
    case_payload: dict[str, object],
    vectorizer,
    model,
    top_k_terms: int,
    top_k_sentences: int,
) -> dict[str, object]:
    model_text = prepare_model_text(str(case_payload["clean_text"]))
    row_vector = vectorizer.transform([model_text]).tocsr()
    coefficients = model.coef_[0]
    intercept = float(model.intercept_[0])
    margin = float(row_vector.dot(coefficients)[0] + intercept)
    accepted_probability = float(model.predict_proba(row_vector)[0, 1])
    predicted_label = int(accepted_probability >= 0.5)

    term_contributions = build_term_contributions(
        row_vector=row_vector[0],
        vectorizer=vectorizer,
        coefficients=coefficients,
        top_k=top_k_terms,
    )
    sentence_evidence = build_sentence_evidence(
        model_text=model_text,
        vectorizer=vectorizer,
        coefficients=coefficients,
        predicted_label=predicted_label,
        top_k=top_k_sentences,
    )

    return {
        "source": case_payload["source"],
        "split": case_payload["split"],
        "case_id": case_payload["case_id"],
        "true_label": LABEL_NAMES.get(case_payload["true_label"])
        if case_payload["true_label"] is not None
        else None,
        "prediction": {
            "predicted_label": LABEL_NAMES[predicted_label],
            "accepted_probability": round(accepted_probability, 6),
            "rejected_probability": round(1.0 - accepted_probability, 6),
            "decision_margin": round(margin, 6),
        },
        "text_summary": {
            "clean_char_length": int(case_payload["clean_char_length"]),
            "model_text_char_length": int(len(model_text)),
            "truncated_for_model": bool(case_payload["clean_char_length"] > len(model_text)),
            "needs_chunking": bool(case_payload["needs_chunking"]),
        },
        "top_term_contributions": term_contributions,
        "sentence_evidence": sentence_evidence,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)

    vectorizer = joblib.load(BASELINE_DIR / "vectorizer.joblib")
    model = joblib.load(BASELINE_DIR / "model.joblib")

    if args.raw_text_file is not None:
        case_payload = load_raw_text(args.raw_text_file)
    else:
        case_payload = load_case(split=args.split, case_id=args.case_id)

    explanation = explain_case(
        case_payload=case_payload,
        vectorizer=vectorizer,
        model=model,
        top_k_terms=args.top_k_terms,
        top_k_sentences=args.top_k_sentences,
    )

    output_path = args.output
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(explanation, indent=2), encoding="utf-8")

    print(json.dumps(explanation, indent=2))


if __name__ == "__main__":
    main()
