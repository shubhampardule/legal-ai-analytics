from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib

from ..config import BASELINE_DIR, LABEL_NAMES, MAX_INPUT_CHARS, MODEL_MAX_TEXT_CHARS
from .text_processing import clean_text


@dataclass
class PredictionArtifacts:
    clean_text: str
    model_text: str
    row_vector: object
    predicted_label_id: int
    predicted_label: str
    accepted_probability: float
    rejected_probability: float
    decision_margin: float
    truncated_for_model: bool


class PredictionService:
    def __init__(self, baseline_dir: Path = BASELINE_DIR) -> None:
        self.vectorizer = joblib.load(baseline_dir / "vectorizer.joblib")
        self.model = joblib.load(baseline_dir / "model.joblib")

    def clean_user_text(self, text: str) -> str:
        raw_text = text or ""
        if len(raw_text) > MAX_INPUT_CHARS:
            raise ValueError(
                f"Input text exceeds the maximum allowed length of {MAX_INPUT_CHARS} characters."
            )
        cleaned = clean_text(raw_text)
        if not cleaned:
            raise ValueError("Input text is empty after cleaning.")
        return cleaned

    def predict_from_clean_text(self, clean_text_value: str) -> PredictionArtifacts:
        model_text = clean_text_value[:MODEL_MAX_TEXT_CHARS]
        row_vector = self.vectorizer.transform([model_text]).tocsr()
        accepted_probability = float(self.model.predict_proba(row_vector)[0, 1])
        decision_margin = float(self.model.decision_function(row_vector)[0])
        predicted_label_id = int(accepted_probability >= 0.5)

        return PredictionArtifacts(
            clean_text=clean_text_value,
            model_text=model_text,
            row_vector=row_vector,
            predicted_label_id=predicted_label_id,
            predicted_label=LABEL_NAMES[predicted_label_id],
            accepted_probability=accepted_probability,
            rejected_probability=float(1.0 - accepted_probability),
            decision_margin=decision_margin,
            truncated_for_model=len(clean_text_value) > len(model_text),
        )

    def to_payload(self, prediction: PredictionArtifacts) -> dict[str, object]:
        return {
            "model_name": "baseline_tfidf_logreg",
            "predicted_label": prediction.predicted_label,
            "accepted_probability": round(prediction.accepted_probability, 6),
            "rejected_probability": round(prediction.rejected_probability, 6),
            "decision_margin": round(prediction.decision_margin, 6),
        }
