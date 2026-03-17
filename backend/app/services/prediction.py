from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

import torch
from transformers import pipeline

from ..config import (
    LABEL_NAMES,
    MAX_INPUT_CHARS,
    PREDICTION_BATCH_SIZE,
    PREDICTION_CHUNK_STRIDE_CHARS,
    PREDICTION_MAX_CHUNKS,
    PREDICTION_MAX_CHUNK_CHARS,
    PREDICTION_MODEL_NAME,
)
from .text_processing import clean_text


@dataclass
class PredictionArtifacts:
    clean_text: str
    model_text: str
    chunk_count: int
    predicted_label_id: int
    predicted_label: str
    accepted_probability: float
    rejected_probability: float
    decision_margin: float
    truncated_for_model: bool


class PredictionService:
    CANDIDATE_LABELS = ["accepted", "rejected"]
    HYPOTHESIS_TEMPLATE = "The legal outcome of this case is {}."
    RELEVANCE_LABELS = ["relevant", "not relevant"]
    RELEVANCE_HYPOTHESIS_TEMPLATE = "This candidate legal case is {} to the query case."

    def __init__(self) -> None:
        if torch.cuda.is_available():
            device_index = 0
            device_name = torch.cuda.get_device_name(0)
            print(f"[PredictionService] GPU hardware detected: {device_name}. Accelerating with CUDA.")
        else:
            device_index = -1
            print("[PredictionService] WARNING: No compatible GPU found. Falling back to CPU (expect slower performance). Please install CUDA-supported PyTorch.")
        
        self.model_name = PREDICTION_MODEL_NAME
        self.classifier = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            tokenizer=self.model_name,
            device=device_index,
        )

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

    def _run_zero_shot(
        self,
        texts: list[str],
        candidate_labels: list[str],
        hypothesis_template: str,
    ) -> list[dict[str, float]]:
        if not texts:
            return []

        outputs = self.classifier(
            texts,
            candidate_labels=candidate_labels,
            multi_label=False,
            hypothesis_template=hypothesis_template,
            batch_size=PREDICTION_BATCH_SIZE,
        )
        if isinstance(outputs, dict):
            outputs = [outputs]

        normalized_scores: list[dict[str, float]] = []
        for output in outputs:
            labels = [str(label).strip().lower() for label in output.get("labels", [])]
            scores = [float(score) for score in output.get("scores", [])]
            label_to_score = {label: score for label, score in zip(labels, scores)}
            target_scores = {
                label: float(label_to_score.get(label.lower(), 0.0))
                for label in candidate_labels
            }
            denom = max(sum(target_scores.values()), 1e-12)
            normalized_scores.append(
                {
                    label.lower(): (score / denom)
                    for label, score in target_scores.items()
                }
            )

        return normalized_scores

    def score_texts_for_outcome(self, texts: list[str]) -> list[dict[str, float]]:
        return self._run_zero_shot(
            texts=texts,
            candidate_labels=self.CANDIDATE_LABELS,
            hypothesis_template=self.HYPOTHESIS_TEMPLATE,
        )

    def score_relevance_pairs(self, query_text: str, candidate_texts: list[str]) -> list[float]:
        query = (query_text or "").strip()
        if not query:
            return [0.0 for _ in candidate_texts]

        merged_inputs: list[str] = []
        for candidate in candidate_texts:
            candidate_text = (candidate or "").strip()
            merged_inputs.append(
                f"Query legal case: {query[:800]}\nCandidate legal case: {candidate_text[:800]}"
            )

        scores = self._run_zero_shot(
            texts=merged_inputs,
            candidate_labels=self.RELEVANCE_LABELS,
            hypothesis_template=self.RELEVANCE_HYPOTHESIS_TEMPLATE,
        )
        return [float(item.get("relevant", 0.0)) for item in scores]

    def _make_chunks(self, text: str) -> tuple[list[str], bool]:
        if len(text) <= PREDICTION_MAX_CHUNK_CHARS:
            return [text], False

        chunks: list[str] = []
        step = max(1, PREDICTION_CHUNK_STRIDE_CHARS)
        start = 0
        text_len = len(text)
        while start < text_len and len(chunks) < PREDICTION_MAX_CHUNKS:
            end = min(start + PREDICTION_MAX_CHUNK_CHARS, text_len)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= text_len:
                break
            start += step

        reached_end = (start + PREDICTION_MAX_CHUNK_CHARS) >= text_len
        truncated = not reached_end
        if not chunks:
            return [text[:PREDICTION_MAX_CHUNK_CHARS]], len(text) > PREDICTION_MAX_CHUNK_CHARS
        return chunks, truncated

    def predict_from_clean_text(self, clean_text_value: str) -> PredictionArtifacts:
        chunks, truncated_for_model = self._make_chunks(clean_text_value)
        chunk_scores = self.score_texts_for_outcome(chunks)
        accepted_probability = float(mean(score["accepted"] for score in chunk_scores))
        rejected_probability = float(mean(score["rejected"] for score in chunk_scores))
        denom = max(accepted_probability + rejected_probability, 1e-12)
        accepted_probability = accepted_probability / denom
        rejected_probability = rejected_probability / denom
        decision_margin = float(accepted_probability - rejected_probability)
        predicted_label_id = int(accepted_probability >= rejected_probability)

        return PredictionArtifacts(
            clean_text=clean_text_value,
            model_text=" ".join(chunks),
            chunk_count=len(chunks),
            predicted_label_id=predicted_label_id,
            predicted_label=LABEL_NAMES[predicted_label_id],
            accepted_probability=accepted_probability,
            rejected_probability=rejected_probability,
            decision_margin=decision_margin,
            truncated_for_model=truncated_for_model,
        )

    def to_payload(self, prediction: PredictionArtifacts) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "predicted_label": prediction.predicted_label,
            "accepted_probability": round(prediction.accepted_probability, 6),
            "rejected_probability": round(prediction.rejected_probability, 6),
            "decision_margin": round(prediction.decision_margin, 6),
            "chunk_count": prediction.chunk_count,
        }
