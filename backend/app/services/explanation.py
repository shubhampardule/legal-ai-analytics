from __future__ import annotations

import numpy as np

from ..config import DEFAULT_TOP_K_SENTENCES, DEFAULT_TOP_K_TERMS
from .prediction import PredictionArtifacts, PredictionService
from .text_processing import extract_sentences


class ExplanationService:
    def __init__(self, prediction_service: PredictionService) -> None:
        self.prediction_service = prediction_service

    def explain_prediction(
        self,
        prediction: PredictionArtifacts,
        clean_char_length: int,
        needs_chunking: bool,
        top_k_terms: int = DEFAULT_TOP_K_TERMS,
        top_k_sentences: int = DEFAULT_TOP_K_SENTENCES,
    ) -> dict[str, object]:
        coefficients = self.prediction_service.model.coef_[0]
        top_terms = self._build_term_contributions(
            row_vector=prediction.row_vector[0],
            coefficients=coefficients,
            top_k=top_k_terms,
        )
        sentence_evidence = self._build_sentence_evidence(
            model_text=prediction.model_text,
            coefficients=coefficients,
            predicted_label_id=prediction.predicted_label_id,
            top_k=top_k_sentences,
        )

        return {
            "text_summary": {
                "clean_char_length": int(clean_char_length),
                "model_text_char_length": int(len(prediction.model_text)),
                "truncated_for_model": bool(prediction.truncated_for_model),
                "needs_chunking": bool(needs_chunking),
            },
            "top_term_contributions": top_terms,
            "sentence_evidence": sentence_evidence,
        }

    def _build_term_contributions(
        self,
        row_vector,
        coefficients: np.ndarray,
        top_k: int,
    ) -> dict[str, list[dict[str, object]]]:
        feature_names = self.prediction_service.vectorizer.get_feature_names_out()
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
            item
            for item in sorted(records, key=lambda record: record["contribution"], reverse=True)
            if item["contribution"] > 0
        ][:top_k]

        rejected_terms = [
            {
                **item,
                "contribution": round(abs(float(item["contribution"])), 6),
            }
            for item in sorted(records, key=lambda record: record["contribution"])
            if item["contribution"] < 0
        ][:top_k]

        return {
            "accepted": accepted_terms,
            "rejected": rejected_terms,
        }

    def _build_sentence_evidence(
        self,
        model_text: str,
        coefficients: np.ndarray,
        predicted_label_id: int,
        top_k: int,
    ) -> dict[str, object]:
        sentences = extract_sentences(model_text)
        if not sentences:
            return {
                "supporting": [],
                "opposing": [],
                "sentence_count": 0,
            }

        sentence_texts = [sentence["text"] for sentence in sentences]
        sentence_matrix = self.prediction_service.vectorizer.transform(sentence_texts)
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

        supporting_key = "accepted_evidence" if predicted_label_id == 1 else "rejected_evidence"
        opposing_key = "rejected_evidence" if predicted_label_id == 1 else "accepted_evidence"

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
