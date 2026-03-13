from __future__ import annotations

import re
from collections import defaultdict

from ..config import DEFAULT_TOP_K_SENTENCES, DEFAULT_TOP_K_TERMS
from .prediction import PredictionArtifacts, PredictionService
from .text_processing import extract_sentences


TERM_RE = re.compile(r"\b[a-z]{3,}\b", re.IGNORECASE)
STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "from", "was", "were", "are", "has", "have",
    "had", "his", "her", "its", "their", "into", "upon", "while", "where", "when", "which", "shall",
    "would", "could", "should", "there", "here", "also", "such", "than", "then", "been", "being", "case",
    "court", "petitioner", "respondent", "appellant", "appeal", "judgment", "order", "section", "article",
}


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
        sentence_evidence = self._build_sentence_evidence(
            model_text=prediction.model_text,
            predicted_label_id=prediction.predicted_label_id,
            top_k=top_k_sentences,
        )
        top_terms = self._build_term_contributions(
            sentence_evidence=sentence_evidence,
            top_k=top_k_terms,
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
        sentence_evidence: dict[str, object],
        top_k: int,
    ) -> dict[str, list[dict[str, object]]]:
        supporting = sentence_evidence.get("supporting", [])
        opposing = sentence_evidence.get("opposing", [])

        accepted_weights: dict[str, float] = defaultdict(float)
        accepted_counts: dict[str, int] = defaultdict(int)
        rejected_weights: dict[str, float] = defaultdict(float)
        rejected_counts: dict[str, int] = defaultdict(int)

        for sentence in supporting:
            sentence_text = str(sentence.get("text", ""))
            score = float(sentence.get("accepted_evidence", 0.0))
            for term in TERM_RE.findall(sentence_text.lower()):
                if term in STOPWORDS:
                    continue
                accepted_weights[term] += score
                accepted_counts[term] += 1

        for sentence in opposing:
            sentence_text = str(sentence.get("text", ""))
            score = float(sentence.get("rejected_evidence", 0.0))
            for term in TERM_RE.findall(sentence_text.lower()):
                if term in STOPWORDS:
                    continue
                rejected_weights[term] += score
                rejected_counts[term] += 1

        accepted_terms = [
            {
                "term": term,
                "contribution": round(weight, 6),
                "frequency": int(accepted_counts[term]),
            }
            for term, weight in sorted(accepted_weights.items(), key=lambda item: item[1], reverse=True)[:top_k]
        ]

        rejected_terms = [
            {
                "term": term,
                "contribution": round(weight, 6),
                "frequency": int(rejected_counts[term]),
            }
            for term, weight in sorted(rejected_weights.items(), key=lambda item: item[1], reverse=True)[:top_k]
        ]

        return {
            "accepted": accepted_terms,
            "rejected": rejected_terms,
        }

    def _build_sentence_evidence(
        self,
        model_text: str,
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

        trimmed_sentences = sentences[:120]
        sentence_texts = [sentence["text"] for sentence in trimmed_sentences]
        sentence_scores = self.prediction_service.score_texts_for_outcome(sentence_texts)

        enriched = []
        for index, sentence in enumerate(trimmed_sentences):
            enriched.append(
                {
                    **sentence,
                    "accepted_evidence": round(float(sentence_scores[index]["accepted"]), 6),
                    "rejected_evidence": round(float(sentence_scores[index]["rejected"]), 6),
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
            "sentence_count": len(trimmed_sentences),
        }
