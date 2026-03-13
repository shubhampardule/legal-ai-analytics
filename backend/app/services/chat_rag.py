from __future__ import annotations

import numpy as np

from .prediction import PredictionService
from .text_processing import clean_text, extract_sentences


class ChatRagService:
    def __init__(self, prediction_service: PredictionService) -> None:
        self.prediction_service = prediction_service

    def answer_question(
        self,
        *,
        clean_case_text: str,
        question: str,
        top_k_context: int = 4,
    ) -> dict[str, object]:
        normalized_question = clean_text(question)
        if len(normalized_question) < 3:
            raise ValueError("Question is too short. Please ask a more specific question.")

        sentence_rows = extract_sentences(clean_case_text)
        if not sentence_rows:
            return {
                "answer": "I could not find any readable content in this case.",
                "citations": [],
                "confidence": "low",
            }

        sentence_texts = [row["text"] for row in sentence_rows]

        question_vec = self.prediction_service.vectorizer.transform([normalized_question])
        sentence_matrix = self.prediction_service.vectorizer.transform(sentence_texts)

        raw_scores = (sentence_matrix @ question_vec.T).toarray().ravel().astype(float)
        sentence_norms = np.sqrt(sentence_matrix.multiply(sentence_matrix).sum(axis=1)).A1
        question_norm = float(np.sqrt(float(question_vec.multiply(question_vec).sum())))
        denom = np.maximum(sentence_norms * max(question_norm, 1e-12), 1e-12)
        cosine_scores = raw_scores / denom

        ranked_indices = np.argsort(-cosine_scores)
        citations: list[dict[str, object]] = []

        for idx in ranked_indices:
            score = float(cosine_scores[idx])
            if score <= 0:
                continue
            sentence_info = sentence_rows[int(idx)]
            citations.append(
                {
                    "rank": len(citations) + 1,
                    "score": round(score, 4),
                    "text": sentence_info["text"],
                    "start_char": sentence_info["start_char"],
                    "end_char": sentence_info["end_char"],
                }
            )
            if len(citations) >= top_k_context:
                break

        if not citations:
            return {
                "answer": "I couldn't find a direct answer in the case text for that question. Try asking with more specific legal terms or names.",
                "citations": [],
                "confidence": "low",
            }

        best = citations[0]
        supporting = citations[1:3]

        if supporting:
            supporting_text = " ".join(item["text"] for item in supporting)
            answer = (
                f"Most relevant passage: {best['text']} "
                f"Additional context: {supporting_text}"
            )
        else:
            answer = f"Most relevant passage: {best['text']}"

        confidence = "high" if best["score"] >= 0.35 else "moderate" if best["score"] >= 0.2 else "low"

        return {
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
        }
