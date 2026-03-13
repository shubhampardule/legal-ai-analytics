from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .prediction import PredictionService
from .text_processing import clean_text, extract_sentences


CHAT_MAX_SENTENCES = 140
CHAT_CANDIDATE_POOL_MULTIPLIER = 6
CHAT_MAX_CANDIDATE_POOL = 36
CHAT_TFIDF_WEIGHT = 0.4
CHAT_RELEVANCE_WEIGHT = 0.6
CHAT_MIN_CONFIDENCE_TO_ANSWER = 0.42


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

        sentence_rows = sentence_rows[:CHAT_MAX_SENTENCES]

        sentence_texts = [row["text"] for row in sentence_rows]

        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=25_000,
            min_df=1,
        )
        sentence_matrix = vectorizer.fit_transform(sentence_texts)
        question_vec = vectorizer.transform([normalized_question])

        raw_scores = (sentence_matrix @ question_vec.T).toarray().ravel().astype(float)
        sentence_norms = np.sqrt(sentence_matrix.multiply(sentence_matrix).sum(axis=1)).A1
        question_norm = float(np.sqrt(float(question_vec.multiply(question_vec).sum())))
        denom = np.maximum(sentence_norms * max(question_norm, 1e-12), 1e-12)
        cosine_scores = raw_scores / denom

        pool_size = min(
            len(sentence_rows),
            max(top_k_context * CHAT_CANDIDATE_POOL_MULTIPLIER, top_k_context + 2, 8),
            CHAT_MAX_CANDIDATE_POOL,
        )

        lexical_ranked_indices = np.argsort(-cosine_scores)[:pool_size]
        lexical_candidates = [sentence_texts[int(idx)] for idx in lexical_ranked_indices]
        relevance_scores = self.prediction_service.score_relevance_pairs(
            query_text=normalized_question,
            candidate_texts=lexical_candidates,
        )

        blended_items: list[dict[str, object]] = []
        for candidate_pos, sent_idx in enumerate(lexical_ranked_indices):
            lexical = float(max(0.0, min(1.0, cosine_scores[int(sent_idx)])))
            relevance = float(relevance_scores[candidate_pos]) if candidate_pos < len(relevance_scores) else 0.0
            blended = CHAT_TFIDF_WEIGHT * lexical + CHAT_RELEVANCE_WEIGHT * relevance
            blended_items.append(
                {
                    "sent_idx": int(sent_idx),
                    "lexical_score": lexical,
                    "relevance_score": relevance,
                    "blended_score": blended,
                }
            )

        blended_items.sort(key=lambda item: float(item["blended_score"]), reverse=True)

        citations: list[dict[str, object]] = []

        for item in blended_items:
            score = float(item["blended_score"])
            if score <= 0:
                continue
            sentence_info = sentence_rows[int(item["sent_idx"])]
            citations.append(
                {
                    "rank": len(citations) + 1,
                    "score": round(score, 4),
                    "lexical_score": round(float(item["lexical_score"]), 4),
                    "relevance_score": round(float(item["relevance_score"]), 4),
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
        supporting = citations[1:min(3, len(citations))]

        if float(best["score"]) < CHAT_MIN_CONFIDENCE_TO_ANSWER:
            return {
                "answer": "I’m not confident enough to answer this from the provided case text. Please ask a narrower question (for example: specific judge name, statute, date, or final holding).",
                "citations": citations,
                "confidence": "low",
            }

        if supporting:
            supporting_text = " ".join(item["text"] for item in supporting)
            answer = (
                f"Based on the case text, the strongest supporting passage is: {best['text']} "
                f"Additional supporting context: {supporting_text}"
            )
        else:
            answer = f"Based on the case text, the strongest supporting passage is: {best['text']}"

        confidence = "high" if best["score"] >= 0.7 else "moderate" if best["score"] >= 0.5 else "low"

        return {
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
        }
