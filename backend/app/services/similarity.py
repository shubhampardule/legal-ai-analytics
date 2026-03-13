from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from ..config import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    RETRIEVAL_EMBEDDING_DIR,
    SIMILARITY_DISCLAIMER,
    SIMILARITY_INDEX_DIR,
    SIMILARITY_MODEL_NAME,
)


SEGMENT_HEAD_CHARS = 4000
SEGMENT_TAIL_CHARS = 4000
TOKEN_MAX_LENGTH = 256
CASE_ID_YEAR_RE = r"^(\d{4})_"


def make_segments(text: str) -> list[str]:
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


class SimilarityService:
    def __init__(
        self,
        index_dir: Path = SIMILARITY_INDEX_DIR,
        retrieval_embedding_dir: Path = RETRIEVAL_EMBEDDING_DIR,
    ) -> None:
        self.index = faiss.read_index(str(index_dir / "ildc_cases_ip.index"))
        self.metadata = pd.read_parquet(retrieval_embedding_dir / "case_metadata.parquet")
        self.metadata["year"] = (
            self.metadata["id"].astype(str).str.extract(CASE_ID_YEAR_RE, expand=False).astype(float).fillna(0).astype(int)
        )
        self.metadata_by_id = self.metadata.set_index("id", drop=False)
        self.embeddings = np.load(
            retrieval_embedding_dir / "case_embeddings.npy",
            mmap_mode="r",
        )
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def search_by_case_id(
        self,
        case_id: str,
        top_k: int = DEFAULT_TOP_K,
        outcome: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> dict[str, object]:
        self._validate_top_k(top_k)
        if case_id not in self.metadata_by_id.index:
            raise KeyError(f"Unknown case ID for similarity search: {case_id}")

        row = self.metadata_by_id.loc[case_id]
        query_row = int(row["embedding_row"])
        query_vector = np.asarray(self.embeddings[query_row], dtype=np.float32)
        query_info = {
            "source": "case_id",
            "case_id": case_id,
            "split": row["split"],
            "label": int(row["label"]),
            "label_name": row["label_name"],
            "clean_char_length": int(row["clean_char_length"]),
            "needs_chunking": bool(row["needs_chunking"]),
            "preview_text": row["preview_text"],
            "year": int(row.get("year", 0)),
        }
        return {
            "embedding_model": SIMILARITY_MODEL_NAME,
            "index_type": "faiss.IndexFlatIP",
            "similarity_metric": "cosine_similarity",
            "top_k": top_k,
            "filters": {
                "outcome": outcome,
                "year_from": year_from,
                "year_to": year_to,
            },
            "disclaimer": SIMILARITY_DISCLAIMER,
            "query": query_info,
            "results": self._search(
                query_vector,
                top_k,
                exclude_row=query_row,
                outcome=outcome,
                year_from=year_from,
                year_to=year_to,
            ),
        }

    def search_by_clean_text(
        self,
        clean_text: str,
        top_k: int = DEFAULT_TOP_K,
        outcome: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> dict[str, object]:
        self._validate_top_k(top_k)
        query_vector = self._encode_text(clean_text)
        query_info = {
            "source": "text",
            "case_id": None,
            "split": None,
            "label": None,
            "label_name": None,
            "clean_char_length": int(len(clean_text)),
            "needs_chunking": bool(len(clean_text) > (SEGMENT_HEAD_CHARS + SEGMENT_TAIL_CHARS)),
            "preview_text": clean_text[:400],
        }
        return {
            "embedding_model": SIMILARITY_MODEL_NAME,
            "index_type": "faiss.IndexFlatIP",
            "similarity_metric": "cosine_similarity",
            "top_k": top_k,
            "filters": {
                "outcome": outcome,
                "year_from": year_from,
                "year_to": year_to,
            },
            "disclaimer": SIMILARITY_DISCLAIMER,
            "query": query_info,
            "results": self._search(
                query_vector,
                top_k,
                exclude_row=None,
                outcome=outcome,
                year_from=year_from,
                year_to=year_to,
            ),
        }

    def _search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        exclude_row: int | None,
        outcome: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> list[dict[str, object]]:
        search_k = min(len(self.metadata), max(top_k * 10, top_k + 1))
        scores, indices = self.index.search(
            np.ascontiguousarray(query_vector.reshape(1, -1).astype(np.float32)),
            search_k,
        )

        results: list[dict[str, object]] = []
        for score, row_idx in zip(scores[0], indices[0]):
            if row_idx < 0:
                continue
            if exclude_row is not None and int(row_idx) == int(exclude_row):
                continue
            meta = self.metadata.iloc[int(row_idx)]
            if outcome in {"accepted", "rejected"} and str(meta["label_name"]) != outcome:
                continue
            case_year = int(meta.get("year", 0))
            if year_from is not None and case_year < int(year_from):
                continue
            if year_to is not None and case_year > int(year_to):
                continue
            results.append(
                {
                    "rank": len(results) + 1,
                    "case_id": meta["id"],
                    "split": meta["split"],
                    "label": int(meta["label"]),
                    "label_name": meta["label_name"],
                    "year": case_year,
                    "similarity_score": round(float(score), 6),
                    "clean_char_length": int(meta["clean_char_length"]),
                    "needs_chunking": bool(meta["needs_chunking"]),
                    "preview_text": meta["preview_text"],
                }
            )
            if len(results) >= top_k:
                break
        return results

    def _encode_text(self, clean_text: str) -> np.ndarray:
        segments = make_segments(clean_text)
        batch = self.tokenizer(
            segments,
            padding=True,
            truncation=True,
            max_length=TOKEN_MAX_LENGTH,
            return_tensors="pt",
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else nullcontext()
        )

        with torch.inference_mode():
            with autocast_context:
                outputs = self.model(**batch)
            pooled = mean_pool(outputs.last_hidden_state, batch["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        pooled_np = pooled.cpu().numpy()
        mean_embedding = pooled_np.mean(axis=0)
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding = mean_embedding / norm
        return mean_embedding.astype(np.float32, copy=False)

    def _validate_top_k(self, top_k: int) -> None:
        if top_k < 1 or top_k > MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}.")

    def _load_tokenizer(self):
        try:
            return AutoTokenizer.from_pretrained(
                SIMILARITY_MODEL_NAME,
                local_files_only=True,
            )
        except OSError:
            return AutoTokenizer.from_pretrained(SIMILARITY_MODEL_NAME)

    def _load_model(self):
        try:
            return AutoModel.from_pretrained(
                SIMILARITY_MODEL_NAME,
                local_files_only=True,
            )
        except OSError:
            return AutoModel.from_pretrained(SIMILARITY_MODEL_NAME)
