from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from preprocess_ildc import clean_text
from train_advanced_minilm_embedding import (
    MODEL_NAME,
    SEGMENT_HEAD_CHARS,
    SEGMENT_TAIL_CHARS,
    TOKEN_MAX_LENGTH,
    make_segments,
    mean_pool,
    resolve_device,
)


INDEX_DIR = Path("artifacts/similarity_index")
RETRIEVAL_EMBEDDING_DIR = Path("artifacts/retrieval_case_embeddings")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve top-k similar legal cases using FAISS over MiniLM embeddings."
    )
    parser.add_argument("--case-id", help="Existing ILDC case ID to use as the query.")
    parser.add_argument(
        "--raw-text-file",
        type=Path,
        help="Optional text file to encode as a new similarity-search query.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of similar cases to return.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON path for saving retrieval results.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if bool(args.case_id) == bool(args.raw_text_file):
        raise ValueError("Use exactly one of --case-id or --raw-text-file.")


def load_index():
    return faiss.read_index(str(INDEX_DIR / "ildc_cases_ip.index"))


def load_metadata() -> pd.DataFrame:
    return pd.read_parquet(RETRIEVAL_EMBEDDING_DIR / "case_metadata.parquet")


def load_embeddings_for_lookup():
    return np.load(RETRIEVAL_EMBEDDING_DIR / "case_embeddings.npy", mmap_mode="r")


def encode_query_text(text: str) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    device = resolve_device()
    model.to(device)
    model.eval()

    segments = make_segments(clean_text(text))
    batch = tokenizer(
        segments,
        padding=True,
        truncation=True,
        max_length=TOKEN_MAX_LENGTH,
        return_tensors="pt",
    )
    batch = {key: value.to(device) for key, value in batch.items()}

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode():
        with autocast_context:
            outputs = model(**batch)
        pooled = mean_pool(outputs.last_hidden_state, batch["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

    pooled_np = pooled.cpu().numpy()
    mean_embedding = pooled_np.mean(axis=0)
    norm = np.linalg.norm(mean_embedding)
    if norm > 0:
        mean_embedding = mean_embedding / norm
    return mean_embedding.astype(np.float32, copy=False)


def build_query_from_case_id(case_id: str, metadata: pd.DataFrame):
    matches = metadata.loc[metadata["id"] == case_id]
    if matches.empty:
        raise KeyError(f"Case ID not found in retrieval metadata: {case_id}")
    row = matches.iloc[0]
    embeddings = load_embeddings_for_lookup()
    embedding_row = int(row["embedding_row"])
    query_vector = np.asarray(embeddings[embedding_row], dtype=np.float32)
    return {
        "query_source": "existing_case",
        "case_id": case_id,
        "embedding_row": embedding_row,
        "split": row["split"],
        "label_name": row["label_name"],
        "needs_chunking": bool(row["needs_chunking"]),
        "clean_char_length": int(row["clean_char_length"]),
        "preview_text": row["preview_text"],
    }, query_vector


def build_query_from_raw_text(path: Path):
    raw_text = path.read_text(encoding="utf-8")
    cleaned = clean_text(raw_text)
    query_vector = encode_query_text(cleaned)
    return {
        "query_source": "raw_text_file",
        "case_id": path.stem,
        "embedding_row": None,
        "split": None,
        "label_name": None,
        "needs_chunking": len(cleaned) > (SEGMENT_HEAD_CHARS + SEGMENT_TAIL_CHARS),
        "clean_char_length": len(cleaned),
        "preview_text": cleaned[:400],
    }, query_vector


def search(
    index,
    metadata: pd.DataFrame,
    query_info: dict[str, object],
    query_vector: np.ndarray,
    top_k: int,
) -> list[dict[str, object]]:
    query_matrix = np.ascontiguousarray(query_vector.reshape(1, -1).astype(np.float32))
    search_k = top_k + 1 if query_info["embedding_row"] is not None else top_k
    scores, rows = index.search(query_matrix, search_k)

    results: list[dict[str, object]] = []
    for score, row_idx in zip(scores[0], rows[0]):
        if row_idx < 0:
            continue
        if query_info["embedding_row"] is not None and int(row_idx) == int(query_info["embedding_row"]):
            continue

        meta = metadata.iloc[int(row_idx)]
        results.append(
            {
                "rank": len(results) + 1,
                "similarity_score": round(float(score), 6),
                "embedding_row": int(meta["embedding_row"]),
                "case_id": meta["id"],
                "split": meta["split"],
                "label": int(meta["label"]),
                "label_name": meta["label_name"],
                "clean_char_length": int(meta["clean_char_length"]),
                "needs_chunking": bool(meta["needs_chunking"]),
                "preview_text": meta["preview_text"],
            }
        )
        if len(results) >= top_k:
            break
    return results


def main() -> None:
    args = parse_args()
    validate_args(args)

    metadata = load_metadata()
    index = load_index()

    if args.case_id:
        query_info, query_vector = build_query_from_case_id(args.case_id, metadata)
    else:
        query_info, query_vector = build_query_from_raw_text(args.raw_text_file)

    results = search(
        index=index,
        metadata=metadata,
        query_info=query_info,
        query_vector=query_vector,
        top_k=args.top_k,
    )

    payload = {
        "retrieval_model": {
            "embedding_model": MODEL_NAME,
            "index_type": "faiss.IndexFlatIP",
            "similarity_metric": "cosine_similarity",
        },
        "query": query_info,
        "results": results,
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
