from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROCESSED_DIR = Path("data/processed/ildc")
ADVANCED_DIR = Path("artifacts/advanced/minilm_embedding_logreg")
OUT_DIR = Path("artifacts/retrieval_case_embeddings")

SPLITS = ("train", "dev", "test")
LABEL_NAMES = {0: "rejected", 1: "accepted"}
PREVIEW_CHARS = 400


def load_split_metadata(split: str) -> pd.DataFrame:
    df = pd.read_parquet(
        PROCESSED_DIR / f"{split}.parquet",
        columns=["id", "label", "split", "clean_char_length", "needs_chunking", "clean_text"],
    ).copy()
    df["label_name"] = df["label"].map(LABEL_NAMES)
    df["preview_text"] = df["clean_text"].fillna("").str.slice(0, PREVIEW_CHARS)
    return df


def load_split_embeddings(split: str) -> np.ndarray:
    path = ADVANCED_DIR / f"{split}_embeddings.npy"
    return np.load(path).astype(np.float32, copy=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metadata_frames: list[pd.DataFrame] = []
    embedding_blocks: list[np.ndarray] = []
    split_offsets: dict[str, dict[str, int]] = {}
    current_start = 0

    for split in SPLITS:
        metadata = load_split_metadata(split)
        embeddings = load_split_embeddings(split)

        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embedding matrix for split={split}, got {embeddings.shape}")
        if len(metadata) != embeddings.shape[0]:
            raise ValueError(
                f"Row count mismatch for split={split}: metadata={len(metadata)} embeddings={embeddings.shape[0]}"
            )

        end_exclusive = current_start + len(metadata)
        split_offsets[split] = {
            "start_row": current_start,
            "end_row_exclusive": end_exclusive,
            "rows": len(metadata),
        }

        metadata = metadata.assign(embedding_row=np.arange(current_start, end_exclusive))
        metadata_frames.append(
            metadata[
                [
                    "embedding_row",
                    "id",
                    "label",
                    "label_name",
                    "split",
                    "clean_char_length",
                    "needs_chunking",
                    "preview_text",
                ]
            ]
        )
        embedding_blocks.append(embeddings)
        current_start = end_exclusive

    case_embeddings = np.vstack(embedding_blocks).astype(np.float32, copy=False)
    case_metadata = pd.concat(metadata_frames, ignore_index=True)

    if case_metadata["id"].duplicated().any():
        duplicates = int(case_metadata["id"].duplicated().sum())
        raise ValueError(f"Duplicate case IDs found in combined metadata: {duplicates}")

    if len(case_metadata) != case_embeddings.shape[0]:
        raise ValueError("Combined metadata and embedding row counts do not match.")

    embedding_path = OUT_DIR / "case_embeddings.npy"
    metadata_path = OUT_DIR / "case_metadata.parquet"
    manifest_path = OUT_DIR / "embedding_manifest.json"

    np.save(embedding_path, case_embeddings)
    case_metadata.to_parquet(metadata_path, index=False)

    norms = np.linalg.norm(case_embeddings, axis=1)
    label_counts = (
        case_metadata["label_name"].value_counts().sort_index().to_dict()
    )
    split_counts = case_metadata["split"].value_counts().sort_index().to_dict()

    manifest = {
        "embedding_source": str(ADVANCED_DIR / "training_report.json"),
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "document_strategy": "head_tail_segment_mean_pooling",
        "rows": int(case_embeddings.shape[0]),
        "embedding_dimension": int(case_embeddings.shape[1]),
        "dtype": str(case_embeddings.dtype),
        "split_counts": {key: int(value) for key, value in split_counts.items()},
        "label_counts": {key: int(value) for key, value in label_counts.items()},
        "split_offsets": split_offsets,
        "preview_chars": PREVIEW_CHARS,
        "embedding_norms": {
            "min": round(float(norms.min()), 6),
            "max": round(float(norms.max()), 6),
            "mean": round(float(norms.mean()), 6),
        },
        "artifacts": {
            "case_embeddings": str(embedding_path),
            "case_metadata": str(metadata_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
