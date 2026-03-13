from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd


RETRIEVAL_EMBEDDING_DIR = Path("artifacts/retrieval_case_embeddings")
OUT_DIR = Path("artifacts/similarity_index")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    embeddings_path = RETRIEVAL_EMBEDDING_DIR / "case_embeddings.npy"
    metadata_path = RETRIEVAL_EMBEDDING_DIR / "case_metadata.parquet"
    manifest_path = RETRIEVAL_EMBEDDING_DIR / "embedding_manifest.json"

    embeddings = np.load(embeddings_path).astype(np.float32, copy=False)
    metadata = pd.read_parquet(metadata_path)
    embedding_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape={embeddings.shape}")
    if len(metadata) != embeddings.shape[0]:
        raise ValueError(
            f"Metadata rows do not match embedding rows: metadata={len(metadata)} embeddings={embeddings.shape[0]}"
        )

    dimension = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dimension)
    index.add(np.ascontiguousarray(embeddings))

    index_path = OUT_DIR / "ildc_cases_ip.index"
    faiss.write_index(index, str(index_path))

    search_manifest = {
        "index_type": "faiss.IndexFlatIP",
        "similarity_metric": "cosine_via_inner_product_on_normalized_embeddings",
        "rows": int(embeddings.shape[0]),
        "dimension": dimension,
        "ntotal": int(index.ntotal),
        "embedding_artifacts": {
            "embeddings": str(embeddings_path),
            "metadata": str(metadata_path),
            "embedding_manifest": str(manifest_path),
        },
        "index_artifact": str(index_path),
        "source_embedding_model": embedding_manifest["embedding_model"],
        "document_strategy": embedding_manifest["document_strategy"],
    }
    (OUT_DIR / "index_manifest.json").write_text(
        json.dumps(search_manifest, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(search_manifest, indent=2))


if __name__ == "__main__":
    main()
