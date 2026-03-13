from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from ..config import LABEL_NAMES, PROCESSED_DIR, PROCESSED_SPLITS


CASE_ID_YEAR_RE = r"^(\d{4})_"


class CaseLookupService:
    def __init__(self, processed_dir: Path = PROCESSED_DIR) -> None:
        self.processed_dir = processed_dir
        self._datasets = {
            split: ds.dataset(str(self.processed_dir / f"{split}.parquet"), format="parquet")
            for split in PROCESSED_SPLITS
        }
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for split in PROCESSED_SPLITS:
            frame = pd.read_parquet(
                self.processed_dir / f"{split}.parquet",
                columns=["id", "label", "split", "clean_char_length", "needs_chunking"],
            ).copy()
            frame["label_name"] = frame["label"].map(LABEL_NAMES)
            frames.append(frame)
        metadata = pd.concat(frames, ignore_index=True)
        metadata["year"] = (
            metadata["id"].astype(str).str.extract(CASE_ID_YEAR_RE, expand=False).astype(float).fillna(0).astype(int)
        )
        if metadata["id"].duplicated().any():
            raise ValueError("Duplicate case IDs detected in case lookup metadata.")
        return metadata.set_index("id", drop=False)

    def list_cases(
        self,
        limit: int = 50,
        offset: int = 0,
        query: str = None,
        outcome: str | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> dict[str, object]:
        filtered_df = self._metadata
        if query:
            q = query.lower()
            mask = filtered_df["id"].str.lower().str.contains(q) | filtered_df["label_name"].str.lower().str.contains(q)
            filtered_df = filtered_df[mask]

        if outcome in {"accepted", "rejected"}:
            filtered_df = filtered_df[filtered_df["label_name"] == outcome]

        if year_from is not None:
            filtered_df = filtered_df[filtered_df["year"] >= int(year_from)]
        if year_to is not None:
            filtered_df = filtered_df[filtered_df["year"] <= int(year_to)]

        total = len(filtered_df)
        subset = filtered_df.iloc[offset:offset+limit]
        items = []
        for _, row in subset.iterrows():
            items.append({
                "id": row["id"],
                "label": int(row["label"]),
                "label_name": row["label_name"],
                "year": int(row["year"]),
                "split": str(row["split"]),
                "clean_char_length": int(row["clean_char_length"]),
                "needs_chunking": bool(row["needs_chunking"]),
            })
        return {
            "total": total,
            "items": items,
            "limit": limit,
            "offset": offset
        }

    def get_case(self, case_id: str) -> dict[str, object]:
        if case_id not in self._metadata.index:
            raise KeyError(f"Unknown case ID: {case_id}")

        metadata_row = self._metadata.loc[case_id]
        split = str(metadata_row["split"])
        text_table = self._datasets[split].to_table(
            columns=["id", "clean_text"],
            filter=ds.field("id") == case_id,
        )
        if text_table.num_rows != 1:
            raise KeyError(f"Case text not found for case ID: {case_id}")

        clean_text = text_table.column("clean_text")[0].as_py() or ""
        return {
            "id": metadata_row["id"],
            "label": int(metadata_row["label"]),
            "label_name": metadata_row["label_name"],
            "split": split,
            "clean_char_length": int(metadata_row["clean_char_length"]),
            "needs_chunking": bool(metadata_row["needs_chunking"]),
            "clean_text": clean_text,
        }

