from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq


PROCESSED_DIR = Path("data/processed/ildc")
SPLITS_DIR = Path("data/splits/ildc")
EXPECTED_SPLITS = ("train", "dev", "test")


def sha256_lines(lines: list[str]) -> str:
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def py_value(value):
    return value.as_py() if hasattr(value, "as_py") else value


def label_counts(table) -> dict[str, int]:
    counts = pc.value_counts(table["label"]).to_pylist()
    return {str(py_value(item["values"])): int(py_value(item["counts"])) for item in counts}


def main() -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "dataset": "ILDC",
        "variant": "ILDCmulti",
        "source_processed_dir": str(PROCESSED_DIR),
        "splits": {},
        "checks": {
            "duplicate_ids_within_split": {},
            "overlap_counts": {},
            "all_splits_disjoint": True,
        },
    }

    ids_by_split: dict[str, set[str]] = {}

    for split in EXPECTED_SPLITS:
        path = PROCESSED_DIR / f"{split}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing processed split file: {path}")

        table = pq.read_table(path, columns=["id", "label", "split"])
        split_values = {py_value(value) for value in pc.unique(table["split"]).to_pylist()}
        if split_values != {split}:
            raise ValueError(f"Unexpected split values in {path}: {split_values}")

        ids = table["id"].to_pylist()
        unique_ids = sorted(set(ids))
        duplicate_count = len(ids) - len(unique_ids)
        ids_by_split[split] = set(unique_ids)

        ids_path = SPLITS_DIR / f"{split}_ids.txt"
        ids_path.write_text("\n".join(unique_ids) + "\n", encoding="utf-8")

        manifest["splits"][split] = {
            "parquet_file": str(path),
            "ids_file": str(ids_path),
            "rows": table.num_rows,
            "unique_ids": len(unique_ids),
            "label_counts": label_counts(table),
            "split_values": sorted(split_values),
            "sha256_sorted_ids": sha256_lines(unique_ids),
        }
        manifest["checks"]["duplicate_ids_within_split"][split] = duplicate_count

    overlap_pairs = [("train", "dev"), ("train", "test"), ("dev", "test")]
    for left, right in overlap_pairs:
        overlap = len(ids_by_split[left] & ids_by_split[right])
        manifest["checks"]["overlap_counts"][f"{left}__{right}"] = overlap
        if overlap:
            manifest["checks"]["all_splits_disjoint"] = False

    if not manifest["checks"]["all_splits_disjoint"]:
        raise ValueError("Split overlap detected. Reproducible split lock failed.")

    manifest_path = SPLITS_DIR / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
