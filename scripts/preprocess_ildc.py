from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


RAW_DIR = Path("data/raw/ildc")
PROCESSED_DIR = Path("data/processed/ildc")
LONG_DOCUMENT_CHAR_THRESHOLD = 50_000

SIGNATURE_NOT_VERIFIED_RE = re.compile(r"Signature Not Verified", re.IGNORECASE)
DIGITAL_SIGNATURE_RE = re.compile(
    r"Digitally signed by [A-Z][A-Z .]{1,80}", re.IGNORECASE
)
DATE_STAMP_RE = re.compile(r"Date:\s*[\d./:\- ]{6,40}", re.IGNORECASE)
EMAIL_LIKE_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b", re.IGNORECASE)
HYPHENATED_LINEBREAK_RE = re.compile(r"([A-Za-z])-\s*\n\s*([A-Za-z])")
MULTI_NEWLINE_DASH_RE = re.compile(r"\s*--\s*")
NEWLINE_RE = re.compile(r"\s*\n\s*")
MULTISPACE_RE = re.compile(r"[ \t\f\v]+")


@dataclass
class SplitStats:
    rows: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)
    empty_clean_text_rows: int = 0
    long_document_rows: int = 0
    raw_char_total: int = 0
    clean_char_total: int = 0
    raw_char_min: int | None = None
    raw_char_max: int | None = None
    clean_char_min: int | None = None
    clean_char_max: int | None = None

    def observe(self, label: int, raw_len: int, clean_len: int, is_long: bool) -> None:
        self.rows += 1
        self.label_counts[str(label)] = self.label_counts.get(str(label), 0) + 1
        self.raw_char_total += raw_len
        self.clean_char_total += clean_len
        self.raw_char_min = raw_len if self.raw_char_min is None else min(self.raw_char_min, raw_len)
        self.raw_char_max = raw_len if self.raw_char_max is None else max(self.raw_char_max, raw_len)
        self.clean_char_min = (
            clean_len if self.clean_char_min is None else min(self.clean_char_min, clean_len)
        )
        self.clean_char_max = (
            clean_len if self.clean_char_max is None else max(self.clean_char_max, clean_len)
        )
        if clean_len == 0:
            self.empty_clean_text_rows += 1
        if is_long:
            self.long_document_rows += 1

    def as_dict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "label_counts": self.label_counts,
            "empty_clean_text_rows": self.empty_clean_text_rows,
            "long_document_rows": self.long_document_rows,
            "raw_char_length": {
                "min": self.raw_char_min,
                "max": self.raw_char_max,
                "mean": round(self.raw_char_total / self.rows, 2) if self.rows else 0.0,
            },
            "clean_char_length": {
                "min": self.clean_char_min,
                "max": self.clean_char_max,
                "mean": round(self.clean_char_total / self.rows, 2) if self.rows else 0.0,
            },
        }


OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("label", pa.int64()),
        pa.field("split", pa.string()),
        pa.field("source_file", pa.string()),
        pa.field("raw_char_length", pa.int32()),
        pa.field("clean_char_length", pa.int32()),
        pa.field("needs_chunking", pa.bool_()),
        pa.field("clean_text", pa.string()),
    ]
)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = SIGNATURE_NOT_VERIFIED_RE.sub(" ", text)
    text = DIGITAL_SIGNATURE_RE.sub(" ", text)
    text = DATE_STAMP_RE.sub(" ", text)
    text = EMAIL_LIKE_RE.sub(" ", text)
    text = HYPHENATED_LINEBREAK_RE.sub(r"\1\2", text)
    text = MULTI_NEWLINE_DASH_RE.sub(" ", text)
    text = NEWLINE_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def split_inputs(raw_dir: Path) -> dict[str, list[Path]]:
    return {
        "train": sorted(raw_dir.glob("multi_train-*.parquet")),
        "dev": sorted(raw_dir.glob("multi_dev-*.parquet")),
        "test": sorted(raw_dir.glob("test-*.parquet")),
    }


def write_processed_split(
    split: str,
    input_files: list[Path],
    output_dir: Path,
    batch_size: int,
    char_threshold: int,
) -> dict[str, object]:
    stats = SplitStats()
    writer: pq.ParquetWriter | None = None
    out_path = output_dir / f"{split}.parquet"

    if out_path.exists():
        out_path.unlink()

    try:
        for input_path in input_files:
            parquet_file = pq.ParquetFile(input_path)
            for batch in parquet_file.iter_batches(
                batch_size=batch_size,
                columns=["id", "text", "label"],
            ):
                rows = batch.to_pylist()
                processed = {
                    "id": [],
                    "label": [],
                    "split": [],
                    "source_file": [],
                    "raw_char_length": [],
                    "clean_char_length": [],
                    "needs_chunking": [],
                    "clean_text": [],
                }

                for row in rows:
                    raw_text = row["text"] or ""
                    cleaned = clean_text(raw_text)
                    raw_len = len(raw_text)
                    clean_len = len(cleaned)
                    label = int(row["label"])
                    is_long = clean_len > char_threshold

                    processed["id"].append(row["id"])
                    processed["label"].append(label)
                    processed["split"].append(split)
                    processed["source_file"].append(input_path.name)
                    processed["raw_char_length"].append(raw_len)
                    processed["clean_char_length"].append(clean_len)
                    processed["needs_chunking"].append(is_long)
                    processed["clean_text"].append(cleaned)

                    stats.observe(
                        label=label,
                        raw_len=raw_len,
                        clean_len=clean_len,
                        is_long=is_long,
                    )

                table = pa.table(processed, schema=OUTPUT_SCHEMA)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
                writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    return {
        "output_file": str(out_path),
        "input_files": [str(path) for path in input_files],
        "stats": stats.as_dict(),
    }


def build_report(
    per_split: dict[str, dict[str, object]],
    char_threshold: int,
    batch_size: int,
) -> dict[str, object]:
    total_rows = 0
    total_long_rows = 0
    total_empty_rows = 0
    total_label_counts: dict[str, int] = {}

    for split_report in per_split.values():
        stats = split_report["stats"]
        total_rows += stats["rows"]
        total_long_rows += stats["long_document_rows"]
        total_empty_rows += stats["empty_clean_text_rows"]
        for label, count in stats["label_counts"].items():
            total_label_counts[label] = total_label_counts.get(label, 0) + count

    return {
        "preprocessing_version": 1,
        "raw_dir": str(RAW_DIR),
        "processed_dir": str(PROCESSED_DIR),
        "char_threshold_for_chunking": char_threshold,
        "batch_size": batch_size,
        "cleaning_steps": [
            "unicode normalization with NFKC",
            "remove signature-not-verified boilerplate",
            "remove digital signature boilerplate",
            "remove date stamps tied to digital signature boilerplate",
            "remove email-like artifacts",
            "dehyphenate line-break split words",
            "replace line breaks with spaces",
            "collapse repeated whitespace",
            "strip leading and trailing whitespace",
        ],
        "per_split": per_split,
        "combined": {
            "rows": total_rows,
            "label_counts": total_label_counts,
            "empty_clean_text_rows": total_empty_rows,
            "long_document_rows": total_long_rows,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and preprocess ILDC parquet files.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help="Directory containing downloaded raw ILDC parquet files.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directory where processed parquet files will be written.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Record batch size used while streaming parquet data.",
    )
    parser.add_argument(
        "--char-threshold",
        type=int,
        default=LONG_DOCUMENT_CHAR_THRESHOLD,
        help="Documents longer than this are marked for chunking later.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir: Path = args.raw_dir
    processed_dir: Path = args.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    input_map = split_inputs(raw_dir)
    if not all(input_map.values()):
        missing = [split for split, files in input_map.items() if not files]
        raise FileNotFoundError(
            f"Missing raw ILDC parquet files for splits: {', '.join(missing)}"
        )

    per_split: dict[str, dict[str, object]] = {}
    for split, input_files in input_map.items():
        per_split[split] = write_processed_split(
            split=split,
            input_files=input_files,
            output_dir=processed_dir,
            batch_size=args.batch_size,
            char_threshold=args.char_threshold,
        )

    report = build_report(
        per_split=per_split,
        char_threshold=args.char_threshold,
        batch_size=args.batch_size,
    )
    report_path = processed_dir / "preprocessing_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
