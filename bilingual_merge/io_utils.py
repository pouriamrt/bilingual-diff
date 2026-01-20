from pathlib import Path

import polars as pl
from loguru import logger


def detect_format(path: Path) -> str:
    suf = path.suffix.lower()
    format_map = {
        ".parquet": "parquet",
        ".csv": "csv",
        ".json": "json",
        ".jsonl": "jsonl",
        ".ndjson": "jsonl",
    }
    if suf in format_map:
        return format_map[suf]
    raise ValueError(f"Unsupported file type: {suf} (use parquet/csv/jsonl/json)")


def read_df(path: Path) -> pl.DataFrame:
    fmt = detect_format(path)
    logger.info(f"Reading {path} as {fmt}")
    if fmt == "parquet":
        return pl.read_parquet(path)
    if fmt == "csv":
        return pl.read_csv(path, infer_schema_length=10_000)
    if fmt == "jsonl":
        return pl.read_ndjson(path)
    if fmt == "json":
        return pl.read_json(path)
    raise RuntimeError("unreachable")
