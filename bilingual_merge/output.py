from pathlib import Path
import polars as pl
from loguru import logger

from bilingual_merge.normalize import prepare


def append_and_dedupe_target(
    target_enfr: pl.DataFrame, to_append_enfr: pl.DataFrame
) -> pl.DataFrame:
    """
    Append to target and dedupe by normalized row_key (prevents duplicates caused by whitespace/casing).
    Returns a clean DF with columns: en, fr
    """
    combined = pl.concat(
        [target_enfr.select(["en", "fr"]), to_append_enfr.select(["en", "fr"])],
        how="vertical",
    )
    combined = (
        prepare(combined, "en", "fr").unique(subset=["row_key"]).select(["en", "fr"])
    )
    return combined


def write_jsonl(df: pl.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing JSONL to {out_path}")
    df.write_ndjson(out_path)
