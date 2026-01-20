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


def write_csv(df: pl.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing CSV to {out_path}")
    df.write_csv(out_path)


def write_similar_items(
    fuzzy_similar: pl.DataFrame,
    semantic_similar: pl.DataFrame,
    tgt: pl.DataFrame,
    base_out_path: Path,
) -> None:
    """
    Write similar items (filtered out by fuzzy or semantic matching) to separate files.
    Shows pairs: source row and the matched target row.
    """
    if not fuzzy_similar.is_empty():
        fuzzy_path = base_out_path.with_suffix(".fuzzy_similar.csv")
        # Create paired output: source row and matched target row
        fuzzy_pairs = []
        for row in fuzzy_similar.iter_rows(named=True):
            match_idx = row["fuzzy_best_match_idx"]
            if match_idx >= 0 and match_idx < tgt.height:
                tgt_row = tgt.row(match_idx, named=True)
                fuzzy_pairs.append(
                    {
                        "source_en": row["en"],
                        "source_fr": row["fr"],
                        "target_en": tgt_row["en"],
                        "target_fr": tgt_row["fr"],
                        "fuzzy_score": row["fuzzy_best_en"],
                    }
                )
            else:
                # Fallback if index is invalid
                fuzzy_pairs.append(
                    {
                        "source_en": row["en"],
                        "source_fr": row["fr"],
                        "target_en": "",
                        "target_fr": "",
                        "fuzzy_score": row["fuzzy_best_en"],
                    }
                )

        fuzzy_df = pl.DataFrame(fuzzy_pairs)
        write_csv(fuzzy_df, fuzzy_path)
        logger.info(f"Fuzzy similar items: {fuzzy_similar.height} rows")

    if not semantic_similar.is_empty():
        semantic_path = base_out_path.with_suffix(".semantic_similar.csv")
        # Create paired output: source row and matched target row
        semantic_pairs = []
        for row in semantic_similar.iter_rows(named=True):
            match_idx = row["semantic_best_match_idx"]
            if match_idx >= 0 and match_idx < tgt.height:
                tgt_row = tgt.row(match_idx, named=True)
                semantic_pairs.append(
                    {
                        "source_en": row["en"],
                        "source_fr": row["fr"],
                        "target_en": tgt_row["en"],
                        "target_fr": tgt_row["fr"],
                        "semantic_score": row["semantic_best_en"],
                    }
                )
            else:
                # Fallback if index is invalid
                semantic_pairs.append(
                    {
                        "source_en": row["en"],
                        "source_fr": row["fr"],
                        "target_en": "",
                        "target_fr": "",
                        "semantic_score": row["semantic_best_en"],
                    }
                )

        semantic_df = pl.DataFrame(semantic_pairs)
        write_csv(semantic_df, semantic_path)
        logger.info(f"Semantic similar items: {semantic_similar.height} rows")
