import hashlib
import polars as pl


def normalize_text_expr(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .cast(pl.Utf8)
        .fill_null("")
        .str.strip_chars()
        .str.to_lowercase()
        .str.replace_all(r"\s+", " ")
    )


def stable_row_key(en: str, fr: str) -> str:
    h = hashlib.sha256()
    h.update(en.encode("utf-8"))
    h.update(b"\x1f")
    h.update(fr.encode("utf-8"))
    return h.hexdigest()


def prepare(df: pl.DataFrame, en_col: str, fr_col: str) -> pl.DataFrame:
    df2 = (
        df.select([pl.col(en_col).alias("en"), pl.col(fr_col).alias("fr")])
        .with_columns(
            [
                normalize_text_expr("en").alias("en_norm"),
                normalize_text_expr("fr").alias("fr_norm"),
            ]
        )
        .with_columns(
            [
                pl.struct(["en_norm", "fr_norm"])
                .map_elements(
                    lambda s: stable_row_key(s["en_norm"], s["fr_norm"]),
                    return_dtype=pl.Utf8,
                )
                .alias("row_key")
            ]
        )
    )
    return df2
