import polars as pl


def find_exact_differences(src: pl.DataFrame, tgt: pl.DataFrame) -> pl.DataFrame:
    """Rows in src but not in tgt by exact normalized row_key, with stripping applied on string columns."""
    # Strip string columns in both src and tgt
    strip_cols = [
        col for col in src.columns if pl.datatypes.is_string_dtype(src.schema[col])
    ]
    if strip_cols:
        src = src.with_columns(
            [pl.col(c).str.strip_chars().alias(c) for c in strip_cols]
        )
        tgt = tgt.with_columns(
            [
                pl.col(c).str.strip_chars().alias(c)
                for c in strip_cols
                if c in tgt.columns
            ]
        )
    return src.join(tgt.select(["row_key"]), on="row_key", how="anti")
