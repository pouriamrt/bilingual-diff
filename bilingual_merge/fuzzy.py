from typing import List, Tuple

import polars as pl
from rapidfuzz import fuzz
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)


def fuzzy_mismatch_filter(
    candidates: pl.DataFrame,
    tgt: pl.DataFrame,
    threshold: int,
    *,
    max_candidates_per_row: int,
    console: Console,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    For each candidate row, compute best fuzzy match score against target EN strings.
    Keep rows whose best score < threshold.
    Returns (kept, similar) where similar are rows filtered out (score >= threshold).

    max_candidates_per_row is a speed cap on number of target rows scanned per candidate.
    """
    tgt_en_all = tgt["en"].to_list()
    if not tgt_en_all:
        return candidates, pl.DataFrame()

    tgt_en = (
        tgt_en_all[:max_candidates_per_row]
        if max_candidates_per_row < len(tgt_en_all)
        else tgt_en_all
    )
    cand_en = candidates["en"].to_list()

    scores: List[int] = []
    best_match_indices: List[int] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Fuzzy matching candidates vs target (EN)", total=len(cand_en)
        )
        for s in cand_en:
            best = 0
            best_idx = -1
            for idx, t in enumerate(tgt_en):
                sc = fuzz.token_set_ratio(s, t)
                if sc > best:
                    best = sc
                    best_idx = idx
                    if best == 100:
                        break
            scores.append(best)
            best_match_indices.append(best_idx)
            progress.advance(task)

    out = candidates.with_columns(
        [
            pl.Series("fuzzy_best_en", scores),
            pl.Series("fuzzy_best_match_idx", best_match_indices),
        ]
    )
    kept = out.filter(pl.col("fuzzy_best_en") < threshold)
    similar = out.filter(pl.col("fuzzy_best_en") >= threshold)
    return kept, similar
