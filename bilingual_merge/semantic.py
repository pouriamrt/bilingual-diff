from typing import List, Tuple

import numpy as np
import polars as pl
from loguru import logger
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)

from bilingual_merge.embeddings.base import Embedder


def semantic_mismatch_filter(
    candidates: pl.DataFrame,
    tgt: pl.DataFrame,
    embedder: Embedder,
    threshold: float,
    *,
    max_candidates_per_row: int,
    console: Console,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Embed candidate EN and target EN. For each candidate compute best cosine similarity
    against target EN. Keep candidates whose best_sim < threshold.
    Returns (kept, similar) where similar are rows filtered out (similarity >= threshold).

    Assumes embedder outputs normalized vectors (or we treat dot product as cosine).
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

    logger.info(f"Embedding target EN: {len(tgt_en)} rows")
    with console.status("Embedding target EN..."):
        tgt_emb = embedder.embed(tgt_en)  # (M, D)

    logger.info(f"Embedding candidate EN: {len(cand_en)} rows")
    with console.status("Embedding candidate EN..."):
        cand_emb = embedder.embed(cand_en)  # (N, D)

    best_sims: List[float] = []
    best_match_indices: List[int] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            "Semantic matching candidates vs target (EN)", total=len(cand_en)
        )
        for i in range(cand_emb.shape[0]):
            sims = tgt_emb @ cand_emb[i]  # (M,)
            if sims.size:
                best_idx = int(np.argmax(sims))
                best_sims.append(float(sims[best_idx]))
                best_match_indices.append(best_idx)
            else:
                best_sims.append(0.0)
                best_match_indices.append(-1)
            progress.advance(task)

    out = candidates.with_columns(
        [
            pl.Series("semantic_best_en", best_sims),
            pl.Series("semantic_best_match_idx", best_match_indices),
        ]
    )
    kept = out.filter(pl.col("semantic_best_en") < threshold)
    similar = out.filter(pl.col("semantic_best_en") >= threshold)
    return kept, similar
