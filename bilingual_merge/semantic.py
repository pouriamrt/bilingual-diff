from typing import List

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
) -> pl.DataFrame:
    """
    Embed candidate EN and target EN. For each candidate compute best cosine similarity
    against target EN. Keep candidates whose best_sim < threshold.

    Assumes embedder outputs normalized vectors (or we treat dot product as cosine).
    """
    tgt_en_all = tgt["en"].to_list()
    if not tgt_en_all:
        return candidates

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
            best_sims.append(float(np.max(sims)) if sims.size else 0.0)
            progress.advance(task)

    out = candidates.with_columns(pl.Series("semantic_best_en", best_sims))
    return out.filter(pl.col("semantic_best_en") < threshold)
