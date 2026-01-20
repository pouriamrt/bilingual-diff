from pathlib import Path
from typing import Literal, Optional, Dict

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from bilingual_merge.config import Config
from bilingual_merge.io_utils import read_df
from bilingual_merge.normalize import prepare
from bilingual_merge.diffing import find_exact_differences
from bilingual_merge.fuzzy import fuzzy_mismatch_filter
from bilingual_merge.semantic import semantic_mismatch_filter
from bilingual_merge.output import append_and_dedupe_target, write_jsonl
from bilingual_merge.embeddings import MiniLMEmbedder, GeminiEmbedder, Embedder

console = Console()
app = typer.Typer(add_completion=False)


def render_summary(title: str, counts: Dict[str, int]) -> None:
    table = Table(title=title)
    table.add_column("Stage", style="bold")
    table.add_column("Rows", justify="right")
    for k, v in counts.items():
        table.add_row(k, f"{v:,}")
    console.print(table)


@app.command()
def main(
    source: Path = typer.Option(
        ..., help="Source dataset (to compare from). parquet/csv/jsonl/json"
    ),
    target: Path = typer.Option(
        ..., help="Target dataset (to be appended to). parquet/csv/jsonl/json"
    ),
    out: Path = typer.Option(..., help="Output JSONL path."),
    en_col: str = typer.Option("en", help="English column name."),
    fr_col: str = typer.Option("fr", help="French column name."),
    fuzzy_threshold: int = typer.Option(
        92, help="Keep rows whose best fuzzy score is < this."
    ),
    semantic_threshold: float = typer.Option(
        0.82, help="Keep rows whose best cosine similarity is < this."
    ),
    embed_backend: Literal["minilm", "gemini"] = typer.Option(
        "minilm", help="Embedding backend."
    ),
    minilm_model: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2", help="MiniLM model id."
    ),
    gemini_model: str = typer.Option(
        "gemini-embedding-001", help="Gemini embedding model id."
    ),
    gemini_api_key: Optional[str] = typer.Option(
        None, help="Gemini API key (or set GEMINI_API_KEY)."
    ),
    max_candidates_per_row: int = typer.Option(
        200, help="Speed cap for scanning target rows per candidate."
    ),
):
    logger.remove()
    logger.add(lambda msg: console.print(msg, end=""), level="INFO")

    cfg = Config(
        source=source,
        target=target,
        out=out,
        en_col=en_col,
        fr_col=fr_col,
        fuzzy_threshold=fuzzy_threshold,
        semantic_threshold=semantic_threshold,
        embed_backend=embed_backend,
        minilm_model=minilm_model,
        gemini_model=gemini_model,
        gemini_api_key=gemini_api_key,
        max_candidates_per_row=max_candidates_per_row,
    )

    # Read
    src_raw = read_df(cfg.source)
    tgt_raw = read_df(cfg.target)

    # Prepare
    src = prepare(src_raw, cfg.en_col, cfg.fr_col)
    tgt = prepare(tgt_raw, cfg.en_col, cfg.fr_col)

    render_summary("Input sizes", {"source": src.height, "target": tgt.height})

    # Exact differences
    candidates = find_exact_differences(src, tgt)
    render_summary(
        "After exact diff (src anti-join tgt)", {"candidates": candidates.height}
    )

    if candidates.is_empty():
        console.print(
            "[green]No new/different rows to append. Writing target as JSONL.[/green]"
        )
        write_jsonl(tgt.select(["en", "fr"]), cfg.out)
        console.print(f"[cyan]Output:[/cyan] {cfg.out}")
        raise typer.Exit(code=0)

    # Fuzzy mismatch filter
    fuzzy_kept = fuzzy_mismatch_filter(
        candidates=candidates,
        tgt=tgt,
        threshold=cfg.fuzzy_threshold,
        max_candidates_per_row=min(cfg.max_candidates_per_row, tgt.height),
        console=console,
    )
    render_summary(
        "After fuzzy filter",
        {"candidates_in": candidates.height, "fuzzy_mismatches": fuzzy_kept.height},
    )

    if fuzzy_kept.is_empty():
        console.print(
            "[yellow]All candidates had strong fuzzy matches. Writing target as JSONL.[/yellow]"
        )
        write_jsonl(tgt.select(["en", "fr"]), cfg.out)
        console.print(f"[cyan]Output:[/cyan] {cfg.out}")
        raise typer.Exit(code=0)

    # Embedding backend
    if cfg.embed_backend == "minilm":
        embedder: Embedder = MiniLMEmbedder(cfg.minilm_model)
    else:
        embedder = GeminiEmbedder(model=cfg.gemini_model, api_key=cfg.gemini_api_key)

    # Semantic mismatch filter
    semantic_kept = semantic_mismatch_filter(
        candidates=fuzzy_kept,
        tgt=tgt,
        embedder=embedder,
        threshold=cfg.semantic_threshold,
        max_candidates_per_row=min(cfg.max_candidates_per_row, tgt.height),
        console=console,
    )
    render_summary(
        "After semantic filter",
        {
            "fuzzy_mismatches_in": fuzzy_kept.height,
            "semantic_mismatches": semantic_kept.height,
        },
    )

    # Append + dedupe + write
    tgt_out = tgt.select(["en", "fr"])
    final_df = append_and_dedupe_target(tgt_out, semantic_kept.select(["en", "fr"]))

    render_summary(
        "Final",
        {
            "target_original": tgt_out.height,
            "appended": semantic_kept.height,
            "target_final_unique": final_df.height,
        },
    )

    write_jsonl(final_df, cfg.out)
    console.print(f"[green]Done.[/green] Output: {cfg.out}")
