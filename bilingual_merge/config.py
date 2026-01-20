from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


@dataclass(frozen=True)
class Config:
    source: Path
    target: Path
    out: Path
    en_col: str
    fr_col: str
    fuzzy_threshold: int
    semantic_threshold: float
    embed_backend: Literal["minilm", "gemini"]
    minilm_model: str
    gemini_model: str
    gemini_api_key: Optional[str]
    max_candidates_per_row: int
