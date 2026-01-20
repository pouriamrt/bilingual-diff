from typing import List
import numpy as np
from loguru import logger

from .base import Embedder

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


class MiniLMEmbedder(Embedder):
    def __init__(self, model_name: str):
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed. Install with:\n"
                "  pip install -e '.[minilm]'"
            )
        logger.info(f"Loading MiniLM model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        )
