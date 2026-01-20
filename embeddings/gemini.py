import os
from typing import List, Optional

import numpy as np
from loguru import logger

from .base import Embedder


def _normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _chunked(texts: List[str], n: int):
    for i in range(0, len(texts), n):
        yield texts[i : i + n]


class GeminiEmbedder(Embedder):
    """
    Gemini embeddings via Google's GenAI SDK.

    Install:
      pip install -e '.[gemini]'
    Set key:
      export GEMINI_API_KEY="..."
    """

    def __init__(self, model: str, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY env var (or pass --gemini-api-key)."
            )

        try:
            from google import genai  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "google-genai is not installed. Install with:\n"
                "  pip install -e '.[gemini]'"
            ) from e

        logger.info(f"Initializing Gemini embedder model={model}")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def embed(self, texts: List[str]) -> np.ndarray:
        out_vecs: List[List[float]] = []
        for batch in _chunked(texts, 64):
            resp = self.client.models.embed_content(model=self.model, contents=batch)
            embeddings = getattr(resp, "embeddings", None)
            if embeddings is None:
                raise RuntimeError("Gemini embed response missing embeddings field.")
            for emb in embeddings:
                vals = getattr(emb, "values", None)
                if vals is None:
                    raise RuntimeError("Gemini embed entry missing values.")
                out_vecs.append(list(vals))

        arr = np.asarray(out_vecs, dtype=np.float32)
        return _normalize(arr)
