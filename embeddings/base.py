from typing import List
import numpy as np


class Embedder:
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an (N, D) float array."""
        raise NotImplementedError
