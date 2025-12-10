# embeddings.py

import os
from functools import lru_cache
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# Decide once which device to use
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def get_bge_model() -> SentenceTransformer:
    """
    Load and cache the BGE embedding model on GPU if available.

    Uses env var BGE_MODEL_NAME if set, otherwise defaults to 'BAAI/bge-base-en'.
    """
    model_name = os.getenv("BGE_MODEL_NAME", "BAAI/bge-base-en")
    print(f"[embeddings] Loading model {model_name} on device: {DEVICE}")
    model = SentenceTransformer(model_name, device=DEVICE)
    return model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of texts into a 2D numpy array of shape (n_texts, dim).
    Embeddings are L2-normalized so dot-product == cosine similarity.
    """
    if not texts:
        return np.zeros((0, 0), dtype="float32")

    model = get_bge_model()

    # You can tune batch_size if GPU memory is tight or generous
    embeddings = model.encode(
        texts,
        batch_size=64,            # increase if you have more VRAM, decrease if OOM
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.astype("float32")
