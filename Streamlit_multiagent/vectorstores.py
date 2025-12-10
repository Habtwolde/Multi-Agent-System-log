# vectorstores.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
import pickle

from embeddings import embed_texts


@dataclass
class VectorDocument:
    text: str
    metadata: Dict[str, Any]


@dataclass
class SimpleVectorStore:
    """
    Minimal vector store:
    - Stores texts + metadata
    - Pre-computes embeddings
    - Supports similarity_search(query, k)
    - Now also supports pickle-based save/load for fast startup
    """

    docs: List[VectorDocument] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None  # shape (n_docs, dim) or None

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> "SimpleVectorStore":
        assert len(texts) == len(metadatas)
        docs = [VectorDocument(t, m) for t, m in zip(texts, metadatas)]
        if texts:
            embs = embed_texts(texts)
        else:
            embs = None
        return cls(docs=docs, embeddings=embs)

    def add_document(self, text: str, metadata: Dict[str, Any]) -> None:
        """Append one new document and update embeddings."""
        self.docs.append(VectorDocument(text=text, metadata=metadata))
        new_emb = embed_texts([text])
        if self.embeddings is None or self.embeddings.size == 0:
            self.embeddings = new_emb
        else:
            self.embeddings = np.vstack([self.embeddings, new_emb])

    def _similarities(self, query_emb: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarities via dot product:
        rows and query must both be L2-normalized (enforced in embed_texts).
        """
        if self.embeddings is None or self.embeddings.size == 0:
            return np.zeros((0,), dtype="float32")
        sims = self.embeddings @ query_emb  # (n_docs, dim) @ (dim,) -> (n_docs,)
        return sims

    def similarity_search(
        self,
        query: str,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Returns list of {text, metadata, score}, sorted by score desc.
        """
        if not self.docs:
            return []

        q_emb = embed_texts([query])[0]
        sims = self._similarities(q_emb)
        k = min(k, len(self.docs))
        top_idx = np.argsort(-sims)[:k]

        results: List[Dict[str, Any]] = []
        for idx in top_idx:
            doc = self.docs[int(idx)]
            results.append(
                {
                    "text": doc.text,
                    "metadata": doc.metadata,
                    "score": float(sims[idx]),
                }
            )
        return results

    # ---------- Serialization helpers ----------

    def save(self, path: str) -> None:
        """Save the entire store to disk using pickle."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "SimpleVectorStore":
        """Load a store from disk."""
        with open(path, "rb") as f:
            store = pickle.load(f)
        return store
