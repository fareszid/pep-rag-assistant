"""Hybrid retrieval: BM25 + embeddings combined with reciprocal rank fusion, optionally
re-ranked by a cross-encoder.

Modes (compared in `pepbot eval`):
- bm25: keywords only, good for exact names like __init_subclass__
- dense: embeddings only, handles paraphrases
- hybrid: RRF of both rankings
- hybrid_rerank: hybrid candidates re-scored by a cross-encoder
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from pepbot.bm25 import BM25
from pepbot.config import (
    CANDIDATES,
    CHUNKS_PATH,
    EMBEDDING_MODEL,
    EMBEDDINGS_PATH,
    RERANK_MODEL,
    RRF_K,
    TOP_K,
)

logger = logging.getLogger(__name__)
MODES = ("bm25", "dense", "hybrid", "hybrid_rerank")


class Embedder(Protocol):
    def embed_passages(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> np.ndarray: ...


class FastEmbedEmbedder:
    """Sentence embeddings with ONNX Runtime (no PyTorch). Vectors are L2-normalised."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model_name)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.vstack(list(self.model.passage_embed(texts, batch_size=64))))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalise(np.vstack(list(self.model.query_embed([text]))))[0]


class FastEmbedReranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = TextCrossEncoder(model_name=model_name)

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        return np.asarray(list(self.model.rerank(query, passages)), dtype=float)


def _normalise(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)


def index_text(chunks: pd.DataFrame, with_heading: bool = True) -> list[str]:
    """Text that gets indexed: "PEP n - title > section" followed by the chunk.

    The header helps chunks that never mention the subject of their PEP.
    """
    if not with_heading:
        return chunks["text"].tolist()
    headings = (
        "PEP " + chunks["pep"].astype(str) + " – " + chunks["title"] + " › " + chunks["section"]
    )
    return (headings + "\n" + chunks["text"]).tolist()


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """score(d) = sum of 1 / (k + rank(d)) over the rankings, ranks start at 1."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


@dataclass(frozen=True)
class Hit:
    row: int
    score: float
    chunk: dict


class Retriever:
    def __init__(
        self,
        chunks: pd.DataFrame,
        embeddings: np.ndarray | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        with_heading: bool = True,
    ):
        self.chunks = chunks.reset_index(drop=True)
        self.texts = index_text(self.chunks, with_heading)
        self.bm25 = BM25().fit(self.texts)
        self.embeddings = embeddings
        self.embedder = embedder
        self.reranker = reranker

    @classmethod
    def load(
        cls,
        chunks_path: Path = CHUNKS_PATH,
        embeddings_path: Path = EMBEDDINGS_PATH,
        rerank: bool = True,
    ) -> Retriever:
        chunks = pd.read_parquet(chunks_path)
        embeddings = np.load(embeddings_path)
        return cls(chunks, embeddings, FastEmbedEmbedder(), FastEmbedReranker() if rerank else None)

    def _bm25_ranking(self, query: str, n: int) -> list[int]:
        return [i for i, _ in self.bm25.search(query, n)]

    def _dense_search(self, query: str, n: int) -> list[tuple[int, float]]:
        """Top n passages by cosine similarity (vectors are normalised)."""
        if self.embeddings is None or self.embedder is None:
            raise RuntimeError("Dense retrieval needs embeddings and an embedder")
        similarity = self.embeddings @ self.embedder.embed_query(query)
        top = np.argpartition(-similarity, min(n, len(similarity) - 1))[:n]
        top = top[np.argsort(-similarity[top], kind="stable")]
        return [(int(i), float(similarity[i])) for i in top]

    def search(
        self, query: str, k: int = TOP_K, mode: str = "hybrid_rerank", candidates: int = CANDIDATES
    ) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "bm25":
            ranked = self.bm25.search(query, k)
        elif mode == "dense":
            ranked = self._dense_search(query, k)
        else:
            fused = reciprocal_rank_fusion(
                [
                    self._bm25_ranking(query, candidates),
                    [i for i, _ in self._dense_search(query, candidates)],
                ]
            )
            ranked = fused[:k] if mode == "hybrid" else self._rerank(query, fused[:candidates], k)
        return [Hit(i, s, self.chunks.iloc[i].to_dict()) for i, s in ranked]

    def _rerank(
        self, query: str, fused: list[tuple[int, float]], k: int
    ) -> list[tuple[int, float]]:
        if self.reranker is None:
            return fused[:k]
        rows = [i for i, _ in fused]
        scores = self.reranker.score(query, [self.texts[i] for i in rows])
        order = np.argsort(-scores, kind="stable")[:k]
        return [(rows[j], float(scores[j])) for j in order]


def build_embeddings(
    chunks: pd.DataFrame, embedder: Embedder | None = None, with_heading: bool = True
) -> np.ndarray:
    embedder = embedder or FastEmbedEmbedder()
    texts = index_text(chunks, with_heading)
    logger.info("Embedding %d passages with %s", len(texts), EMBEDDING_MODEL)
    return embedder.embed_passages(texts).astype(np.float32)
