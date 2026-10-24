"""Okapi BM25 on scipy sparse matrices.

Term weights are computed once in fit(), so scoring a query is just summing a few columns.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
from scipy import sparse

TOKEN = re.compile(r"[a-z0-9_]+")
_STOPWORDS = """
a an and are as at be but by can do does for from has have how i if in into is it its of on or
should so such that the their then there these this to was what when where which who why will
with would you your
"""
STOPWORDS = frozenset(_STOPWORDS.split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def fit(self, documents: list[str]) -> BM25:
        vocabulary: dict[str, int] = {}
        rows, cols, counts = [], [], []
        lengths = np.zeros(len(documents))
        for i, doc in enumerate(documents):
            tokens = tokenize(doc)
            lengths[i] = len(tokens)
            for term, count in Counter(tokens).items():
                rows.append(i)
                cols.append(vocabulary.setdefault(term, len(vocabulary)))
                counts.append(count)
        tf = sparse.csr_matrix(
            (np.asarray(counts, dtype=float), (rows, cols)),
            shape=(len(documents), len(vocabulary)),
        )
        n_docs = len(documents)
        df = np.bincount(tf.indices, minlength=len(vocabulary))
        self.idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5))

        # weight(d, t) = idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
        norm = self.k1 * (1 - self.b + self.b * lengths / max(lengths.mean(), 1e-9))
        tf = tf.tocoo()
        weights = tf.data * (self.k1 + 1) / (tf.data + norm[tf.row]) * self.idf[tf.col]
        self.matrix = sparse.csc_matrix((weights, (tf.row, tf.col)), shape=tf.shape)
        self.vocabulary = vocabulary
        return self

    def scores(self, query: str) -> np.ndarray:
        ids = [self.vocabulary[t] for t in tokenize(query) if t in self.vocabulary]
        if not ids:
            return np.zeros(self.matrix.shape[0])
        return np.asarray(self.matrix[:, ids].sum(axis=1)).ravel()

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.scores(query)
        top = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]
