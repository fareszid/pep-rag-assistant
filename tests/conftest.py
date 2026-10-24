"""Small in-memory corpus and fake models, so the tests run offline."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from pepbot.bm25 import tokenize

SAMPLE_PEP = """PEP: 9999
Title: Walrus Assignment Expressions
Author: Jane Doe <jane@example.com>,
        John Roe <john@example.com>
Status: Final
Type: Standards Track
Created: 01-Jan-2020
Python-Version: 3.8


Abstract
========

This proposal adds the ``:=`` operator, informally called the walrus operator,
so that a value can be assigned inside an expression. See :pep:`572` and the
`docs <https://docs.python.org>`_ for details.

Rationale
=========

Naming the result of an expression is useful in comprehensions and in
``while`` loops that read data in chunks.

Examples
--------

The loop below reads a file in fixed-size chunks until it is exhausted:

.. code-block:: python

    while (chunk := f.read(8192)):
        process(chunk)

Copyright
=========

This document is placed in the public domain.
"""


class HashingEmbedder:
    """Deterministic bag-of-words embedder (hashing trick)."""

    dim = 256

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim)
        for token in tokenize(text):
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed_passages(self, texts):
        return np.vstack([self._vector(t) for t in texts])

    def embed_query(self, text):
        return self._vector(text)


class LengthReranker:
    """Ranks shorter passages higher, easy to check in tests."""

    def score(self, query, passages):
        return -np.array([len(p) for p in passages], dtype=float)


@pytest.fixture
def chunks() -> pd.DataFrame:
    rows = [
        (
            572,
            "Assignment Expressions",
            "Abstract",
            "Assign values inside expressions with the walrus operator :=.",
        ),
        (
            8,
            "Style Guide for Python Code",
            "Indentation",
            "Use 4 spaces per indentation level. Never mix tabs and spaces.",
        ),
        (
            20,
            "The Zen of Python",
            "The Zen",
            "Beautiful is better than ugly. Explicit is better "
            "than implicit. Simple is better than complex.",
        ),
        (
            498,
            "Literal String Interpolation",
            "Abstract",
            "f-strings evaluate expressions inside braces at runtime.",
        ),
        (484, "Type Hints", "Rationale", "Type hints let static checkers verify annotations."),
    ]
    frame = pd.DataFrame(rows, columns=["pep", "title", "section", "text"])
    frame["chunk_id"] = [f"pep{p:04d}-000" for p in frame["pep"]]
    frame["url"] = [f"https://peps.python.org/pep-{p:04d}/" for p in frame["pep"]]
    return frame


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder()
