import numpy as np
import pytest

from pepbot.bm25 import BM25, tokenize
from pepbot.retrieve import Retriever, index_text, reciprocal_rank_fusion

from .conftest import LengthReranker


def test_tokenize_drops_stopwords_and_keeps_identifiers():
    assert tokenize("What is the __init__ method of a Class?") == ["__init__", "method", "class"]


def test_bm25_ranks_matching_document_first():
    docs = ["the walrus operator assigns", "tabs and spaces", "walrus walrus facts", "spaces"]
    bm25 = BM25().fit(docs)
    ranked = [i for i, _ in bm25.search("walrus operator", k=4)]
    assert ranked[0] == 0  # matches both terms
    assert set(ranked) == {0, 2}  # documents without query terms are not returned
    assert bm25.search("unknown words", k=3) == []


def test_bm25_rare_terms_weigh_more():
    bm25 = BM25().fit(["common rare", "common", "common", "common"])
    assert bm25.idf[bm25.vocabulary["rare"]] > bm25.idf[bm25.vocabulary["common"]]


def test_reciprocal_rank_fusion():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1]], k=60)
    scores = dict(fused)
    assert fused[0][0] == 1  # ranked 1st and 2nd beats 3rd and 1st
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[2] == pytest.approx(1 / 62)


def test_index_text_adds_contextual_header(chunks):
    assert index_text(chunks)[0].startswith("PEP 572 – Assignment Expressions › Abstract\n")
    assert index_text(chunks, with_heading=False)[0] == chunks["text"][0]


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "hybrid_rerank"])
def test_every_mode_finds_the_walrus_pep(chunks, embedder, mode):
    embeddings = embedder.embed_passages(index_text(chunks))
    retriever = Retriever(chunks, embeddings, embedder, LengthReranker())
    hits = retriever.search("walrus operator assignment expressions", k=3, mode=mode)
    assert hits and len(hits) <= 3
    if mode != "hybrid_rerank":
        assert hits[0].chunk["pep"] == 572


def test_reranker_reorders_candidates(chunks, embedder):
    embeddings = embedder.embed_passages(index_text(chunks))
    retriever = Retriever(chunks, embeddings, embedder, LengthReranker())
    hits = retriever.search("python", k=5, mode="hybrid_rerank", candidates=5)
    lengths = [len(retriever.texts[h.row]) for h in hits]
    assert lengths == sorted(lengths)


def test_dense_mode_requires_embeddings(chunks):
    with pytest.raises(RuntimeError):
        Retriever(chunks).search("walrus", mode="dense")
    with pytest.raises(ValueError):
        Retriever(chunks).search("walrus", mode="magic")


def test_dense_scores_are_cosine(chunks, embedder):
    embeddings = embedder.embed_passages(index_text(chunks))
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1)
