from types import SimpleNamespace as NS

import pytest

from pepbot.evaluate import (
    evaluate_generation,
    evaluate_retrieval,
    first_relevant_rank,
    score_ranks,
)
from pepbot.retrieve import Hit


def hits(*peps):
    return [Hit(i, 1.0, {"pep": p}) for i, p in enumerate(peps)]


def test_first_relevant_rank():
    assert first_relevant_rank(hits(8, 572, 572), {572}) == 2
    assert first_relevant_rank(hits(8, 20), {572}) is None


def test_score_ranks():
    metrics = score_ranks([1, 3, None, 12])
    assert metrics["hit@1"] == 0.25
    assert metrics["hit@3"] == 0.5
    assert metrics["hit@10"] == 0.5
    assert metrics["mrr@10"] == pytest.approx((1 + 1 / 3) / 4)


def test_evaluate_retrieval():
    questions = [{"question": "a", "gold": [1]}, {"question": "b", "gold": [2]}]
    results = {"a": hits(1, 3), "b": hits(3, 4)}
    summary, detail = evaluate_retrieval(results.get, questions, "fake")
    assert summary["hit@1"] == 0.5 and summary["config"] == "fake"
    assert detail["rank"].tolist()[0] == 1


def test_evaluate_generation():
    questions = [{"question": "a", "gold": [1]}, {"question": "b", "gold": [2]}]

    def answer(q):
        peps = {"a": [1], "b": [5]}[q]
        return NS(citations=[NS(pep=p) for p in peps], input_tokens=10, output_tokens=5)

    summary, _ = evaluate_generation(answer, questions)
    assert summary["answers_citing_gold_pep"] == 0.5
    assert summary["answers_with_citations"] == 1.0
