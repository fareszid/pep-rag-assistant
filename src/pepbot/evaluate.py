"""Evaluation.

Retrieval (offline): each question in eval/questions.jsonl lists the PEPs that answer it, and
a retrieved passage counts as relevant if it comes from one of them. Metrics: Hit@k and MRR@10.

Generation (needs an API key): asks Claude each question and checks that the answer cites
one of the right PEPs.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from pepbot.retrieve import Hit

K_VALUES = (1, 3, 5, 10)


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def first_relevant_rank(hits: list[Hit], gold: set[int]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if int(hit.chunk["pep"]) in gold:
            return rank
    return None


def score_ranks(ranks: list[int | None]) -> dict[str, float]:
    metrics = {
        f"hit@{k}": float(np.mean([r is not None and r <= k for r in ranks])) for k in K_VALUES
    }
    metrics["mrr@10"] = float(np.mean([1 / r if r is not None and r <= 10 else 0 for r in ranks]))
    return metrics


def evaluate_retrieval(
    search: Callable[[str], list[Hit]], questions: list[dict], name: str
) -> tuple[dict, pd.DataFrame]:
    rows = []
    for q in questions:
        started = time.perf_counter()
        hits = search(q["question"])
        latency = time.perf_counter() - started
        rows.append(
            {
                "config": name,
                "question": q["question"],
                "category": q.get("category", ""),
                "gold": q["gold"],
                "rank": first_relevant_rank(hits, set(q["gold"])),
                "top_pep": int(hits[0].chunk["pep"]) if hits else None,
                "latency_ms": 1000 * latency,
            }
        )
    detail = pd.DataFrame(rows)
    summary = {
        "config": name,
        **score_ranks(detail["rank"].tolist()),
        "latency_ms_p50": float(detail["latency_ms"].median()),
    }
    return summary, detail


def evaluate_generation(
    answer_fn: Callable[[str], object], questions: list[dict]
) -> tuple[dict, pd.DataFrame]:
    rows = []
    for q in questions:
        answer = answer_fn(q["question"])
        cited = {c.pep for c in answer.citations}
        rows.append(
            {
                "question": q["question"],
                "gold": q["gold"],
                "cited_peps": sorted(cited),
                "cites_gold": bool(cited & set(q["gold"])),
                "has_citation": bool(cited),
                "input_tokens": answer.input_tokens,
                "output_tokens": answer.output_tokens,
            }
        )
    detail = pd.DataFrame(rows)
    summary = {
        "answers": len(detail),
        "answers_with_citations": float(detail["has_citation"].mean()),
        "answers_citing_gold_pep": float(detail["cites_gold"].mean()),
        "mean_input_tokens": float(detail["input_tokens"].mean()),
        "mean_output_tokens": float(detail["output_tokens"].mean()),
    }
    return summary, detail


def plot_retrieval(summary: pd.DataFrame, n_questions: int, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ["hit@1", "hit@5", "mrr@10"]
    colors = ["#94a3b8", "#2563eb", "#16a34a"]
    data = summary.set_index("config")[metrics]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.26
    x = np.arange(len(data))
    for i, (metric, color) in enumerate(zip(metrics, colors, strict=True)):
        bars = ax.bar(x + (i - 1) * width, data[metric], width, label=metric, color=color)
        ax.bar_label(bars, labels=[f"{v:.2f}" for v in data[metric]], fontsize=7.5, padding=2)
    ax.set_xticks(x, data.index, rotation=12)
    ax.set_ylim(0, 1.08)
    ax.set_title(f"Retrieval quality on {n_questions} PEP questions (passage from a gold PEP)")
    ax.set_ylabel("Score")
    ax.legend(loc="upper left", ncols=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
