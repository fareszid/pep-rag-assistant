"""Command line entry point.

pepbot ingest      download the PEP sources and split them into chunks
pepbot index       embed every chunk (ONNX model, CPU)
pepbot ask "What is the walrus operator?" [--mode hybrid_rerank] [--no-llm]
pepbot eval        retrieval benchmark (--generation also checks Claude's citations)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd

from pepbot.config import (
    CHUNKS_PATH,
    EMBEDDINGS_PATH,
    EVAL_QUESTIONS,
    INDEX_DIR,
    REPORTS_DIR,
    TOP_K,
)

logger = logging.getLogger("pepbot")
PLAIN_EMBEDDINGS = INDEX_DIR / "embeddings_plain.npy"


def cmd_ingest(_: argparse.Namespace) -> None:
    from pepbot.ingest import build_chunks, download_peps

    download_peps()
    chunks = build_chunks()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    chunks.to_parquet(CHUNKS_PATH, index=False)
    print(f"{len(chunks):,} chunks from {chunks['pep'].nunique()} PEPs -> {CHUNKS_PATH}")


def cmd_index(args: argparse.Namespace) -> None:
    from pepbot.retrieve import FastEmbedEmbedder, build_embeddings

    chunks = pd.read_parquet(CHUNKS_PATH)
    embedder = FastEmbedEmbedder()
    np.save(EMBEDDINGS_PATH, build_embeddings(chunks, embedder, with_heading=True))
    if args.ablation:  # version without headers, only used by the eval
        np.save(PLAIN_EMBEDDINGS, build_embeddings(chunks, embedder, with_heading=False))
    print(f"Embedded {len(chunks):,} chunks -> {EMBEDDINGS_PATH}")


def cmd_ask(args: argparse.Namespace) -> None:
    import anthropic

    from pepbot.generate import format_answer, generate_answer
    from pepbot.retrieve import Retriever

    retriever = Retriever.load(rerank=args.mode == "hybrid_rerank")
    hits = retriever.search(args.question, k=args.k, mode=args.mode)
    if args.no_llm:
        for i, hit in enumerate(hits, start=1):
            c = hit.chunk
            print(f"\n[{i}] PEP {c['pep']} – {c['title']} › {c['section']}  ({c['url']})")
            print("    " + c["text"][:400].replace("\n", "\n    "))
        return
    try:
        answer = generate_answer(args.question, hits)
    except anthropic.AuthenticationError:
        sys.exit("Authentication failed: set ANTHROPIC_API_KEY, or use --no-llm.")
    except anthropic.RateLimitError:
        sys.exit("Rate limited by the Claude API, try again shortly.")
    except anthropic.APIStatusError as exc:
        sys.exit(f"Claude API error {exc.status_code}: {exc.message}")
    except anthropic.APIConnectionError:
        sys.exit("Could not reach the Claude API (network error).")
    except TypeError as exc:  # the SDK raises TypeError when it finds no credentials
        sys.exit(f"{exc}\nSet ANTHROPIC_API_KEY, or use --no-llm for retrieval only.")
    print(format_answer(answer))


def cmd_eval(args: argparse.Namespace) -> None:
    from pepbot.evaluate import (
        evaluate_generation,
        evaluate_retrieval,
        load_questions,
        plot_retrieval,
    )
    from pepbot.retrieve import FastEmbedEmbedder, FastEmbedReranker, Retriever

    questions = load_questions(EVAL_QUESTIONS)
    chunks = pd.read_parquet(CHUNKS_PATH)
    embedder, reranker = FastEmbedEmbedder(), FastEmbedReranker()
    full = Retriever(chunks, np.load(EMBEDDINGS_PATH), embedder, reranker)
    configs = [("BM25", full, "bm25"), ("Dense", full, "dense")]
    if PLAIN_EMBEDDINGS.exists():
        plain = Retriever(chunks, np.load(PLAIN_EMBEDDINGS), embedder, reranker, with_heading=False)
        configs = [
            ("BM25, no headers", plain, "bm25"),
            ("Dense, no headers", plain, "dense"),
            *configs,
        ]
    configs += [("Hybrid (RRF)", full, "hybrid"), ("Hybrid + rerank", full, "hybrid_rerank")]

    summaries, details = [], []
    for name, retriever, mode in configs:
        summary, detail = evaluate_retrieval(
            lambda q, r=retriever, m=mode: r.search(q, k=10, mode=m), questions, name
        )
        logger.info(
            "%-20s hit@1 %.3f hit@5 %.3f mrr %.3f",
            name,
            summary["hit@1"],
            summary["hit@5"],
            summary["mrr@10"],
        )
        summaries.append(summary)
        details.append(detail)

    summary = pd.DataFrame(summaries)
    detail = pd.concat(details)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "figures").mkdir(exist_ok=True)
    by_category = (
        detail[detail["config"] == "Hybrid + rerank"]
        .assign(hit5=lambda d: d["rank"].le(5))
        .groupby("category")["hit5"]
        .agg(["mean", "size"])
    )
    report = {
        "questions": len(questions),
        "chunks": len(chunks),
        "peps": int(chunks["pep"].nunique()),
        "retrieval": summaries,
        "hit@5_by_category_hybrid_rerank": by_category.round(3).to_dict(orient="index"),
    }
    detail.to_csv(REPORTS_DIR / "retrieval_eval_details.csv", index=False)
    plot_retrieval(summary, len(questions), REPORTS_DIR / "figures" / "retrieval_eval.png")

    if args.generation:
        from pepbot.generate import generate_answer

        gen_summary, gen_detail = evaluate_generation(
            lambda q: generate_answer(q, full.search(q, k=TOP_K)), questions
        )
        report["generation"] = gen_summary
        gen_detail.to_csv(REPORTS_DIR / "generation_eval_details.csv", index=False)

    (REPORTS_DIR / "eval.json").write_text(json.dumps(report, indent=2))
    print(summary.round(3).to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="pepbot", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest").set_defaults(func=cmd_ingest)
    index = sub.add_parser("index")
    index.add_argument(
        "--ablation", action="store_true", help="also embed chunks without contextual headers"
    )
    index.set_defaults(func=cmd_index)
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    ask.add_argument(
        "--mode", default="hybrid_rerank", choices=["bm25", "dense", "hybrid", "hybrid_rerank"]
    )
    ask.add_argument("-k", type=int, default=TOP_K)
    ask.add_argument("--no-llm", action="store_true", help="show retrieved passages only")
    ask.set_defaults(func=cmd_ask)
    ev = sub.add_parser("eval")
    ev.add_argument(
        "--generation",
        action="store_true",
        help="also generate answers with Claude (requires an API key)",
    )
    ev.set_defaults(func=cmd_eval)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "huggingface_hub"):  # huggingface downloads log every HTTP request
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    main()
