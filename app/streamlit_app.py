"""Streamlit chat app. Build the index first (`pepbot ingest`, `pepbot index`), then run
`streamlit run app/streamlit_app.py`.

Without an Anthropic API key it still shows the retrieved passages.
"""

from __future__ import annotations

import os

import anthropic
import streamlit as st

from pepbot.config import CHUNKS_PATH, EMBEDDINGS_PATH
from pepbot.generate import generate_answer
from pepbot.retrieve import MODES, Retriever

st.set_page_config(page_title="Ask the PEPs", page_icon="🐍", layout="wide")


@st.cache_resource(show_spinner="Loading the index and models…")
def get_retriever() -> Retriever:
    return Retriever.load(rerank=True)


if not CHUNKS_PATH.exists() or not EMBEDDINGS_PATH.exists():
    st.error("No search index yet. Build it first with `make ingest` and `make index`.")
    st.stop()

retriever = get_retriever()
has_key = bool(os.getenv("ANTHROPIC_API_KEY"))

with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Retrieval", MODES, index=MODES.index("hybrid_rerank"))
    k = st.slider("Passages sent to Claude", 2, 10, 6)
    use_llm = st.toggle("Generate an answer with Claude", value=has_key, disabled=not has_key)
    if not has_key:
        st.info("Set `ANTHROPIC_API_KEY` to get written answers. Retrieval works without it.")
    st.caption(
        f"{len(retriever.chunks):,} passages from {retriever.chunks['pep'].nunique()} PEPs · "
        "BM25 + bge-small embeddings, fused with RRF, re-ranked by a MiniLM cross-encoder."
    )

st.title("🐍 Ask the PEPs")
st.caption(
    "Questions about Python's design, answered from the Python Enhancement Proposals "
    "with citations."
)

examples = [
    "Why does Python have the walrus operator?",
    "How do I type a decorator that keeps the wrapped function's signature?",
    "What is the plan for removing the GIL?",
]
cols = st.columns(len(examples))
clicked = next((q for col, q in zip(cols, examples, strict=True) if col.button(q)), None)
question = st.chat_input("Ask about any PEP…") or clicked

if question:
    st.chat_message("user").write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching…"):
            hits = retriever.search(question, k=k, mode=mode)
        if use_llm:
            try:
                with st.spinner("Writing an answer…"):
                    answer = generate_answer(question, hits)
                st.markdown(answer.text)
                for c in answer.citations:
                    with st.expander(f"[{c.number}] PEP {c.pep} – {c.title} › {c.section}"):
                        st.markdown(f"> {c.cited_text}")
                        st.markdown(f"[Read on peps.python.org]({c.url})")
                st.caption(
                    f"{answer.model} · {answer.input_tokens:,} input / "
                    f"{answer.output_tokens:,} output tokens"
                )
            except anthropic.RateLimitError:
                st.error("Rate limited by the Claude API. Please retry in a moment.")
            except anthropic.APIStatusError as exc:
                st.error(f"Claude API error {exc.status_code}: {exc.message}")
            except anthropic.APIConnectionError:
                st.error("Could not reach the Claude API.")
        st.subheader("Retrieved passages", divider="gray")
        for i, hit in enumerate(hits, start=1):
            c = hit.chunk
            with st.expander(
                f"{i}. PEP {c['pep']} – {c['title']} › {c['section']}",
                expanded=not use_llm and i == 1,
            ):
                st.markdown(c["text"])
                st.markdown(f"[Open section]({c['url']}) · score {hit.score:.3f}")
