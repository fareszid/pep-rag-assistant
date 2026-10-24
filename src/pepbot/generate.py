"""Answer generation with Claude.

The retrieved passages are sent as search_result blocks with citations enabled, so the API
returns structured citations (URL, title, cited text) for the sentences that use them.
No need to prompt for [1] markers and parse them back.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import anthropic

from pepbot.config import CLAUDE_MODEL
from pepbot.retrieve import Hit

EFFORT = os.getenv("PEPBOT_EFFORT", "medium")

SYSTEM_PROMPT = """You are a precise assistant for questions about Python Enhancement \
Proposals (PEPs). Answer using only the search results provided with the question.

- Lead with the direct answer, then add the details a Python developer needs. Name the PEP \
numbers you rely on and use Markdown code formatting for code.
- If the search results do not contain the answer, say so plainly instead of guessing.
- Keep answers focused: usually two to four short paragraphs or a short list."""


@dataclass(frozen=True)
class Citation:
    number: int
    pep: int
    title: str
    section: str
    url: str
    cited_text: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    stop_reason: str | None = None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


def search_result_block(hit: Hit) -> dict:
    """Turn a hit into a search_result block, one text block per paragraph."""
    chunk = hit.chunk
    paragraphs = [p for p in chunk["text"].split("\n\n") if p.strip()]
    return {
        "type": "search_result",
        "source": chunk["url"],
        "title": f"PEP {chunk['pep']}: {chunk['title']} — {chunk['section']}",
        "content": [{"type": "text", "text": p} for p in paragraphs],
        "citations": {"enabled": True},
    }


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    content = [search_result_block(hit) for hit in hits]
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def parse_response(response, hits: list[Hit]) -> Answer:
    """Join the text blocks and add a [n] marker after each cited part."""
    if response.stop_reason == "refusal":
        return Answer(
            text="The model declined to answer this question.",
            hits=hits,
            stop_reason="refusal",
            model=response.model,
        )
    numbers: dict[int, int] = {}
    citations: list[Citation] = []
    parts: list[str] = []
    for block in response.content:
        if block.type != "text":
            continue  # skip thinking and fallback blocks
        parts.append(block.text)
        markers = []
        for citation in getattr(block, "citations", None) or []:
            if citation.type != "search_result_location":
                continue
            index = citation.search_result_index
            if index not in numbers:
                numbers[index] = len(numbers) + 1
                chunk = hits[index].chunk
                citations.append(
                    Citation(
                        numbers[index],
                        int(chunk["pep"]),
                        chunk["title"],
                        chunk["section"],
                        chunk["url"],
                        citation.cited_text,
                    )
                )
            marker = f"[{numbers[index]}]"
            if marker not in markers:
                markers.append(marker)
        if markers:
            parts.append("".join(markers))
    usage = getattr(response, "usage", None)
    return Answer(
        text="".join(parts).strip(),
        citations=citations,
        hits=hits,
        stop_reason=response.stop_reason,
        model=response.model,
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
    )


def generate_answer(
    question: str,
    hits: list[Hit],
    client: anthropic.Anthropic | None = None,
    model: str = CLAUDE_MODEL,
) -> Answer:
    client = client or anthropic.Anthropic()
    response = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=build_messages(question, hits),
        output_config={"effort": EFFORT},
        # if a safety classifier declines, the API retries on the recommended fallback model
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    return parse_response(response, hits)


def format_answer(answer: Answer) -> str:
    """Markdown version of the answer, used by the CLI."""
    lines = [answer.text, ""]
    if answer.citations:
        lines.append("**Sources**")
        lines += [
            f"{c.number}. [PEP {c.pep} – {c.title} › {c.section}]({c.url})"
            for c in answer.citations
        ]
    return "\n".join(lines).strip()
