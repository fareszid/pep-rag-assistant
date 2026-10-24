from types import SimpleNamespace as NS

from pepbot.generate import build_messages, format_answer, generate_answer, parse_response
from pepbot.retrieve import Hit


def make_hits(chunks):
    return [Hit(i, 1.0, row.to_dict()) for i, row in chunks.iterrows()]


def text(t, citations=None):
    return NS(type="text", text=t, citations=citations)


def cite(index, cited="..."):
    return NS(type="search_result_location", search_result_index=index, cited_text=cited)


def test_build_messages_uses_citable_search_results(chunks):
    hits = make_hits(chunks.assign(text=chunks["text"] + "\n\nSecond paragraph."))
    [message] = build_messages("What is :=?", hits[:2])
    blocks = message["content"]
    assert [b["type"] for b in blocks] == ["search_result", "search_result", "text"]
    first = blocks[0]
    assert first["citations"] == {"enabled": True}
    assert first["source"] == "https://peps.python.org/pep-0572/"
    assert len(first["content"]) == 2  # one text block per paragraph
    assert blocks[-1]["text"] == "What is :=?"


def test_parse_response_numbers_citations_in_order_of_first_use(chunks):
    hits = make_hits(chunks)
    response = NS(
        stop_reason="end_turn",
        model="claude-opus-5-5",
        usage=NS(input_tokens=900, output_tokens=120),
        content=[
            NS(type="thinking", thinking=""),
            text("Use the walrus operator", [cite(0), cite(0)]),
            text(" and f-strings", [cite(3)]),
            text(", both from PEPs.", None),
            text(" Again walrus", [cite(0)]),
        ],
    )
    answer = parse_response(response, hits)
    assert answer.text == (
        "Use the walrus operator[1] and f-strings[2], both from PEPs. Again walrus[1]"
    )
    assert [(c.number, c.pep) for c in answer.citations] == [(1, 572), (2, 498)]
    assert answer.input_tokens == 900
    assert "**Sources**" in format_answer(answer)


def test_refusal_is_reported(chunks):
    response = NS(stop_reason="refusal", model="m", content=[], usage=None)
    answer = parse_response(response, make_hits(chunks))
    assert answer.stop_reason == "refusal" and answer.citations == []


class FakeClient:
    def __init__(self, response):
        self.kwargs = None
        self.beta = NS(messages=NS(create=self._create))
        self._response = response

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return self._response


def test_generate_answer_request_shape(chunks):
    response = NS(
        stop_reason="end_turn",
        model="claude-opus-5-5",
        usage=None,
        content=[text("Answer", [cite(1)])],
    )
    client = FakeClient(response)
    answer = generate_answer("indentation?", make_hits(chunks), client=client)
    assert client.kwargs["model"] == "claude-opus-5-5"
    assert client.kwargs["fallbacks"] == "default"
    assert client.kwargs["betas"] == ["server-side-fallback-2026-07-01"]
    assert client.kwargs["output_config"]["effort"] in {"low", "medium", "high"}
    assert "temperature" not in client.kwargs
    assert answer.citations[0].pep == 8
