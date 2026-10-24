"""Run generate_answer through the real Anthropic SDK with a mocked HTTP transport.

No network or API key needed: the transport records the request body and returns a
Messages API response that the SDK parses.
"""

import json

import anthropic
import httpx2
import pytest

from pepbot.generate import generate_answer
from pepbot.retrieve import Hit

RESPONSE = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-5-5",
    "content": [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {
            "type": "text",
            "text": "Use the walrus operator to assign inside an expression",
            "citations": [
                {
                    "type": "search_result_location",
                    "source": "https://x/572",
                    "title": "PEP 572",
                    "cited_text": "Assign values inside expressions.",
                    "search_result_index": 0,
                    "start_block_index": 0,
                    "end_block_index": 1,
                }
            ],
        },
        {"type": "text", "text": "."},
    ],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 812, "output_tokens": 64},
}


@pytest.fixture
def captured_client():
    captured = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content)
        return httpx2.Response(200, json=RESPONSE)

    client = anthropic.Anthropic(
        api_key="test-key",
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )
    return client, captured


def test_request_and_parsing_through_the_sdk(captured_client, chunks):
    client, captured = captured_client
    hits = [Hit(i, 1.0, row.to_dict()) for i, row in chunks.iterrows()]
    answer = generate_answer("What is :=?", hits[:2], client=client)

    body = captured["body"]
    assert captured["url"].startswith("https://api.anthropic.com/v1/messages")
    assert "server-side-fallback-2026-07-01" in captured["headers"]["anthropic-beta"]
    assert body["model"] == "claude-opus-5-5"
    assert body["fallbacks"] == "default"
    assert body["output_config"]["effort"] in {"low", "medium", "high"}
    blocks = body["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["search_result", "search_result", "text"]
    assert blocks[0]["citations"] == {"enabled": True}

    assert answer.text == "Use the walrus operator to assign inside an expression[1]."
    assert answer.citations[0].pep == 572
    assert answer.citations[0].cited_text == "Assign values inside expressions."
    assert (answer.input_tokens, answer.output_tokens) == (812, 64)
