import json

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from paper_extraction import models
from paper_extraction.api import create_app
from paper_extraction.source_image_cache import cached_transcription
from paper_extraction.store import Store
from paper_extraction.usage import invoke_tracked, usage_session, usage_stage, usage_summary


def test_usage_counts_schema_and_http_retries_without_persisting_text(tmp_path, monkeypatch):
    replies = [
        httpx.Response(500, json={"error": {"message": "temporary"}}),
        ("not JSON", {"prompt_tokens": 100, "completion_tokens": 9, "total_tokens": 109,
                      "prompt_tokens_details": {"cached_tokens": 30},
                      "completion_tokens_details": {"reasoning_tokens": 3}}),
        ('{"candidates":[],"uncertainties":[]}', {"prompt_tokens": 60, "completion_tokens": 4, "total_tokens": 64}),
        ("image reading", None),
        ("tool decision", None),
    ]

    def serve(request):
        reply = replies.pop(0)
        if isinstance(reply, httpx.Response):
            return reply
        text, usage = reply
        data = {"id": "fake", "model": "fixture", "object": "chat.completion", "created": 0,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}
        if usage is not None:
            data["usage"] = usage
        return httpx.Response(200, json=data)

    settings = dict(provider="openai", base_url="https://offline.invalid", api_key="test-private-key",
                    text_model="fixture-text", vision_model="fixture-vision", text_reasoning_effort="medium")
    store = Store(tmp_path)
    store.create("b", {"provider": "openai"}, [dict(id="paper1", name="Fixture", kind="tex", root=None, source="fixture.tex")])
    work = store.work(store.paper("paper1"))
    with httpx.Client(transport=httpx.MockTransport(serve)) as http:
        monkeypatch.setattr(models, "ChatOpenAI", lambda **kw: ChatOpenAI(http_client=http, **kw))
        with usage_session(work, "paper1", 1, settings), usage_stage("extract", 0):
            adapter = models.Models(settings)
            packet = {"passages": [{"id": "p2-1", "text": "private passage"},
                                   {"id": "private-location", "text": "private note"}]}
            assert not adapter.structured(models.Findings, "private instruction", packet).candidates
            with usage_stage("source_image_lookup"):
                for cached in (False, True):
                    assert cached_transcription(work, settings, b"private pixels", 2, adapter.transcribe)[1] is cached
        # A resumed attempt shares a work directory but must get its own accounting.
        with usage_session(work, "paper1", 2, settings), usage_stage("review", 1):
            invoke_tracked(adapter.tool_model, [HumanMessage(content="private question")],
                           operation="evidence_decision", channel="text")

    assert not replies
    content = (work / "model-usage.jsonl").read_text("utf-8")
    assert all(sensitive not in content for sensitive in ("private", "not JSON", "image reading", "tool decision"))
    rows = [json.loads(line) for line in content.splitlines()]
    assert [row["http_requests"] for row in rows] == [2, 1, 1, 0, 1]
    assert rows[0]["source_ids"] == rows[1]["source_ids"] == ["p2-1"]
    assert rows[0]["source_text_bytes"] == len(b"private passage")
    assert rows[0]["reasoning_effort"] == "medium" and rows[0]["operation"] == "Findings"
    assert rows[2]["model"] == "fixture-vision" and rows[2]["stage"] == "source_image_lookup"
    assert rows[4]["attempt"] == 2 and rows[4]["draft_round"] == 1
    summary = usage_summary(work.parent)
    client = TestClient(create_app(tmp_path, runner=lambda *_: None))
    assert client.get("/api/papers/paper1/usage").json() == summary
    assert client.get("/api/papers/missing/usage").status_code == 404
    totals = summary["totals"]
    assert totals["model_calls"] == 4 and totals["http_requests"] == 5
    assert totals["schema_retry_calls"] == totals["cache_hits"] == 1
    assert totals["input_tokens"] == 160 and totals["output_tokens"] == 13
    assert totals["input_tokens_unknown_calls"] == totals["output_tokens_unknown_calls"] == 2
    assert totals["cache_read_tokens"] == 30 and totals["reasoning_tokens"] == 3


def test_usage_failure_with_no_response_has_unknown_tokens(tmp_path):
    class OfflineFailure:
        def invoke(self, messages):
            raise RuntimeError("content and credential must stay out of the ledger")

    assert not usage_summary(tmp_path)["recorded"]
    with usage_session(tmp_path / "attempt-1", "p", 4, {}), pytest.raises(RuntimeError):
        invoke_tracked(OfflineFailure(), [], operation="evidence_decision", channel="text")
    row, = usage_summary(tmp_path)["groups"]
    assert row["attempt"] == 4 and row["error_calls"] == 1 and row["input_tokens_unknown_calls"] == 1
    assert "input_tokens" not in row
