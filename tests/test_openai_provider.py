"""Exercise the OpenAI adapter against an in-memory HTTP transport."""
import json

import httpx
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from paper_extraction import models
from paper_extraction.settings import provider_settings


def test_openai_tools_images_and_length_retry(monkeypatch):
    requests = []
    call = {"id": "lookup", "type": "function",
            "function": {"name": "read_span", "arguments": '{"location":"p1-1"}'}}
    replies = [
        ({"content": None, "tool_calls": [call]}, "tool_calls"),
        ({"content": '{"passed":'}, "length"),
        ({"content": '{"passed":true,"issues":[]}'}, "stop"),
    ]

    def serve(request):
        assert request.url.path == "/v1/chat/completions"
        body = json.loads(request.content)
        requests.append(body)
        message, finish = replies.pop(0)
        return httpx.Response(200, json={"id": "fixture", "object": "chat.completion", "created": 0,
            "model": body["model"], "choices": [{"index": 0, "message": {"role": "assistant", **message},
                                                 "finish_reason": finish}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 128, "total_tokens": 228}})

    monkeypatch.setenv("OPENAI_API_KEY", "offline-fixture")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fixture.invalid/v1")
    monkeypatch.setenv("OPENAI_TEXT_MODEL", "text-fixture")
    monkeypatch.setenv("OPENAI_VISION_MODEL", "vision-fixture")
    assert "api_key" not in provider_settings("openai")
    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        monkeypatch.setattr(models, "ChatOpenAI", lambda **kwargs: ChatOpenAI(http_client=client, **kwargs))
        adapter = models.Models(provider_settings("openai", with_key=True))

        @tool
        def read_span(location: str) -> str:
            """Read a registered source passage."""
            return location

        assert adapter.text.bind_tools([read_span]).invoke("Read the source.").tool_calls[0]["args"] == {"location": "p1-1"}
        assert adapter.structured(models.Review, "Check this page", {}, image=b"fixture").passed
    assert [r["model"] for r in requests] == ["text-fixture", "vision-fixture", "vision-fixture"]
    assert requests[1]["messages"][1]["content"][1]["type"] == "image_url"
    assert requests[2]["max_completion_tokens"] > 128
    assert all("thinking" not in r and "max_tokens" not in r for r in requests)
