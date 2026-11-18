"""Exercise the OpenAI adapter against an in-memory HTTP transport."""
import json

import httpx
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from paper_extraction import models
from paper_extraction.api import create_app
from paper_extraction.settings import provider_settings
from paper_extraction.store import Store


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


def test_official_gpt_62_tool_roundtrip_replays_reasoning_without_server_state(monkeypatch):
    requests = []

    def serve(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if len(requests) == 1:
            output = [dict(type="reasoning", id="rs_fixture", summary=[], encrypted_content="fixture-encrypted"),
                      dict(type="function_call", id="fc_fixture", call_id="call_fixture", name="read_span",
                           arguments='{"location":"p1-1"}', status="completed")]
        else:
            output = [dict(type="message", id="msg_fixture", role="assistant", status="completed",
                           content=[dict(type="output_text", text="Source checked.", annotations=[])])]
        return httpx.Response(200, json={"id": "resp_fixture", "object": "response", "created_at": 0,
            "model": body["model"], "output": output, "status": "completed"})

    @tool
    def read_span(location: str) -> str:
        """Read a registered source passage."""
        return location

    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        monkeypatch.setattr(models, "ChatOpenAI", lambda **kwargs: ChatOpenAI(http_client=client, **kwargs))
        settings = dict(provider="openai", base_url="https://api.openai.com/v1", api_key="offline-fixture",
                        text_model="gpt-6.2-sol", vision_model="gpt-6.2-sol",
                        text_reasoning_effort="high", vision_reasoning_effort="low")
        adapter = models.Models(settings)
        original = HumanMessage(content="Read the selected source passage.")
        model = adapter.tool_model.bind_tools([read_span])
        reply = model.invoke([original])
        tool_result = ToolMessage(content="The registered source passage.", tool_call_id=reply.tool_calls[0]["id"])
        assert model.invoke([original, reply, tool_result]).text == "Source checked."
        assert not adapter.text._use_responses_api({}) and not adapter.vision._use_responses_api({})
        assert adapter.text._get_request_payload([original])["reasoning_effort"] == "high"
        assert adapter.vision._get_request_payload([original])["reasoning_effort"] == "low"
        legacy = models.Models({k: v for k, v in settings.items() if not k.endswith("_reasoning_effort")})
        assert "reasoning_effort" not in legacy.text._get_request_payload([original])
    for path, body in requests:
        assert path == "/v1/responses" and body["reasoning"] == {"effort": "high"} and body["store"] is False
        assert body["include"] == ["reasoning.encrypted_content"]
    assert any(i.get("encrypted_content") == "fixture-encrypted" for i in requests[1][1]["input"])
    assert requests[1][1]["input"][-1] == {
        "type": "function_call_output", "output": "The registered source passage.", "call_id": "call_fixture",
    }


def test_openai_efforts_are_saved_probed_and_frozen_per_batch(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-fixture")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://fixture.invalid/v1")
    monkeypatch.setenv("OPENAI_TEXT_MODEL", "gpt-6.2-sol")
    monkeypatch.setenv("OPENAI_VISION_MODEL", "gpt-5.4-mini")
    monkeypatch.setattr("secrets.randbelow", lambda _: 234)
    requests = []

    def serve(request):
        body = json.loads(request.content)
        requests.append(body)
        message = "READY" if len(requests) == 1 else "1234"
        return httpx.Response(200, json={"id": "fixture", "object": "chat.completion", "created": 0,
            "model": body["model"], "choices": [{"index": 0, "message": {"role": "assistant", "content": message},
                                                 "finish_reason": "stop"}]})

    with httpx.Client(transport=httpx.MockTransport(serve)) as offline:
        monkeypatch.setattr("langchain_openai.ChatOpenAI", lambda **kwargs: ChatOpenAI(http_client=offline, **kwargs))
        with TestClient(create_app(tmp_path, runner=lambda *_: None)) as client:
            config = client.get("/api/config").json()
            openai = next(p for p in config["providers"] if p["id"] == "openai")
            assert [openai["reasoning"][mode]["effort"] for mode in ("text", "vision")] == ["medium", "medium"]
            assert client.post("/api/config/openai/reasoning", json={"vision": "max"}).status_code == 400
            assert client.post("/api/config/openai/reasoning", json={"text": "high"}).status_code == 200
            reply = client.post("/api/config/openai/reasoning", json={"vision": "low"}).json()["reasoning"]
            assert [reply[mode]["effort"] for mode in ("text", "vision")] == ["high", "low"]
            for mode in ("text", "vision"):
                assert client.post("/api/config/test", json={"provider": "openai", "mode": mode}).json()["ok"]
            paper_name = "sample.tex"
            preview = client.post("/api/imports", files=[("files", (paper_name, b"Sample text"))],
                                  data={"paths": json.dumps([paper_name])}).json()
            choice = preview["papers"][0]
            batch = client.post("/api/batches", json={"provider": "openai", "import_id": preview["id"],
                                "papers": [{"id": choice["id"], "entry": choice["entry"]}]})
            assert batch.status_code == 201, batch.text
            assert client.post("/api/config/openai/reasoning", json={"text": "max"}).status_code == 200
    assert [r["reasoning_effort"] for r in requests] == ["high", "low"]
    assert requests[1]["messages"][0]["content"][1]["type"] == "image_url"
    store = Store(tmp_path)
    assert store.reasoning_efforts("openai") == {"text": "max", "vision": "low"}
    frozen = store.settings(batch.json()["id"])
    assert [frozen[mode + "_reasoning_effort"] for mode in ("text", "vision")] == ["high", "low"]
    assert "api_key" not in frozen
