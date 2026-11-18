"""Run text, image and tool requests against an offline provider transport."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from paper_extraction import models, settings
from paper_extraction.api import create_app
from paper_extraction.store import Store


def completion(body, message, finish="stop"):
    return httpx.Response(200, json={
        "id": "fixture", "object": "chat.completion", "created": 1, "model": body["model"],
        "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish}],
    })


@pytest.mark.parametrize(("provider", "text_name", "vision_name", "vision_effort"), [
    ("deepseek", "deepseek-flash", "deepseek-flash", "max"),
    ("zhipu", "glm-5.3", "glm-4.6v", None),
])
def test_enabled_thinking_replays_tools_and_respects_per_model_fields(
        monkeypatch, provider, text_name, vision_name, vision_effort):
    bodies = []

    def serve(request):
        body = json.loads(request.content)
        bodies.append(body)
        assert body["thinking"] == {"type": "enabled"}
        if len(bodies) == 1:
            return completion(body, {"content": None, "reasoning_content": "first-round-state",
                                     "tool_calls": [{"id": "call_fixture", "type": "function",
                                                     "function": {"name": "read_span", "arguments": '{"location":"t1-1"}'}}]},
                              finish="tool_calls")
        if len(bodies) == 2:
            assert body["messages"][1]["reasoning_content"] == "first-round-state"
            assert body["messages"][2]["tool_call_id"] == "call_fixture"
            return completion(body, {"content": "Checked", "reasoning_content": "second-round-state"})
        return completion(body, {"content": '{"passed":true,"issues":[]}'})

    @tool
    def read_span(location: str) -> str:
        """Read a source passage."""
        return location

    with httpx.Client(transport=httpx.MockTransport(serve)) as offline:
        reasoning_client = models.ReasoningChatOpenAI
        monkeypatch.setattr(models, "ReasoningChatOpenAI", lambda **kw: reasoning_client(http_client=offline, **kw))
        monkeypatch.setattr(models, "ChatOpenAI", lambda **kw: ChatOpenAI(http_client=offline, **kw))
        config = dict(provider=provider, text_model=text_name, vision_model=vision_name, api_key="fixture",
                      base_url="http://fixture.invalid/v1", text_reasoning_effort="low")
        if vision_effort:
            config["vision_reasoning_effort"] = vision_effort
        adapter = models.Models(config)
        tool_model = adapter.tool_model.bind_tools([read_span])
        original = HumanMessage(content="Read the source.")
        reply = tool_model.invoke([original])
        answer = tool_model.invoke([original, reply, ToolMessage(content="Source", tool_call_id="call_fixture")])
        assert answer.content == "Checked"
        assert adapter.structured(models.Review, "Check the page.", {}, image=b"fixture").passed

    assert [body["model"] for body in bodies] == [text_name, text_name, vision_name]
    assert [body.get("reasoning_effort") for body in bodies] == ["low", "low", vision_effort]


def test_each_provider_saves_its_own_efforts_for_probes_and_new_batches(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_EXTRACTION_ENV_FILE", str(tmp_path / "unused.env"))
    for prefix, key in (("DEEPSEEK", "DEEPSEEK_API_KEY"), ("ZHIPU", "ZAI_API_KEY")):
        monkeypatch.setenv(key, "mock-key")
        monkeypatch.setenv(prefix + "_BASE_URL", "http://fixture.invalid/v1")
        monkeypatch.delenv(prefix + "_TEXT_MODEL", raising=False)
        monkeypatch.delenv(prefix + "_VISION_MODEL", raising=False)
    monkeypatch.setattr("secrets.randbelow", lambda upper: 123)
    bodies = []

    def serve(request):
        body = json.loads(request.content)
        bodies.append(body)
        return completion(body, {"content": "1123" if isinstance(body["messages"][0]["content"], list) else "READY"})

    with httpx.Client(transport=httpx.MockTransport(serve)) as offline:
        monkeypatch.setattr("langchain_openai.ChatOpenAI", lambda **kw: ChatOpenAI(http_client=offline, **kw))
        with TestClient(create_app(tmp_path, runner=lambda *_: None)) as client:
            config = {p["id"]: p["reasoning"] for p in client.get("/api/config").json()["providers"]}
            assert [config["deepseek"][mode]["effort"] for mode in settings.MODEL_MODES] == ["high", "high"]
            assert config["zhipu"]["text"]["effort"] == "low"
            assert config["zhipu"]["vision"] == {"effort": None, "choices": []}
            assert client.post("/api/config/zhipu/reasoning", json={"vision": "low"}).status_code == 400
            assert client.post("/api/config/deepseek/reasoning", json={"text": "medium"}).status_code == 400
            for provider, patch in (("deepseek", {"text": "low", "vision": "max"}), ("zhipu", {"text": "max"})):
                assert client.post(f"/api/config/{provider}/reasoning", json=patch).status_code == 200
                for mode in settings.MODEL_MODES:
                    assert client.post("/api/config/test", json={"provider": provider, "mode": mode}).json()["ok"]

            preview = client.post("/api/imports", data={"paths": json.dumps(["sample.tex"])},
                                  files={"files": ("sample.tex", r"\documentclass{article}")}).json()
            batch = client.post("/api/batches", json={"provider": "deepseek", "import_id": preview["id"],
                                                     "papers": preview["papers"]})
            assert batch.status_code == 201, batch.text
            assert client.post("/api/config/deepseek/reasoning", json={"text": "high"}).status_code == 200

    assert all(body["thinking"] == {"type": "enabled"} for body in bodies)
    assert [body.get("reasoning_effort") for body in bodies] == ["low", "max", "max", None]
    store = Store(tmp_path)
    assert store.reasoning_efforts("deepseek") == {"text": "high", "vision": "max"}
    assert store.reasoning_efforts("zhipu") == {"text": "max"}
    frozen = store.settings(batch.json()["id"])
    assert [frozen.get(mode + "_reasoning_effort") for mode in settings.MODEL_MODES] == ["low", "max"]
    assert "api_key" not in frozen
