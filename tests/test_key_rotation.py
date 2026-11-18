"""Exercise live key rotation between persisted nodes without external model calls."""
import pytest

from paper_extraction import agent, worker
from paper_extraction.models import Findings, Review
from paper_extraction.selection import WholePaperSelection
from paper_extraction.store import Store
from test_agent import OfflineModels, ToolChat


@pytest.mark.parametrize("mode", ["complete", "missing", "echo", "denied"])
def test_worker_refreshes_only_the_key_between_model_nodes(tmp_path, monkeypatch, mode):
    old_key, new_key = "initial-fixture-secret", "rotated-fixture-secret"
    env_file = tmp_path / "private.env"
    monkeypatch.setenv("PAPER_EXTRACTION_ENV_FILE", str(env_file))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def write_config(value):
        env_file.write_text(f"OPENAI_API_KEY={value}\nOPENAI_BASE_URL=https://changed.invalid\n"
                            "OPENAI_TEXT_MODEL=updated-text\nOPENAI_VISION_MODEL=updated-vision\n", encoding="utf-8")

    write_config(old_key)
    frozen = {"provider": "openai", "text_model": "bound-text", "vision_model": "bound-vision",
              "base_url": "https://snapshot.invalid", "timeout": 77}
    source = tmp_path / "paper.tex"
    source.write_text("The source claim applies to compact objects.", encoding="utf-8")
    store = Store(tmp_path / "app")
    store.create("b", frozen, [dict(id="p", name="sample", kind="tex", source=str(source), root=None)])
    calls, clients = [], []

    class Chat(ToolChat):
        def invoke(self, messages):
            key = self.owner.key
            calls.append(("tools", key))
            if len(calls) == 1:
                write_config("" if mode == "missing" else new_key)
            if key == new_key and mode in {"echo", "denied"}:
                if mode == "denied":
                    class ProviderError(Exception):
                        status_code = 401
                    raise ProviderError("Incorrect API key: rotated-fixt…et")
                raise RuntimeError(f"echoed credentials: {old_key} {new_key}")
            return super().invoke(messages)

    class Clients(OfflineModels):
        def __init__(self, settings):
            super().__init__()
            self.key = settings["api_key"]
            self.text = Chat(self)
            clients.append(dict(settings))

        def structured(self, schema, instructions, payload, image=None):
            calls.append((schema, self.key))
            return super().structured(schema, instructions, payload, image)

    monkeypatch.setattr(agent, "Models", Clients)
    monkeypatch.setattr(agent, "_render_pages", lambda _: [(1, b"fixture")])
    def build(directory, *unused):
        case = directory / "artifacts"
        case.mkdir()
        for filename in agent._DOCUMENTS:
            (case / filename).write_text("offline")
        return case
    monkeypatch.setattr(agent, "build_case", build)
    worker.process(store.home, "p")
    paper = store.paper("p")
    work = store.work(paper)
    assert calls[:3] == [("tools", old_key), ("tools", old_key), (Findings, old_key)]
    if mode == "missing":
        assert paper["result"]["resumable"] and "缺少" in paper["reason"]
        assert len(clients) == 1 and len(calls) == 3
        write_config(new_key)
        store.retry("p")
        assert store.work(store.paper("p")) == work
        worker.process(store.home, "p")
        paper = store.paper("p")
    assert clients == [{**frozen, "api_key": old_key}, {**frozen, "api_key": new_key}]
    assert calls[3] == (WholePaperSelection, new_key)
    assert all(key == new_key for _, key in calls[3:])
    if mode in {"echo", "denied"}:
        assert paper["result"]["resumable"] and paper["status"] == "failed"
        assert ("HTTP 401" if mode == "denied" else "[redacted]") in paper["reason"]
    else:
        assert paper["status"] == "completed", paper["reason"]
        assert calls.count((Findings, old_key)) == 1 and calls[-1] == (Review, new_key)
    persisted = [file.read_bytes() for file in work.glob("checkpoint.sqlite*")]
    persisted.append(str(store.describe("b")).encode())
    for secret in (old_key.encode(), new_key.encode(), "rotated-fixt…et".encode()):
        assert all(secret not in data for data in persisted)
