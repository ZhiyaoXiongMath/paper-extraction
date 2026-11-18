"""A model change starts a fresh per-paper attempt while the batch's settings remain frozen."""
import json

import pytest
from fastapi.testclient import TestClient

from paper_extraction import agent, worker
from paper_extraction.api import create_app
from paper_extraction.store import Store


def failed_batch(tmp_path):
    store = Store(tmp_path)
    source = tmp_path / "snapshot.tex"
    source.write_text("Snapshot", encoding="utf-8")
    previous = {"provider": "deepseek", "text_model": "original-text", "vision_model": "original-vision",
                "base_url": "https://old.example.invalid", "timeout": 120}
    store.create("batch", previous, [dict(id=k, name=k, kind="tex", source=str(source), root=None) for k in ("first", "sibling")])
    initial = store.paper("first")
    work = store.work(initial)
    work.mkdir(parents=True)
    (work / "checkpoint.sqlite").write_bytes(b"prior checkpoint")
    store.finish(initial, {"status": "failed", "resumable": True, "reason": "Connection interrupted."})
    sibling = store.paper("sibling")
    directory = store.work(sibling)
    directory.mkdir(parents=True)
    for name in agent._DOCUMENTS:
        (directory / name).write_text("Original published copy", encoding="utf-8")
    (directory / "published.json").write_text("Prior receipt.", encoding="utf-8")
    store.finish(sibling, {"status": "completed", "result_dir": str(directory)})
    return store, initial, previous


def test_switch_then_retry_and_resume_uses_public_frozen_attempt_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_EXTRACTION_ENV_FILE", str(tmp_path / "unused.env"))
    monkeypatch.setenv("OPENAI_API_KEY", "offline-openai-token")
    monkeypatch.setenv("OPENAI_TEXT_MODEL", "frozen-text")
    monkeypatch.setenv("OPENAI_VISION_MODEL", "frozen-vision")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://frozen.example.invalid")
    store, old, batch_settings = failed_batch(tmp_path)
    old_result = store.paper("first")["result"]
    sibling = store.paper("sibling")
    assert store.retry("first", provider="openai") == "openai"
    restarted = Store(tmp_path)
    new = restarted.paper("first")
    assert new["attempt"] == 2 and restarted.work(new) != store.work(old)
    assert new["source"] == old["source"] and new["root"] == old["root"]
    assert "retry_legacy_review" not in new["result"] and not (restarted.work(new) / "checkpoint.sqlite").exists()
    assert restarted.settings("batch") == restarted.paper_settings(old) == batch_settings
    assert restarted.paper_settings(new)["text_model"] == "frozen-text"
    assert restarted.paper_settings(restarted.paper("sibling")) == batch_settings
    assert restarted.paper("sibling") == sibling and (store.work(sibling)/"published.json").read_text() == "Prior receipt."
    with restarted.db() as db:
        frozen = json.loads(db.execute("SELECT settings FROM paper_configs WHERE paper_id='first'").fetchone()[0])
        previous = json.loads(db.execute("SELECT result FROM archived_reviews WHERE paper_id='first' AND attempt=1").fetchone()[0])
    assert "api_key" not in frozen and previous == old_result

    monkeypatch.setenv("OPENAI_TEXT_MODEL", "later-env-text")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://changed.example.invalid")
    monkeypatch.setenv("OPENAI_API_KEY", "rotated-offline-token")
    calls = []

    def fake_run(work, source, root, settings, notify, *, retry_legacy_review, read_api_key, save_draft,
                 previous_failure, pause_requested, count_draft_starts):
        assert previous_failure is None
        assert not pause_requested()
        assert count_draft_starts() == 0
        calls.append((work, dict(settings), retry_legacy_review))
        assert read_api_key() == settings["api_key"]
        work.mkdir(parents=True, exist_ok=True)
        (work/"checkpoint.sqlite").touch()
        return {"status": "failed", "reason": "Offline transient error.", "resumable": True}

    monkeypatch.setattr(agent, "run_paper", fake_run)
    worker.process(tmp_path, "first")
    assert calls[0][0] == store.work(new) and calls[0][2] is False
    assert calls[0][1] == {**frozen, "api_key": "rotated-offline-token"}
    restarted.retry("first")
    third = Store(tmp_path).paper("first")
    assert third["attempt"] == 3 and store.work(third) == store.work(new)
    assert store.paper_settings(third) == frozen
    providers = {p["id"]: (p["provider"], p["text_model"]) for p in store.describe("batch")["papers"]}
    assert providers == {"first": ("openai", "frozen-text"), "sibling": ("deepseek", "original-text")}
    assert (store.work(old)/"checkpoint.sqlite").read_bytes() == b"prior checkpoint"


def test_missing_openai_key_leaves_original_attempt_untouched(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_EXTRACTION_ENV_FILE", str(tmp_path / "unused.env"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    store, old, batch_settings = failed_batch(tmp_path)
    initial = store.paper("first")
    with TestClient(create_app(tmp_path, runner=lambda *args: pytest.fail("Unexpected dispatch"))) as client:
        response = client.post("/api/papers/first/retry", json={"provider": "openai"})
        assert response.status_code == 400 and "OPENAI_API_KEY" in response.text
    assert store.paper("first") == initial and store.settings("batch") == batch_settings
    with store.db() as db:
        assert db.execute("SELECT count(*) FROM archived_reviews").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM paper_configs").fetchone()[0] == 0


def test_new_profile_is_pinned_through_provider_switch(tmp_path, monkeypatch):
    from paper_extraction.profile_inputs import new_batch_profile

    monkeypatch.setenv("OPENAI_API_KEY", "offline-key")
    store = Store(tmp_path)
    snapshot = new_batch_profile()
    store.create("batch", {"provider": "deepseek", **snapshot},
                 [dict(id="paper", name="example", kind="tex", source="example.tex", root=None)])
    store.finish(store.paper("paper"), {"status": "failed", "reason": "fixture"})
    store.retry("paper", provider="openai")
    effective = store.paper_settings(store.paper("paper"))
    assert effective["provider"] == "openai"
    assert all(effective[name] == value for name, value in snapshot.items())


@pytest.mark.parametrize("requested,expected", [(None, "deepseek"), ({"provider": "openai"}, "openai")])
def test_http_retry_with_or_without_provider(tmp_path, monkeypatch, requested, expected):
    monkeypatch.setenv("PAPER_EXTRACTION_ENV_FILE", str(tmp_path / "unused.env"))
    monkeypatch.setenv("OPENAI_API_KEY", "offline-key")
    store, _, _ = failed_batch(tmp_path)
    with TestClient(create_app(tmp_path, runner=lambda *args: None)) as client:
        kwargs = {"json": requested} if requested else {}
        response = client.post("/api/papers/first/retry", **kwargs)
        assert response.status_code == 200 and response.json()["provider"] == expected
        data = client.get("/api/batches/batch").json()
        assert data["provider"] == "deepseek" and data["papers"][0]["provider"] == expected
    assert store.paper_settings(store.paper("first"))["provider"] == expected


def test_switching_back_freezes_new_settings_without_overwriting_prior_attempt(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_EXTRACTION_ENV_FILE", str(tmp_path / "unused.env"))
    monkeypatch.setenv("OPENAI_API_KEY", "offline-openai")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-deepseek")
    store, original, original_settings = failed_batch(tmp_path)
    store.retry("first", "openai")
    second = store.paper("first")
    store.finish(second, {"status": "failed", "resumable": True, "reason": "offline"})
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        store.retry("first")
    assert store.paper("first")["attempt"] == 2
    monkeypatch.setenv("DEEPSEEK_TEXT_MODEL", "explicit-switch-back-model")
    store.retry("first", "deepseek")
    third = store.paper("first")
    assert third["attempt"] == 3 and store.work(third) != store.work(second)
    assert store.paper_settings(original) == original_settings
    assert store.paper_settings(second)["provider"] == "openai"
    assert store.paper_settings(third)["text_model"] == "explicit-switch-back-model"

