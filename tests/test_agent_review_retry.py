"""Manual retry upgrades a completed legacy content review without replaying source extraction."""
import json
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from paper_extraction import agent, worker
from paper_extraction.store import CONTENT_REVIEW_VERSION, Store, legacy_content_failure
from paper_extraction.selection import SELECTION_VERSION
from test_agent import OfflineModels


def old_result():
    reviews = [{"stage": "review", "passed": False, "attempt": i, "route": "draft",
                "issues": ["Old individual reviewers asked for all proof constants."]} for i in (1, 2, 3)]
    candidates = [{"kind": kind, "claim": "source claim", "assumptions": "compact", "significance": "central",
                   "source_number": kind, "decision": "admitted", "decision_reason": "central",
                   "dependencies": [], "evidence_ids": ["t1-1"]} for kind in ("current", "historical")]
    return {"status": "failed", "draft": {"groups": [{"theorem": "Old rejected draft"}]},
            "candidates": candidates, "reason": reviews[-1]["issues"][0], "reviews": reviews}


@pytest.mark.parametrize("stage,version,attempt", [
    ("review", None, 1), ("compile", None, 3), ("visual", None, 3), ("review", CONTENT_REVIEW_VERSION, 3),
])
def test_current_or_engineering_failure_does_not_qualify(stage, version, attempt):
    status = old_result()
    status["reviews"][-1].update(stage=stage, attempt=attempt, review_version=version)
    assert not legacy_content_failure(status)


@pytest.mark.parametrize("versions", [(None, None, None), (None, "source-packet-v1", "source-packet-v1")])
def test_explicit_retry_forks_after_extraction_and_keeps_original_trace(tmp_path, monkeypatch, versions):
    store = Store(tmp_path / "home")
    source = tmp_path / "original.tex"
    source.write_text("The source claim is a statement on compact objects.", encoding="utf-8")
    settings = {"provider": "offline", "base_url": "http://localhost", "text_model": "mock", "vision_model": "mock"}
    paper = dict(id="paper", name="Original", kind="tex", source=str(source), root=None)
    store.create("batch", settings, [paper])
    initial = store.paper("paper")
    work = store.work(initial)
    work.mkdir(parents=True)
    terminal = old_result()
    for record, version in zip(terminal["reviews"], versions):
        if version:
            record["review_version"] = version
    config = {"configurable": {"thread_id": agent._thread_id(source, None, settings, agent._rules())}}
    with SqliteSaver.from_conn_string(str(work / "checkpoint.sqlite")) as saver:
        previous = StateGraph(agent.PaperState)
        previous.add_node("extract", lambda s: {"cursor": 1, "uncertainties": [], "target_ids": [],
                                                 "issues": [], "attempt": 3, **terminal})
        previous.add_edge(START, "extract")
        previous.add_edge("extract", END)
        compiled = previous.compile(checkpointer=saver)
        compiled.invoke({"cursor": 0}, config)
        previous_checkpoint = compiled.get_state(config).config
    store.finish(initial, terminal)

    models = OfflineModels()
    models.missing_application = True
    monkeypatch.setattr(agent, "Models", lambda cfg: models)
    monkeypatch.setattr(worker, "provider_settings", lambda *a, **kw: {"api_key": "fixture"})
    monkeypatch.setattr(agent, "_render_pages", lambda path: [(1, b"fixture")])

    def build(work, *unused):
        case = work / "new-compile"
        case.mkdir()
        for n in agent._DOCUMENTS:
            (case / n).write_text("fixture")
        return case

    monkeypatch.setattr(agent, "build_case", build)
    # Reading a previously failed work directory cannot spend tokens by itself.
    old = agent.run_paper(work, source, None, settings, lambda *args: None)
    assert old["status"] == "failed" and models.extracts == models.drafts == 0
    store.retry("paper")
    queued = store.paper("paper")
    assert queued["result"]["retry_legacy_review"] is True and store.work(queued) == work
    with store.db() as db:
        archived = json.loads(db.execute("SELECT result FROM archived_reviews WHERE paper_id='paper' AND attempt=1").fetchone()[0])
        assert archived["draft"] == terminal["draft"] and archived["reviews"] == terminal["reviews"]
    worker.process(store.home, "paper")
    outcome = store.paper("paper")["result"]
    assert outcome["status"] == "completed", outcome
    assert models.drafts == 2 and models.extracts == 1  # exclusively the requested missing application
    assert outcome["reviews"][0]["attempt"] == 1 and outcome["reviews"][0]["review_version"] == CONTENT_REVIEW_VERSION
    assert not any("proof constants" in str(r) for r in outcome["reviews"])
    logs = store.describe("batch")["papers"][0]["events"]
    [scan] = [event["data"] for event in logs if event["stage"] == "extract"]
    assert scan["targeted"] is True and set(scan["locations"]) == {"t1-1"}
    assert "chunk" not in scan and "total" not in scan
    with SqliteSaver.from_conn_string(str(work / "checkpoint.sqlite")) as saver:
        historical = saver.get(previous_checkpoint)
        assert historical["channel_values"]["reviews"] == terminal["reviews"]


def test_latest_current_verdict_prevents_migrating_old_intermediate_reviews(tmp_path):
    store = Store(tmp_path)
    store.create("batch", {}, [dict(id="id", name="Paper", kind="tex", source="paper.tex", root=None)])
    prior = store.paper("id")
    work = store.work(prior)
    work.mkdir(parents=True)
    (work / "checkpoint.sqlite").touch()
    result = old_result()
    result["repair_policy"] = "source-bound-repair-1"
    result["reviews"][1]["review_version"] = "source-packet-v1"
    result["reviews"][2]["review_version"] = CONTENT_REVIEW_VERSION
    result["selection_version"] = SELECTION_VERSION
    store.finish(prior, result)
    store.retry("id")
    queued = store.paper("id")
    assert queued["attempt"] == 2 and store.work(queued) != work
    assert "retry_legacy_review" not in queued["result"]
