import json
import threading

import pytest
from fastapi.testclient import TestClient

from paper_extraction import agent, worker
from paper_extraction.api import create_app
from paper_extraction.models import AdmissionReview, Draft, Findings, Review
from paper_extraction.retry_context import previous_failure
from paper_extraction.scope_reconciliation import ScopeReconciliation, scope_failure_reason
from paper_extraction.selection import WholePaperSelection, select_contributions
from paper_extraction.store import Store
from test_agent import OfflineModels
from test_whole_selection import finding, pick


def failed_scope():
    rows = [{"kind": "current", "decision": "admitted", "scope_evidence_id": "t1-1",
             "selection_evidence_ids": ["t1-1", "p99-1"], "source_number": "Theorem " + str(i),
             "independent_question": "Is statement " + str(i) + " independent?",
             "decision_reason": "The former reader treated claim " + str(i) + " as a separate result."}
            for i in range(1, 8)]
    return {"status": "failed", "reason": "v13 范围例外：七项陈述需要重新核对", "candidates": rows, "draft": {}}


def failed_review():
    note = "Check the hypotheses of the result."
    return {"status": "failed", "reason": note,
            "draft": {"title": "Old text must stay private", "evidence_links": []},
            "reviews": [{"stage": "review", "attempt": 1, "passed": False, "issues": [note],
                         "defects": [{"draft_field": "groups.0.theorem", "kind": "incorrect_claim",
                                      "required_change": note, "evidence_id": "t1-1",
                                      "current_excerpt": "unbounded previous draft text" * 1000}]}]}


def test_runtime_exception_after_a_verdict_does_not_become_paper_guidance():
    review = failed_review()
    review["reason"] = "APITimeoutError: deadline exceeded"
    assert not previous_failure(review, 1)
    assert not previous_failure({"status": "failed", "reason": "pdflatex missing",
                                 "reviews": [{"stage": "compile", "passed": False, "attempt": 1,
                                              "issues": ["pdflatex missing"]}]}, 1)


def test_interrupted_selection_carries_the_original_binding_across_models():
    row = {"kind": "current", "source_number": "Theorem 5", "claim": "a sourced claim", "assumptions": "compact", "dependencies": [],
           "decision": "admitted", "decision_reason": "prior estimate", "evidence_ids": ["p7-1"]}
    state = {"status": "failed", "resumable": True, "candidates": [row],
             "reason": "ValueError: Scope reconciliation needs corrected source bindings for c1"}
    [issue] = previous_failure(state, 2)["issues"]
    assert (issue["previous_candidate"], issue["source_number"], issue["evidence_ids"]) == ("c1", "Theorem 5", ["p7-1"])


def test_fresh_scope_reconciliation_receives_the_previous_questions_and_can_still_report_an_exception():
    previous = previous_failure(failed_scope(), 1)
    source = [{"id": "t1-1", "section": "Introduction", "text": "The paper solves five independent problems."}]
    rows = [finding("current", f"Theorem {i}", f"Statement {i}", "t1-1") for i in range(5)]
    schemas = []
    class Model:
        def structured(self, schema, instructions, packet):
            schemas.append(schema)
            assert instructions.startswith(agent._rules())
            assert len(packet["previous_failure"]["issues"]) == 7
            if schema is WholePaperSelection:
                return schema(selected=[pick(f"c{i}", "t1-1") for i in range(1, 6)])
            return schema(assessments=[dict(candidate_id=f"c{i}", source_id="t1-1", outcome="admitted",
                       source_reason=f"The paper solves question {i}.", independent_question=f"Question {i}?")
                       for i in range(1, 6)])
    result = select_contributions(source, rows, Model(), previous, agent._rules())
    assert schemas == [WholePaperSelection, ScopeReconciliation]
    assert "5 个当前定理组" in scope_failure_reason(result)


@pytest.mark.parametrize("mode", ["scope_with_switch", "review_with_no_checkpoint"])
def test_single_paper_retry_reaches_the_relevant_model_node_with_prior_questions(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("PAPER_EXTRACTION_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("OPENAI_API_KEY", "offline-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-key")
    source = tmp_path / "source.tex"
    source.write_text("The source claim holds for compact objects.", encoding="utf-8")
    store = Store(tmp_path / "home")
    store.create("b", {"provider": "deepseek", "text_model": "offline", "vision_model": "offline",
                       "base_url": "http://localhost"},
                 [dict(id="p", name="single", kind="tex", source=str(source), root=None)])
    first = store.paper("p")
    old = failed_scope() if mode == "scope_with_switch" else failed_review()
    if mode == "scope_with_switch":
        store.work(first).mkdir(parents=True)
        (store.work(first) / "checkpoint.sqlite").write_bytes(b"old isolated work")
    store.finish(first, old)
    seen = {}
    models = OfflineModels()
    original = models.structured
    def response(schema, instructions, packet, image=None):
        seen[schema] = (packet.get("previous_failure"), instructions)
        if packet.get("previous_failure"):
            assert "Old text must stay private" not in str(packet["previous_failure"])
            assert "unbounded previous draft text" not in str(packet["previous_failure"])
            assert all(issue["evidence_ids"] == ["t1-1"] for issue in packet["previous_failure"]["issues"])
            assert instructions.startswith(agent._rules())
        return original(schema, instructions, packet, image)
    models.structured = response
    monkeypatch.setattr(agent, "Models", lambda _: models)
    monkeypatch.setattr(agent, "_render_pages", lambda _: [(1, b"fixture")])
    def build(work, *_):
        dest = work / "offline-fixture"
        dest.mkdir()
        for name in agent._DOCUMENTS:
            (dest / name).write_text("offline fixture", encoding="utf-8")
        return dest
    monkeypatch.setattr(agent, "build_case", build)
    done = threading.Event()
    def run(home, pid):
        try:
            worker.process(home, pid)
        finally:
            done.set()
    with TestClient(create_app(store.home, runner=run)) as client:
        kwargs = {"json": {"provider": "openai"}} if mode == "scope_with_switch" else {}
        assert client.post("/api/papers/p/retry", **kwargs).status_code == 200
        assert done.wait(10)
    new = store.paper("p")
    assert new["status"] == "completed", new["reason"]
    with store.db() as db:
        event = json.loads(db.execute("SELECT data FROM events WHERE paper_id='p' AND stage='start' AND attempt=2").fetchone()[0])
    assert event["previous_failure"] == previous_failure(old, 1)
    assert store.work(first) != store.work(new) and (store.work(new) / "checkpoint.sqlite").is_file()
    assert seen[Findings][0] is None and seen[Review][0] is None
    if mode == "scope_with_switch":
        scope = seen[WholePaperSelection][0]
        assert scope["stage"] == "select" and scope["issue_count"] == len(scope["issues"]) == 7
        assert [item["source_number"] for item in scope["issues"]] == [f"Theorem {i}" for i in range(1, 8)]
        assert seen[Draft][0] is None and seen[AdmissionReview][0] is None
        assert len(new["result"]["draft"]["groups"]) == 1
    else:
        assert seen[WholePaperSelection][0] is None
        assert seen[Draft][0] == seen[AdmissionReview][0] == previous_failure(old, 1)
