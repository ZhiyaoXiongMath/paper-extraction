"""Paper-wide editorial decisions and checkpoint upgrades, without external model calls."""
import json

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from paper_extraction import agent
from paper_extraction.models import AdmissionReview, Draft
from paper_extraction.selection import (
    SELECTION_VERSION, SelectionCorrection, WholePaperSelection, apply_selection, candidate_register, select_contributions,
)
from paper_extraction.store import CONTENT_REVIEW_VERSION, Store, stale_selection_failure
from test_agent import OfflineModels, paper  # noqa: F401
from test_agent_review_retry import old_result
from test_saved_draft_review import old_terminal


def finding(kind, label, text, loc):
    return {"kind": kind, "source_number": label, "claim": text, "assumptions": "compact",
            "significance": "locally significant", "decision": "admitted", "decision_reason": "local",
            "dependencies": [], "evidence_ids": [loc]}


def pick(cid, loc, **extra):
    return {"candidate_id": cid, "independent_question": "Which objects admit a solution?",
            "source_reason": "The introduction poses the existence problem and cites this result.",
            "evidence_ids": [loc], **extra}


def test_global_decisions_bind_core_and_defs_without_promoting_construction():
    records = [
        {"id": "p1", "text": "Abstract. Main theorem: exact criterion.", "section": "Introduction"},
        {"id": "p2", "text": "Previously the criterion required finite type.", "section": "History"},
        {"id": "p3", "text": "Corollary 8. As an application, all odd ranks occur.", "section": "Applications"},
        {"id": "p4", "text": "Proof. Auxiliary witnesses implement the argument.", "section": "Proof"},
    ]
    candidates = [
        finding("setting", None, "A. Author and B. Author, Exact Criteria", "p1"),
        finding("current", "Comparison", "Formerly finite type was needed; A proved it.", "p2"),
        finding("definition", "Definition 1", "The invariant of a compact object.", "p1"),
        finding("current", "Theorem 2", "Solvability is equivalent to vanishing of the invariant.", "p1"),
        finding("current", "Theorem 2 restated", "If and only if the invariant vanishes a solution exists.", "p4"),
        finding("application", "Corollary 8", "Solutions exist in every odd rank.", "p3"),
        finding("current", "Remark", "Possible extension to negative rank.", "p3"),
    ]
    candidates += [finding("current", f"Lemma {n}", f"At step {n} the chosen witness is controlled.", "p4")
                   for n in range(120)]
    repeated = {**candidates[3], "dependencies": ["invariant"], "evidence_ids": ["p3"]}
    register = candidate_register([*candidates, repeated])
    assert len(register) == len(candidates) and register[3]["evidence_ids"] == ["p1", "p3"]
    choice = WholePaperSelection(selected=[
        pick("c2", "p2", role="historical"), pick("c4", "p1", definition_ids=["c3"], variant_ids=["c5"]),
        pick("c6", "p3")], metadata_ids=["c1"],
        unresolved=[{"candidate_id": "c7", "source_reason": "The original leaves the extension open."}],
        exclusions=[{"candidate_ids": ["c5", "c7..c127"], "source_reason": "These witnesses serve the proof steps."}])
    rows = apply_selection(register, choice, records)
    assert [r["candidate_id"] for r in rows if r["decision"] == "admitted"] == ["c1", "c2", "c3", "c4", "c6"]
    assert rows[1]["kind"] == "historical" and rows[1]["local_kind"] == "current"
    assert rows[2]["needed_by"] == ["c4"] and rows[3]["selection_evidence_ids"] == ["p1"]
    assert rows[4]["represented_by"] == "c4" and rows[6]["decision"] == "unresolved"
    assert all(r["decision"] == "excluded" and r["local_decision"] == "admitted" for r in rows[7:])
    assert [r["candidate_id"] for r in candidate_register(rows)] == [r["candidate_id"] for r in rows]


def test_selection_repairs_invalid_ids_using_register_and_original_evidence():
    records = [{"id": "intro", "text": "We prove the main result.", "section": "Introduction"}]
    candidates = [finding("current", "Theorem A", "A solution exists.", "intro")]

    class Reply:
        calls = 0

        def structured(self, schema, instructions, payload):
            self.calls += 1
            if self.calls == 1:
                assert schema is WholePaperSelection and "B2." in instructions
                assert payload["passages"] == records
                assert payload["candidate_register"][0]["candidate_id"] == "c1"
                return schema(selected=[pick("c1", "wrong-id")])
            assert schema is SelectionCorrection
            assert "requires registered source IDs" in payload["conflict"]
            assert payload["previous_bindings"][0]["evidence_ids"] == ["wrong-id"]
            assert payload["original_passages"] == records and payload["affected_selected_ids"] == ["c1"]
            return schema(selected=[pick("c1", "intro")])

    model = Reply()
    assert select_contributions(records, candidates, model)[0]["decision"] == "admitted"
    assert model.calls == 2


@pytest.mark.parametrize("pending, prior_verdict", [
    (False, None), (True, None), (False, "confirmation"), (False, "citation"),
    (False, "patch"), (False, "layout"), (True, "patch"),
])
def test_old_compile_failure_and_pending_publish_reselect_before_delivery(tmp_path, monkeypatch, pending, prior_verdict):
    source = tmp_path / "source.tex"
    source.write_text("The source claim is a statement on compact objects.", encoding="utf-8")
    settings = {"provider": "offline", "text_model": "mock", "vision_model": "mock", "base_url": "http://localhost"}
    store = Store(tmp_path / "home")
    store.create("batch", settings, [dict(id="id", name="Original", kind="tex", source=str(source), root=None)])
    prior = store.paper("id")
    work = store.work(prior)
    work.mkdir(parents=True)
    if prior_verdict:
        state = {**old_terminal(prior_verdict), "selection_version": "v13-whole-paper-2", "recheck_saved_selection": True}
    else:
        state = old_result()
        state["reviews"].append({"stage": "compile", "passed": False, "issues": ["Outdated layout"], "attempt": 3})
        state["reviews"][-2]["review_version"] = CONTENT_REVIEW_VERSION
    config = {"configurable": {"thread_id": agent._thread_id(source, None, settings, agent._rules())}}
    with SqliteSaver.from_conn_string(str(work / "checkpoint.sqlite")) as saver:
        old = StateGraph(agent.PaperState)
        old.add_node("extract", lambda s: {"cursor": 1, "issues": [], "attempt": 3, "uncertainties": [], **state})
        old.add_edge(START, "extract")
        if pending:
            old.add_node("publish", lambda s: {"status": "completed"})
            old.add_edge("extract", "publish")
            old.add_edge("publish", END)
        else:
            old.add_edge("extract", END)
        old.compile(checkpointer=saver, interrupt_before=["publish"] if pending else []).invoke({"cursor": 0}, config)
    models = OfflineModels()
    monkeypatch.setattr(agent, "Models", lambda cfg: models)

    def build(work, *args):
        result_dir = work / "fresh"
        result_dir.mkdir()
        for name in agent._DOCUMENTS:
            (result_dir / name).write_text("fresh output")
        return result_dir

    monkeypatch.setattr(agent, "build_case", build)
    monkeypatch.setattr(agent, "_render_pages", lambda path: [(1, b"fixture")])
    events = []
    if not pending:
        store.finish(prior, state)
        assert stale_selection_failure(state)
        assert not stale_selection_failure({**state, "selection_version": SELECTION_VERSION})
        assert agent.run_paper(work, source, None, settings, lambda *args: events.append(args))["status"] == "failed"
        assert not events
        store.retry("id")
        queued = store.paper("id")
        assert store.work(queued) == work and queued["result"]["retry_legacy_review"]
        with store.db() as db:
            original = json.loads(db.execute("SELECT result FROM archived_reviews").fetchone()[0])
        assert original["reviews"] == state["reviews"]
    result = agent.run_paper(work, source, None, settings, lambda *args: events.append(args),
                             retry_legacy_review=not pending)
    assert result["status"] == "completed", result
    assert models.extracts == 0 and models.drafts == 1
    assert result["selection_version"] == SELECTION_VERSION
    assert len([data for _, data in events if data.get("tool") == "select_contributions"]) == 1
    assert [r["stage"] for r in result["reviews"]] == ["review", "compile", "visual"]


def test_targeted_source_read_reselects_and_writer_only_sees_global_candidates(paper):
    models, run, _, events = paper
    models.missing_application = True
    original = models.structured
    received = []

    def traced(schema, instructions, packet, image=None):
        if schema in (Draft, AdmissionReview):
            received.append([c["candidate_id"] for c in packet["candidates"]])
            assert all(c["decision"] == "admitted" and c["independent_question"] for c in packet["candidates"])
        return original(schema, instructions, packet, image)

    models.structured = traced
    result = run()
    assert result["status"] == "completed", result
    assert received == [["c1", "c2"], ["c1", "c2"], ["c1", "c2", "c3"], ["c1", "c2", "c3"]]
    assert len([data for _, data in events if data.get("tool") == "select_contributions"]) == 2



def test_shared_label_definition_and_repeated_secondary_roles_are_resolved():
    catalog = [{"id": "a", "text": "Definition and main assertion.", "section": "Main results"}]
    register = candidate_register([
        finding("current", "1.1", "For every X there exists a solution.", "a"),
        finding("definition", "1.1", "A regular solution is defined by a finite norm.", "a"),
        finding("setting", "2.3", "Every solution on a compact object is unique.", "a"),
        finding("definition", None, "The test invariant is the limit of the means.", "a"),
    ])
    choice = WholePaperSelection(selected=[
        pick("c1", "a", definition_ids=["c2", "c4"], variant_ids=["c1", "c2"]),
        pick("c3", "a", role="application", definition_ids=["c2"])],
        metadata_ids=["c2", "c3"], unresolved=[{"candidate_id": "c2", "source_reason": "Check this earlier label."}],
        exclusions=[{"candidate_ids": ["c1..c7", "c200"], "source_reason": "Other candidates serve the proof."}])
    rows = apply_selection(register, choice, catalog)
    assert rows[1]["needed_by"] == ["c1", "c3"] and rows[1]["context_for"] == ["c1"]
    assert rows[1]["kind"] == "definition" and rows[1]["decision"] == "admitted"
    assert rows[2]["local_kind"] == "setting" and rows[2]["kind"] == "application"
    assert rows[3]["kind"] == "definition" and rows[3]["needed_by"] == ["c1"]


def test_true_conflicting_positive_choices_repair_only_affected_source_rows():
    catalog = [{"id": f"p{i}", "text": f"Original statement {i}.", "section": "Main results"} for i in range(1, 5)]
    candidates = [finding("current", str(i), str(i), f"p{i}") for i in range(1, 5)]

    class Model:
        def structured(self, schema, _prompt, packet):
            if schema is WholePaperSelection:
                return schema(selected=[pick("c1", "p1", variant_ids=["c2"]), pick("c2", "p2"), pick("c3", "p3")],
                              exclusions=[{"candidate_ids": ["c4"], "source_reason": "An intermediate bound."}])
            assert schema is SelectionCorrection
            assert packet["affected_selected_ids"] == ["c1", "c2"]
            assert {p["id"] for p in packet["original_passages"]} == {"p1", "p2"}
            assert {row["candidate_id"] for row in packet["previous_bindings"]} == {"c1", "c2"}
            return schema(selected=[pick("c1", "p1", variant_ids=["c2"])])

    rows = select_contributions(catalog, candidates, Model())
    assert [row["decision"] for row in rows] == ["admitted", "excluded", "admitted", "excluded"]
    assert rows[1]["represented_by"] == "c1" and rows[3]["decision_reason"] == "An intermediate bound."

def test_retry_preserves_previous_interrupted_selection_and_work(tmp_path):
    db = Store(tmp_path)
    db.create("batch", {}, [dict(id="id", name="Original", kind="tex", source="input.tex", root=None)])
    prior = db.paper("id")
    work = db.work(prior)
    work.mkdir(parents=True)
    (work / "checkpoint.sqlite").touch()
    outcome = {"status": "failed", "reason": "SelectionCorrection timeout", "resumable": True,
               "candidates": [finding("definition", "1.1", "Fixed norm", "p1")]}
    db.finish(prior, outcome)
    db.retry("id")
    assert db.work(db.paper("id")) == work
    with db.db() as conn:
        earlier = json.loads(conn.execute("SELECT result FROM archived_reviews").fetchone()[0])
    assert earlier["reason"] == outcome["reason"] and earlier["candidates"] == outcome["candidates"]

def test_selected_definitions_and_setting_keep_original_role_without_consuming_core_slots():
    catalog = [{"id": "intro", "text": "The invariant is positive if its associated form is positive.",
                "section": "Introduction"}]
    register = candidate_register([
        finding("definition", "Definition 3", "Positive invariant means that its associated form is positive.", "intro"),
        finding("setting", None, "We work with compact spaces.", "intro"),
        finding("definition", "Definition 5", "A reference form.", "intro"),
        finding("setting", "Theorem A", "Every compact space has a solution.", "intro"),
    ])
    picks = [pick("c1", "intro"), pick("c2", "intro"), pick("c3", "absent"), pick("c4", "intro", role="current")]
    rows = apply_selection(register, WholePaperSelection(selected=picks), catalog)
    assert [(row["kind"], row["decision"]) for row in rows] == [
        ("definition", "admitted"), ("setting", "admitted"), ("definition", "unresolved"), ("current", "admitted")]
    assert "independent_question" not in rows[0] and rows[0]["selection_evidence_ids"] == ["intro"]
    assert rows[3]["independent_question"]
