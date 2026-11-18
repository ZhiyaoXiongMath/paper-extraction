"""An explicit retry can re-review an intact, already selected final draft."""
from copy import deepcopy

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from paper_extraction import agent
from paper_extraction.editorial import EDIT_CONTRACT_VERSION
from paper_extraction.models import AdmissionReview, DraftPatch, Review
from paper_extraction.selection import SELECTION_VERSION
from paper_extraction.store import CONTENT_REVIEW_VERSION, Store, legacy_confirmation_failure
from test_agent import OfflineModels


def old_defect(field, change):
    return {"kind": "incorrect_claim", "draft_field": field, "current_excerpt": "compact objects",
            "source_excerpt": "compact objects", "evidence_id": "t1-1", "required_change": change}


def encode_issue(d):
    return (f"{d['draft_field']}: {d['required_change']} (draft: {d['current_excerpt']}; "
            f"original {d['evidence_id']}: {d['source_excerpt']})")


def old_terminal(mode):
    history = r"Earlier work covers compact objects. \cite[p.~1]{source}"
    theorem = r"The source claim applies to compact objects."
    if mode != "citation":
        theorem += r" \cite[p.~1]{source}"
    draft = {"title": "Sample", "authors": ["Author"], "paragraph": "Setting.", "history": [history],
             "shared_definitions": [], "groups": [{"theorem": theorem, "paragraph": "", "definitions": []}],
             "references": [{"key": "source", "text": "Author, Sample."}],
             "evidence_links": [{"block_id": k, "evidence_ids": ["t1-1"]}
                                for k in ("paragraph", "history.0", "groups.0.theorem")]}
    rows = [{"kind": k, "claim": "source claim", "assumptions": "compact", "decision": "admitted",
             "decision_reason": "source-supported", "significance": "central", "source_number": k,
             "dependencies": [], "evidence_ids": ["t1-1"], "selection_evidence_ids": ["t1-1"],
             "candidate_id": f"c{i}"} for i, k in enumerate(("current", "historical"), 1)]
    if mode in {"confirmation", "confirmation_layout"}:
        defects = [old_defect("history.0", "The two assertions are equivalent and the draft copies the source; "
                             "no defect. (Retained only to record that this was checked.)"),
                   old_defect("groups.0.theorem", 'Equivalent; the phrase "in this case" and "reports" '
                              'do not alter the claim. No defect.')]
        last = {"stage": "review", "route": "draft", "passed": False, "attempt": 3,
                "review_version": CONTENT_REVIEW_VERSION, "defects": defects, "evidence_ids": [],
                "issues": [encode_issue(d) for d in defects]}
    elif mode == "patch":
        old = old_defect("groups.0.theorem", "Restrict the result to connected compact objects.")
        old.update(operation="replace", replacement_text="connected compact objects", source_excerpt="connected compact objects")
        review = AdmissionReview(route="draft", issues=[old], evidence_ids=[])
        last = {"stage": "review", "route": "draft", "passed": False, "attempt": 3, "defects": [old],
                "review_version": CONTENT_REVIEW_VERSION, "issues": review.repair_issues(), "evidence_ids": []}
    elif mode == "layout":
        last = {"stage": "compile", "passed": False, "attempt": 3, "issues": ["v13 block height exceeded"],
                "repair_targets": [{"draft_field": "paragraph", "current_text": draft["paragraph"],
                                    "diagnostics": ["introparagraph exceeds its v13 height"]}]}
    else:
        last = {"stage": "compile", "passed": False, "attempt": 3, "issues": [
            r"group 1 theorem needs a citation identifying its source location, e.g. \cite[p. 1]{source}."]}
    return {"status": "failed", "selection_version": SELECTION_VERSION, "draft": draft, "candidates": rows,
            "reason": "; ".join(last["issues"]), "reviews": [last]}


def test_only_affirmative_legacy_verdict_can_reuse_unchanged_draft():
    old = old_terminal("confirmation")
    assert legacy_confirmation_failure(old)
    for change in ["No defect. Add the missing compactness assumption.",
                   "The approximation is invalid. No defect.", "Recover a missing application."]:
        mixed = deepcopy(old)
        item = mixed["reviews"][0]["defects"][1]
        item["required_change"] = change
        mixed["reviews"][0]["issues"][1] = encode_issue(item)
        assert not legacy_confirmation_failure(mixed)
    mismatched = deepcopy(old)
    mismatched["reviews"][0]["issues"].append("A citation really must be corrected.")
    assert not legacy_confirmation_failure(mismatched)


@pytest.mark.parametrize("mode", ["confirmation", "citation", "patch", "layout", "confirmation_layout"])
def test_explicit_retry_reuses_final_draft_and_actual_selection_version(tmp_path, monkeypatch, mode):
    store = Store(tmp_path / "home")
    root = tmp_path / "source"
    root.mkdir()
    source = root / "original.tex"
    source.write_text("The source claim applies to connected compact objects.", encoding="utf-8")
    settings = {"provider": "offline", "base_url": "http://localhost", "text_model": "t", "vision_model": "v"}
    store.create("batch", settings, [dict(id="p", name="Source", kind="tex", source=str(source), root=str(root))])
    before = store.paper("p")
    work = store.work(before)
    work.mkdir(parents=True)
    previous = old_terminal(mode)
    store.finish(before, previous)
    cfg = {"configurable": {"thread_id": agent._thread_id(source, root, settings, agent._rules())}}
    with SqliteSaver.from_conn_string(str(work / "checkpoint.sqlite")) as saver:
        old = StateGraph(agent.PaperState)
        old.add_node("old", lambda _: {"cursor": 1, "attempt": 3, "target_ids": [],
                                       "issues": previous["reviews"][-1]["issues"], **previous})
        old.add_edge(START, "old")
        old.add_edge("old", END)
        graph = old.compile(checkpointer=saver)
        graph.invoke({"cursor": 0}, cfg)
        checkpoint = graph.get_state(cfg).config

    seen = []
    class OnlyReview(OfflineModels):
        def structured(self, schema, instructions, payload, image=None):
            if schema is DraftPatch and mode in {"patch", "layout", "confirmation_layout"}:
                [target] = payload["repair_targets"]
                seen.append(schema)
                text = (target["current_text"].replace("compact", "connected compact")
                        if mode == "patch" else "Scope.")
                return DraftPatch(replacements=[dict(block_id=target["block_id"],
                                                     text=text, evidence_ids=target["required_evidence_ids"])])
            assert schema in (AdmissionReview, Review), f"Unexpected {schema}: attempted to regenerate the draft"
            seen.append(schema)
            return super().structured(schema, instructions, payload, image)
    monkeypatch.setattr(agent, "Models", lambda _: OnlyReview())
    monkeypatch.setattr(agent, "_render_pages", lambda _: [(1, b"page")])
    def build(directory, entry, source_root, draft):
        assert r"\cite" in draft["groups"][0]["theorem"]
        if mode == "confirmation_layout" and draft["paragraph"] == "Setting.":
            raise agent.BuildFailure("v13 block height exceeded",
                                     [{"draft_field": "paragraph", "current_text": draft["paragraph"],
                                       "diagnostics": ["introparagraph exceeds its v13 height"]}])
        dest = directory / "published-fixture"
        dest.mkdir()
        for filename in agent._DOCUMENTS:
            (dest / filename).write_text("publication fixture")
        return dest
    monkeypatch.setattr(agent, "build_case", build)
    events = []
    def notify(name, data):
        events.append(name)
    assert agent.run_paper(work, source, root, settings, notify)["status"] == "failed"
    assert not seen
    store.retry("p")
    queued = store.paper("p")
    assert store.work(queued) == work and queued["result"]["retry_legacy_review"]
    result = agent.run_paper(work, source, root, settings, notify, retry_legacy_review=True)
    assert result["status"] == "completed", result
    before_review = ([AdmissionReview, DraftPatch] if mode == "confirmation_layout" else
                     [DraftPatch] if mode in {"patch", "layout"} else [])
    assert seen == before_review + [AdmissionReview, Review]
    assert "compile" in events and "publish" in events
    assert result["selection_version"] == previous["selection_version"]
    assert result["candidates"] == previous["candidates"]
    assert result["reviews"][0]["edit_contract_version"] == EDIT_CONTRACT_VERSION
    if mode == "confirmation":
        assert result["draft"] == previous["draft"]
    elif mode == "patch":
        changed = deepcopy(previous["draft"])
        changed["groups"][0]["theorem"] = changed["groups"][0]["theorem"].replace("compact", "connected compact")
        assert changed == result["draft"]
    elif mode in {"layout", "confirmation_layout"}:
        assert result["draft"] == {**previous["draft"], "paragraph": "Scope."}
    else:
        assert result["draft"]["groups"][0]["theorem"].startswith(previous["draft"]["groups"][0]["theorem"])
    with SqliteSaver.from_conn_string(str(work / "checkpoint.sqlite")) as saver:
        assert saver.get(checkpoint)["channel_values"]["draft"] == previous["draft"]

