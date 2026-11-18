"""Restore a version-gate fork only on the failed paper's own checked lineage."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from paper_extraction.saved_selection_recovery import saved_selection_recovery
from paper_extraction.selection import SELECTION_VERSION, candidate_register, exclusion_notes, selection_packet
from test_whole_selection import finding


def history():
    selected = [dict(finding(kind, "Theorem 1.1", "Compact solutions exist.", "p1"),
                     candidate_id=f"c{i}") for i, kind in enumerate(("current", "historical"), 1)]
    draft = dict(title="Current paper", authors=["A"], paragraph="Setting.", history=["Previous conclusion."],
                 shared_definitions=[], groups=[dict(theorem="Compact solutions exist.", definitions=[], paragraph="")],
                 references=[dict(key="source", text="A, Current paper.")],
                 evidence_links=[dict(block_id=k, evidence_ids=["p1"]) for k in ("paragraph", "history.0", "groups.0.theorem")])
    reviews = [dict(stage="review", passed=True, route="pass", attempt=1, issues=[]),
               dict(stage="compile", passed=False, attempt=1, issues=["Block needs a locator."])]
    prior = dict(cursor=4, attempt=1, selection_version=SELECTION_VERSION, recheck_saved_selection=True,
                 status="", draft=draft, candidates=selected, issues=reviews[-1]["issues"], reviews=reviews, target_ids=[])
    skipped = {**prior, "recheck_saved_selection": False}
    drift = {**skipped, "selection_version": "new-selection",
             "candidates": [*selected, dict(finding("current", "Lemma 2", "An iteration step.", "p2"), candidate_id="c3")]}
    failed = {**drift, "status": "failed", "attempt": 3,
              "draft": {**draft, "paragraph": "Expanded draft."},
              "reviews": [*reviews, dict(stage="review", route="draft", passed=False, attempt=3)]}
    return [(prior, "draft"), (skipped, "select"), (drift, "draft"), (failed, None)]


def recover(states, terminal, locations=("p1", "p2")):
    entries = []
    for i, (state, next_node) in enumerate(states):
        channels = dict(state)
        if next_node:
            channels["branch:to:" + next_node] = None
        cfg = {"configurable": {"checkpoint_id": i}}
        entries.append(SimpleNamespace(checkpoint={"channel_values": channels}, config=cfg,
                                       parent_config=entries[-1].config if entries else None))
    saver = SimpleNamespace(get_tuple=lambda cfg:entries[cfg["configurable"].get("checkpoint_id", -1)])
    index = SimpleNamespace(catalog=lambda:[{"id": loc} for loc in locations])
    return saved_selection_recovery(saver, {"configurable": {}}, terminal, index)


def test_only_source_bound_accepted_draft_on_unrelated_reselection_is_recovered():
    states = history()
    saved = recover(states, states[-1][0])
    assert saved["recovery_checkpoint_id"] == 0 and saved["selection_version"] == SELECTION_VERSION
    assert saved["draft"]["paragraph"] == "Setting." and len(saved["candidates"]) == 2
    assert saved["reviews"][-2]["passed"] is True and saved["issues"] == ["Block needs a locator."]


@pytest.mark.parametrize("case", ["new_evidence", "changed_draft", "no_accepted_review", "missing_source",
                                  "completed", "new_writer", "different_head", "wanted_select", "prior_policy"])
def test_recovery_rejects_unrelated_or_updated_lineages(case):
    states = deepcopy(history())
    terminal = deepcopy(states[-1][0])
    if case == "new_evidence":
        states.insert(-1, ({**states[2][0], "target_ids": ["p2"]}, "extract"))
    if case == "changed_draft":
        states[1][0]["draft"] = {**states[1][0]["draft"], "paragraph": "A changed question."}
    if case == "no_accepted_review":
        states[0][0]["reviews"][0]["passed"] = False
    if case == "completed":
        terminal["status"] = "completed"
    if case == "new_writer":
        terminal["writer_version"] = "field-repair-v1"
    if case == "different_head":
        terminal["draft"]["paragraph"] = "A draft from another branch."
    if case == "wanted_select":
        states[0][0]["recheck_saved_selection"] = False
    if case == "prior_policy":
        for state, _ in states[:2]:
            state["selection_version"] = "v13-whole-paper-2"
    locations = ["p2"] if case == "missing_source" else ["p1", "p2"]
    assert recover(states, terminal, locations) is None


def test_under_budget_selector_receives_candidate_sources_without_prior_exclusion_notes():
    catalogue = [
        {"id": "p1", "text": "The paper is organized as follows. We iterate a local construction to prove the main theorem."},
        {"id": "p2", "text": "Proposition 3.1. A perturbation on a small domain solves the auxiliary system."},
        {"id": "p3", "text": "In the proof of the main theorem, iterating Proposition 3.1 yields a global solution."},
        {"id": "p4", "text": "Theorem 1.1. Global solutions exist on all closed objects."},
    ]
    candidates = [finding("current", "Proposition 3.1", "A local perturbation exists.", "p2"),
                  finding("current", "Theorem 1.1", "A global solution exists.", "p4")]
    rows = candidate_register(candidates)
    packet = selection_packet(catalogue, rows)
    assert {"p1", "p2", "p4"} <= {p["id"] for p in packet["passages"]}
    assert "original_uses" not in packet and "decision_reason" not in packet["candidate_register"][0]
    rows[0].update(decision="excluded", decision_reason="The source iterates this local step to establish Theorem 1.1.")
    notes = exclusion_notes(rows)
    assert len(notes) == 1 and notes[0]["reason"] == rows[0]["decision_reason"]
    assert notes[0]["observations"][0]["label"] == "Proposition 3.1"

