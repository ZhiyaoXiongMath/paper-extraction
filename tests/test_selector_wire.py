"""Exercise the selection, binding correction, and over-budget requests over an offline HTTP wire."""
from hashlib import sha256
from importlib.resources import files
import json

from paper_extraction.scope_reconciliation import SCOPE_FIELDS, scope_failure_reason
from paper_extraction.selection import CORRECTION_FIELDS, SELECTION_FIELDS, select_contributions
from test_agent_json import adapter
from test_whole_selection import finding, pick


def test_v13_rule_is_the_complete_wire_prefix_at_each_selection_branch():
    source = files("paper_extraction").joinpath("profile_assets", "intro-rules-v13.txt").read_bytes()
    assert sha256(source).hexdigest() == "aa99ff2fc233fa4e34c32cf85dd6a504cd306349120db99899cedf6bffb2d7e7"
    rules = source.decode("utf-8")
    catalog = [{"id": f"p{i}", "label": f"Original page {i}", "text": f"Source result {i}."} for i in range(1, 6)]
    candidates = [dict(finding("current", f"Theorem {i}", f"Original statement {i}.", f"p{i}"),
                       decision_reason="LOCAL_VOTE_SENTINEL", significance="LOCAL_VOTE_SENTINEL") for i in range(1, 6)]
    proposals = [{**pick(f"c{i}", f"p{i}"), "source_reason": "PRIOR_SELECTOR_VOTE_SENTINEL",
                  "independent_question": "PRIOR_QUESTION_SENTINEL"} for i in range(1, 6)]
    proposals[0]["evidence_ids"] = ["unregistered"]
    fixed = {**proposals[0], "evidence_ids": ["p1"]}
    final = [{"candidate_id": f"c{i}", "outcome": "admitted", "source_id": f"p{i}",
              "source_reason": f"The original records result {i}.", "independent_question": f"Question {i}?"}
             for i in range(1, 6)]
    final[-1].update(outcome="linked_clause", retained_id="c1")
    replies = [{"selected": proposals}, {"selected": [fixed]}, {"assessments": final}]
    wire = []
    model = adapter([(json.dumps(doc), "stop", 200) for doc in replies], wire)
    result = select_contributions(catalog, candidates, model)
    assert len(wire) == 3
    assert [r["scope_group_id"] for r in result] == ["c1", "c2", "c3", "c4", "c1"]
    assert scope_failure_reason(result) == ""
    for request, fields in zip(wire, (SELECTION_FIELDS, CORRECTION_FIELDS, SCOPE_FIELDS)):
        system = request["messages"][0]
        assert system["role"] == "system"
        assert system["content"].startswith(rules + "\n" + fields + "\n")
        assert system["content"].count(rules) == 1
        appended = system["content"][len(rules):]
        for old in ("positive ORIGINAL evidence", "significance beyond", "separately motivated",
                    "occurring outside", "Keep distinct substantial assertions", "author's use"):
            assert old not in appended
        user_data = json.dumps(request["messages"][1:])
        assert "LOCAL_VOTE_SENTINEL" not in user_data
        assert "PRIOR_SELECTOR_VOTE_SENTINEL" not in user_data and "PRIOR_QUESTION_SENTINEL" not in user_data
    payload = json.loads(wire[1]["messages"][1]["content"])
    assert payload["affected_selected_ids"] == ["c1"]
    assert payload["previous_bindings"][0]["evidence_ids"] == ["unregistered"]
    assert "previous_selection" not in payload
    scope = json.loads(wire[2]["messages"][1]["content"])
    assert scope["proposed_current_group_count"] == 5 and scope["standard_current_group_max"] == 4


def test_under_budget_b4_requires_an_explicit_unresolved_central_id():
    catalog = [{"id": "p1", "text": "The source discusses an unsettled comparison."}]
    candidate = finding("current", "Theorem 1", "A comparison is conjectured.", "p1")
    reply = {"selected": [], "unresolved": [{"candidate_id": "c1", "source_reason": "Admission remains open."}]}
    for central in ([], ["c1"]):
        model = adapter([(json.dumps({**reply, "central_open_ids": central}), "stop", 50)], [])
        rows = select_contributions(catalog, [candidate], model)
        assert rows[0]["decision"] == "unresolved"
        assert bool(scope_failure_reason(rows)) is bool(central)


def test_format_retry_carries_the_full_rule_and_only_the_json_field_error():
    source = files("paper_extraction").joinpath("profile_assets", "intro-rules-v13.txt").read_bytes().decode("utf-8")
    invalid = {"selected": [{"candidate_id": "c1", "independent_question": "Q?", "evidence_ids": [],
                             "source_reason": "DISCARDED_VALIDATION_VOTE_SENTINEL"}]}
    valid = {"selected": [pick("c1", "p1")]}
    wire = []
    model = adapter([(json.dumps(doc), "stop", 60) for doc in (invalid, valid)], wire)
    rows = select_contributions([{"id": "p1", "text": "A sourced result."}],
                                [finding("current", "Theorem 1", "Result", "p1")], model)
    assert rows[0]["decision"] == "admitted" and len(wire) == 2
    assert wire[0]["messages"][0] == wire[1]["messages"][0]
    assert wire[1]["messages"][0]["content"].startswith(source + "\n" + SELECTION_FIELDS)
    retry = wire[1]["messages"][-1]["content"]
    assert "selected.0.evidence_ids" in retry and "previous_complete_json" not in retry
    assert "DISCARDED_VALIDATION_VOTE_SENTINEL" not in json.dumps(wire[1])

