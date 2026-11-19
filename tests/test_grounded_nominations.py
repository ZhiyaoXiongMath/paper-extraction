"""Repeated numbers, distinct clauses and the handoff from review to final TeX."""
import shutil

import pytest

from paper_extraction import agent
from paper_extraction.admission import constrain_current_results
from paper_extraction.lean_workflow import AddedCores, EvidenceReview
from paper_extraction.nomination_ledger import delivered_nominations, register_nominations, settle_nominations
from paper_extraction.profile_inputs import new_batch_profile
from paper_extraction.review_response import EvidenceReviewForm, editorial_response
from paper_extraction.segment_evidence import SegmentLeads
from paper_extraction.selection import LeanChoice, NominationChoice
from test_lean_workflow import APPLICATION, Replay
from test_nomination_delivery import formal_source


def proposal(loc, quote, loss="The independent equality case is absent."):
    return {"scope": "current", "source_heading": "Theorem 17", "evidence_id": loc,
            "source_excerpt": quote, "omission_loss": loss}


def test_source_and_clause_identity_with_duplicates_and_reordered_candidates():
    text = "Theorem 17. Every A is B. Under C, equality holds."
    records = {key: {"id": key, "page": i, "text": text} for i, key in enumerate(("r1", "r2"), 1)}
    a, b, c = proposal("r1", "Every A is B."), proposal("r1", "Under C, equality holds."), proposal("r2", "Every A is B.")
    ledger, keys, rows = register_nominations("snapshot", [a, b, a, c], records, {}, [], {})
    assert len(keys) == len(rows) == 3 and all(ledger[k]["source_verified"] for k in keys)
    assert len({ledger[k]["candidate_id"] for k in keys}) == 3
    for row in rows:
        row.update(decision="excluded", decision_reason="Only a routine substitution is supported.")
    ledger = settle_nominations(ledger, keys, rows)
    replay, again, repeated = register_nominations("snapshot", [a], records, ledger, rows[::-1], {})
    assert again == [keys[0]] and replay[keys[0]]["reused"] and len(repeated) == 3
    other_scope = {**a, "omission_loss": "A stronger conclusion for a separate class was omitted.",
                   "candidate_id": ledger[keys[0]]["candidate_id"]}
    reopened, new, with_new_scope = register_nominations("snapshot", [other_scope], records, ledger, rows, {})
    assert new != again and not reopened[new[0]]["reused"]
    assert reopened[new[0]]["candidate_id"] != ledger[keys[0]]["candidate_id"]
    selected = [{**r, "decision": "admitted"} for r in with_new_scope]
    constrain_current_results(selected, records)
    assert all(r["decision"] == "admitted" for r in selected)
    updated_records = {**records, "r1": {**records["r1"], "text": text + " A new range is proved."}}
    fresh, updated, _ = register_nominations("snapshot", [a], updated_records, ledger, rows, {})
    assert updated != again and not fresh[updated[0]]["reused"]


def test_quote_in_neighboring_theorem_and_unknown_boundaries_require_source_read():
    records = {"p": {"id": "p", "page": 1, "text": "Theorem 17. One assertion.\nTheorem 18. Other assertion."}}
    wrong = proposal("p", "Other assertion.")
    ledger, keys, _ = register_nominations("snapshot", [wrong], records, {}, [], {})
    assert not ledger[keys[0]]["source_verified"]
    with_other_kind = {**wrong, "source_heading": "Corollary 17"}
    ledger, keys, _ = register_nominations("snapshot", [with_other_kind], records, {}, [], {})
    assert not ledger[keys[0]]["source_verified"]


def test_formal_statement_spanning_tex_records_keeps_both_source_bindings():
    records = {
        "first": {"id": "first", "line": 40, "end_line": 42, "file": "part.tex", "text": "header\nfirst\nsecond",
                  "source_statements": [{"id": "file.s88", "kind": "Theorem", "number": "17", "label": "",
                                         "unnumbered": False, "line": 41, "end_line": 44}]},
        "last": {"id": "last", "line": 43, "end_line": 45, "file": "part.tex",
                 "text": "Every A with B is C.\nend\nThe unrelated case."},
    }
    row = {**proposal("last", "Every A with B is C."), "source_heading": "Chapter II, Theorem 17"}
    ledger, keys, candidates = register_nominations("snap", [row], records, {}, [], {})
    assert ledger[keys[0]]["source_verified"] and ledger[keys[0]]["source_statement_id"] == "file.s88"
    assert candidates[0]["evidence_ids"] == ["last", "first"]
    after = {**row, "source_excerpt": "The unrelated case."}
    ledger, keys, _ = register_nominations("snap", [after], records, {}, [], {})
    assert not ledger[keys[0]]["source_verified"]


def test_coverage_rechecked_against_delivered_text_even_for_same_candidate():
    entry = {"candidate_id": "c9", "decision": "represented", "represented_by": "c5",
             "covered_excerpt": "With sign allowed, equality holds."}
    draft = {"core_bindings": {"groups.0.theorem": "c5"}, "groups": [{"theorem": entry["covered_excerpt"]}]}
    assert delivered_nominations({"k": entry}, draft)["k"]["printed_field"] == "groups.0.theorem"
    draft["groups"][0]["theorem"] = "Equality holds for positive inputs."
    assert delivered_nominations({"k": entry}, draft)["k"]["printed_field"] is None
    assert delivered_nominations({"k": entry}, {"core_bindings": draft["core_bindings"]})["k"]["printed_field"] is None


@pytest.mark.skipif(not shutil.which("pdflatex"), reason="requires local compiler")
@pytest.mark.parametrize("decision", ["admit", "unresolved", "bad_coverage", "writer_unavailable", "replacement_unavailable"])
def test_verified_typed_nomination_runs_scoped_selection_without_another_extraction(tmp_path, monkeypatch, decision):
    source = formal_source(tmp_path)
    model = Replay("omitted")
    original = model.structured
    packets = []

    def reply(schema, instructions, packet, image=None):
        if schema is EvidenceReview and model.review_count == 0:
            model.review_count += 1
            return editorial_response(EvidenceReviewForm, {"verdict": {"route": "extract", "issues": [],
                "nominations": [{**proposal(model.application, APPLICATION), "source_heading": "Corollary 1.2"}]}},
                packet, EvidenceReview)
        if schema is NominationChoice:
            packets.append(packet)
            if decision == "unresolved":
                return schema(selected=[], unresolved=[{"candidate_id": "c2", "source_reason": "The application scope remains open."}])
            if decision == "bad_coverage":
                return schema(selected=[], covered=[{"candidate_id": "c2", "printed_candidate_id": "c1",
                              "printed_excerpt": "An assertion that was never printed.", "reason": "Same scope."}])
            previous = original(LeanChoice, instructions, packet, image)
            answer = previous.model_dump()
            if decision == "replacement_unavailable":
                answer["supersessions"] = [{"retained_id": "c1", "selected_id": "c2",
                                            "reason": "Newly selected stronger statement subsumes the earlier one."}]
            return schema.model_validate(answer)
        if decision == "replacement_unavailable" and schema.__name__ in {"AuthoritativeDraft", "AddedCores"}:
            raise ConnectionError("offline simulated replacement failure")
        if schema is AddedCores and decision == "writer_unavailable":
            raise ConnectionError("offline simulated writer failure")
        return original(schema, instructions, packet, image)

    model.structured = reply
    monkeypatch.setattr(agent, "Models", lambda _: model)
    result = agent.run_paper(tmp_path / "work", source, source.parent, new_batch_profile(), lambda *_: None)
    assert result["status"] == "completed", result.get("reason")
    assert len(packets) == 1 and SegmentLeads not in model.calls
    assert packets[0]["retained_cores"][0]["printed_field"] == "groups.0.theorem"
    outcome, = result["delivery"]["nomination_outcomes"]
    if decision == "admit":
        assert result["delivery"]["status"] == "reviewed"
        assert outcome["decision"] == "admitted" and outcome["printed_field"] == "groups.1.theorem"
    else:
        assert result["delivery"]["status"] == "delivered" and outcome["printed_field"] is None
        assert outcome["publication_state"] == ("unprinted" if decision.endswith("unavailable") else "unresolved")
        assert "Corollary 1.2" in result["delivery"]["note"]
