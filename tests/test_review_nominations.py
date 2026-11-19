"""Offline wire regressions for nomination + local-edit verdicts and one-row repair."""
import json

import pytest

from paper_extraction.lean_workflow import EvidenceReview
from paper_extraction.models import AdmissionReview
from test_partial_response import defect, wire


NOMINATION = {
    "scope": "current", "source_heading": "Theorem 3.6", "evidence_id": "t1-17",
    "source_excerpt": "Every complete space satisfies this bound.",
    "omission_loss": "The independent rigidity case is absent from Theorem 3.2.",
}
EXISTING = defect("groups.1.theorem", "t1-14")
PACKET = {"draft": {"groups": [{"theorem": "Base."}, {"theorem": "x>0"}]},
          "passages": [{"id": "t1-14", "text": "First case with equality."}],
          "source_locations": [{"id": "t1-14"}, {"id": "t1-17"}],
          "checked_source_images": [{"source_id": "t1-17", "transcription": "Theorem 3.6 in original print"}]}


def setup(*replies):
    model, calls = wire(*replies)
    model.targeted_revision = True
    model.review_source_index = {"t1-17": {"id": "t1-17", "text": NOMINATION["source_excerpt"]}}
    return model, calls


@pytest.mark.parametrize("schema", [AdmissionReview, EvidenceReview])
def test_nomination_and_independent_edit_reach_their_original_routes(schema):
    reply = {"route": "extract", "issues": [EXISTING], "nominations": [NOMINATION]}
    model, calls = setup(reply)
    decision = model.structured(schema, "Check", PACKET)
    verdict = decision.verdict if schema is EvidenceReview else decision
    edit, nomination = verdict.issues
    assert len(calls) == 1 and edit.model_dump() == EXISTING
    assert verdict.evidence_ids == ["t1-17"]
    assert nomination.operation == "recover" and nomination.draft_field == "groups"
    assert nomination.current_excerpt == nomination.replacement_text == ""
    assert "Theorem 3.6" in nomination.required_change


@pytest.mark.parametrize("schema", [AdmissionReview, EvidenceReview])
def test_local_missing_result_repair_keeps_verified_edits_and_reads_registered_evidence(schema):
    good = {**NOMINATION, "scope": "historical", "source_heading": "Earlier comparison", "evidence_id": "t1-14"}
    original = {"route": "extract", "issues": [EXISTING], "nominations": [good, {**NOMINATION, "omission_loss": ""}]}
    model, calls = setup({"verdict": original} if schema is EvidenceReview else original, {"nominations": [NOMINATION]})
    decision = model.structured(schema, "Check", PACKET)
    verdict = decision.verdict if schema is EvidenceReview else decision
    assert [i.draft_field for i in verdict.issues] == ["groups.1.theorem", "history", "groups"]
    assert verdict.issues[0].model_dump() == EXISTING
    correction = json.loads(calls[1]["messages"][1]["content"])
    assert [p["id"] for p in correction["passages"]] == ["t1-17"]
    assert correction["source_images"] == PACKET["checked_source_images"]
    assert "First case with equality." not in str(correction)


def test_repair_legacy_missing_result_without_a_draft_block():
    bad = {**EXISTING, "draft_field": "groups", "kind": "missing_required_content", "operation": "recover",
           "current_excerpt": "", "evidence_id": "t1-17", "replacement_text": "Do not draft this yet."}
    model, calls = setup({"route": "extract", "issues": [EXISTING, bad], "evidence_ids": ["t1-17"]},
                         {"issues": [{**bad, "replacement_text": ""}]})
    result = model.structured(EvidenceReview, "Check", PACKET)
    assert result.verdict.issues[0].model_dump() == EXISTING
    correction = json.loads(calls[1]["messages"][1]["content"])
    assert "rows_in_order" in correction and correction["rows_in_order"][0]["draft_text"] == ""


def test_nomination_cannot_pass_or_escape_registered_sources_and_budget():
    wrong = {"route": "pass", "issues": [], "nominations": [NOMINATION]}
    wrong_source = {"route": "extract", "issues": [], "nominations": [{**NOMINATION, "evidence_id": "p999"}]}
    valid = {"route": "extract", "issues": [], "nominations": [NOMINATION]}
    model, calls = setup(wrong, wrong_source, valid)
    result = model.structured(AdmissionReview, "Check", PACKET)
    assert len(calls) == 3 and result.evidence_ids == ["t1-17"]
    assert "p999" not in result.evidence_ids


def test_provisional_nomination_does_not_suppress_requested_evidence():
    model, calls = setup({"verdict": {"route": "extract", "issues": [], "nominations": [NOMINATION]},
                          "source_checks": [{"source_id": "t1-17", "query": "locate the numbered heading",
                                             "question": "What is the heading?"}]})
    result = model.structured(EvidenceReview, "Check", PACKET)
    assert len(calls) == 1 and result.verdict is None and result.source_checks[0].source_id == "t1-17"
