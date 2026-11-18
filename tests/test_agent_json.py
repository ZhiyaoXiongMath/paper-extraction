"""Real ChatOpenAI JSON wire format with an offline HTTP transport."""
import json

import httpx
import pytest
from langchain_openai import ChatOpenAI

from paper_extraction.models import AdmissionReview, Findings, Models, Review


def candidate(kind="current"):
    return {"kind": kind, "claim": "Private mathematical passage", "assumptions": "", "significance": "central",
            "source_number": "1.1", "dependencies": [], "decision": "admitted", "decision_reason": "central",
            "evidence_ids": ["p3-1"]}


def adapter(replies, requests):
    def server(req):
        requests.append(json.loads(req.content))
        content, finish, output = replies.pop(0)
        return httpx.Response(200, json={"id": "fixture", "object": "chat.completion", "created": 0,
            "model": "mock", "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                         "finish_reason": finish}],
            "usage": {"prompt_tokens": 128, "completion_tokens": output, "total_tokens": output+128}})
    instance = Models.__new__(Models)
    instance.provider = "deepseek"
    instance.text = ChatOpenAI(api_key="offline-fixture", model="mock", base_url="http://offline.invalid",
                               http_client=httpx.Client(transport=httpx.MockTransport(server)),
                               extra_body={"thinking": {"type": "disabled"}})
    return instance


def test_length_increases_provider_budget_and_discards_partial_text():
    complete = {"candidates": [candidate()], "uncertainties": []}
    requests = []
    model = adapter([('{"candidates":[{"claim":"TRUNCATED_MARKER"', "length", 8192),
                     (json.dumps(complete), "stop", 256)], requests)
    result = model.structured(Findings, "Extract claims", {"passages": ["first passage"]})
    assert len(result.candidates) == 1
    assert requests[1]["max_tokens"] == 16384 and requests[1]["thinking"]["type"] == "disabled"
    assert "TRUNCATED_MARKER" not in str(requests[1]["messages"])
    assert requests[1]["response_format"] == {"type": "json_object"}


def test_review_schema_feedback_allows_withdrawing_a_reminder():
    legacy = {"route": "draft", "issues": ["The draft already says noncompact; keep that qualifier."],
              "evidence_ids": []}
    correct = {"route": "pass", "issues": [], "evidence_ids": []}
    requests = []
    model = adapter([(json.dumps(reply), "stop", 128) for reply in (legacy, correct)], requests)
    review = model.structured(AdmissionReview, "Review the current introduction.", {
        "draft": {"groups": [{"theorem": "Complete noncompact manifolds with nonnegative Ricci curvature have infinite volume."}]},
        "passages": [{"id": "p9-1", "text": "Complete noncompact manifolds with nonnegative Ricci curvature have infinite volume."}],
    })
    assert review.route == "pass" and review.repair_issues() == []
    retry = json.loads(requests[1]["messages"][-1]["content"])
    assert "issues.0" in retry["correction"] and "schema" in retry["return"]


def test_model_corrects_a_confirmed_claim_and_retains_a_real_edit():
    real = {"kind": "incorrect_claim", "draft_field": "groups.0.theorem", "current_excerpt": "every metric",
            "evidence_id": "p3-1", "source_excerpt": "every complete metric",
            "required_change": "Restore the hypothesis of completeness.", "operation": "replace",
            "replacement_text": "every complete metric"}
    no_change = {**real, "draft_field": "history.0", "required_change": "Equivalent; no defect.",
                 "replacement_text": "every closed metric"}
    wrong = {"route": "draft", "issues": [no_change, real], "evidence_ids": []}
    correction = {**wrong, "issues": [real]}
    requests = []
    result = adapter([(json.dumps(x), "stop", 200) for x in (wrong, correction)], requests).structured(
        AdmissionReview, "Review these assertions.", {})
    assert len(requests) == 2 and [i.draft_field for i in result.issues] == ["groups.0.theorem"]
    assert "every complete metric" in result.repair_issues()[0]
    assert "inconsistent with an edit operation" in requests[1]["messages"][-1]["content"]


def test_failed_visual_confirmation_is_reasked_instead_of_assumed_to_pass():
    failed = {"passed": False, "issues": ["Page 1: No visible defect."]}
    real = {"passed": False, "issues": ["The last symbol on line 3 is clipped at the right margin."]}
    requests = []
    model = adapter([(json.dumps(x), "stop", 80) for x in (failed, real)], requests)
    model.vision = model.text
    verdict = model.structured(Review, "Inspect this fixture page.", {"page": 1}, image=b"fixture")
    assert len(requests) == 2 and not verdict.passed and verdict.issues == real["issues"]


def test_repair_invalid_rows_preserves_unaffected_findings():
    good = candidate()
    invalid = {**candidate("prior"), "claim": "Prior theorem", "source_number": "A", "evidence_ids": ["p1-1"]}
    original = {"candidates": [good, invalid], "uncertainties": ["Some history is unresolved"]}
    repaired = {"candidates": [{**invalid, "kind": "historical"}], "uncertainties": []}
    requests = []
    result = adapter([(json.dumps(original), "stop", 500), (json.dumps(repaired), "stop", 200)], requests).structured(
        Findings, "Extract claims", {"passages": ["original evidence"]})
    correction = json.loads(requests[1]["messages"][-1]["content"])
    assert [row["original_index"] for row in correction["rows_to_fix_in_order"]] == [1]
    assert "historical" in correction["correction"] and "candidates.1.kind" in correction["correction"]
    assert result.model_dump() == {"candidates": [good, repaired["candidates"][0]], "uncertainties": original["uncertainties"]}


def test_final_diagnostic_reports_schema_expectation_without_paper_text():
    invalid = json.dumps({"candidates": [candidate("prior")], "uncertainties": []})
    requests = []
    with pytest.raises(ValueError) as err:
        adapter([(invalid, "stop", 200)] * 3, requests).structured(Findings, "Extract", {})
    reason = str(err.value)
    assert "Findings" in reason and "candidates.0.kind" in reason and "literal_error" in reason
    assert "output_tokens=200" in reason and "Private mathematical passage" not in reason

