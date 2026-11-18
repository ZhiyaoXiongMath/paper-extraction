"""Real ChatOpenAI JSON wire format with an offline HTTP transport."""
import json

import httpx
import pytest
from langchain_openai import ChatOpenAI

from paper_extraction.models import Findings, Models


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

