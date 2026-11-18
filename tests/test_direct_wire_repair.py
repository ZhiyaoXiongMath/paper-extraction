"""Exercise lossless reply decoding through the adapter with real source validation."""
import json
from copy import deepcopy

import httpx
import pytest
from langchain_openai import ChatOpenAI

from paper_extraction import models
from paper_extraction.lean_workflow import DirectResult
from paper_extraction.profile_inputs import new_batch_profile
from paper_extraction.usage import usage_session
from test_lean_workflow import field, manuscript


@pytest.mark.parametrize("distortion,expected_calls", [
    ("fenced-bindings", 1), ("mapped-evidence", 1), ("identical-links", 1),
    ("conflicting-links", 2), ("unknown-location", 2), ("wrong-id", 2), ("archive-fence", 2),
])
def test_direct_json_distortions(tmp_path, monkeypatch, distortion, expected_calls):
    private = "source-private-math"
    paper = {"passages": [{"id": "p1-1", "text": private}, {"id": "p2-1", "text": "second passage"}]}
    row = field(private, "c1", "p1-1")
    valid = {"cores": [{"candidate_id": "c1", "kind": "current", "source_number": "Theorem 1",
                        "claim": private, "independent_question": "Which case?", "source_reason": "The main case.",
                        "evidence_ids": ["p1-1"]}], "draft": manuscript([row])}
    modified = deepcopy(valid)
    draft = modified["draft"]
    if distortion == "fenced-bindings":
        draft["core_bindings"] = [{"block_id": "groups.0.theorem", "candidate_id": "c1"}]
    elif distortion == "mapped-evidence":
        draft["evidence_links"] = {r["block_id"]: r["evidence_ids"] for r in draft["evidence_links"]}
    elif distortion.endswith("links"):
        extra = deepcopy(draft["evidence_links"][0])
        if distortion == "conflicting-links":
            extra["evidence_ids"] = ["p2-1"]
        draft["evidence_links"].append(extra)
    elif distortion == "unknown-location":
        modified["cores"][0]["evidence_ids"] = ["private-unknown-location"]
    elif distortion == "wrong-id":
        modified["cores"][0]["candidate_id"] = "c5"
    replies = [json.dumps(modified), json.dumps(valid)]
    if distortion in {"fenced-bindings", "archive-fence"}:
        replies[0] = chr(96) * 3 + "json\n" + replies[0] + "\n" + chr(96) * 3
    calls = []
    def respond(req):
        calls.append(json.loads(req.content))
        return httpx.Response(200, json={"id": "response", "model": "fixture", "created": 0,
            "object": "chat.completion", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": replies.pop(0)}}]})
    settings = dict(provider="openai", base_url="https://offline.invalid", text_model="fixture",
                    vision_model="fixture", api_key="test-private-key")
    if distortion != "archive-fence":
        settings.update(new_batch_profile())
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(models, "ChatOpenAI", lambda **opts: ChatOpenAI(http_client=client, **opts))
        with usage_session(tmp_path, "wire", 1, settings):
            result = models.Models(settings).structured(DirectResult, "Read the source", paper)
    assert result == DirectResult.model_validate(valid, context=paper)
    assert len(calls) == expected_calls
    assert all(private in str(request["messages"]) for request in calls)
    ledger = (tmp_path / "model-usage.jsonl").read_text("utf-8")
    assert "private" not in ledger
    rows = [json.loads(line) for line in ledger.splitlines()]
    assert len(rows) == expected_calls
    if expected_calls == 2 and distortion != "archive-fence":
        code = {"wrong-id": "nonconsecutive_core_ids", "unknown-location": "unknown_source_id",
                "conflicting-links": "evidence_link_coverage"}[distortion]
        assert rows[1]["previous_schema_failure"][0]["code"] == code
        assert code in calls[1]["messages"][-1]["content"]
    else:
        assert all("previous_schema_failure" not in r for r in rows)
