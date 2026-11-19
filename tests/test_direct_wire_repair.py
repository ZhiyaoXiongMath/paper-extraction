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
    ("conflicting-links", 2), ("missing-link", 2), ("extra-link", 1),
    ("invalid-patch", 2), ("unknown-location", 2), ("wrong-id", 2), ("archive-fence", 2),
    ("block-native", 1), ("block-missing-link", 2),
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
    if distortion.startswith("block-"):
        draft["paragraph"] = {"text": draft["paragraph"], "evidence_ids": ["p1-1"]}
        group = draft["groups"][0]
        group["theorem"] = {"text": group["theorem"], "candidate_id": "c1",
                            "formal_source": draft.pop("formal_sources")["groups.0.theorem"]}
        if distortion == "block-native":
            group["theorem"]["evidence_ids"] = ["p1-1"]
        del draft["core_bindings"], draft["evidence_links"]
    elif distortion == "fenced-bindings":
        draft["core_bindings"] = [{"block_id": "groups.0.theorem", "candidate_id": "c1"}]
    elif distortion == "mapped-evidence":
        draft["evidence_links"] = {r["block_id"]: r["evidence_ids"] for r in draft["evidence_links"]}
    elif distortion in {"missing-link", "invalid-patch"}:
        draft["evidence_links"].pop()
    elif distortion == "extra-link":
        draft["evidence_links"].append({"block_id": "groups.0.paragraph", "evidence_ids": ["p1-1"]})
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
    if distortion in {"conflicting-links", "missing-link", "invalid-patch", "block-missing-link"}:
        index = 0 if distortion == "conflicting-links" else -1
        link = deepcopy(valid["draft"]["evidence_links"][index])
        if distortion == "invalid-patch":
            link["evidence_ids"] = ["p2-1"]  # Registered but never shown for the repair.
        replies[1] = json.dumps({"evidence_links": [link]})
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
            adapter = models.Models(settings)
            if distortion == "invalid-patch":
                with pytest.raises(models.StructuredResponseError, match="after two attempts"):
                    adapter.structured(DirectResult, "Read the source", paper)
            else:
                result = adapter.structured(DirectResult, "Read the source", paper)
                assert result == DirectResult.model_validate(valid, context=paper)
    assert len(calls) == expected_calls
    if distortion.startswith("block-"):
        first = calls[0]["messages"][0]["content"]
        assert "BlockDirectResult" in first and "core_bindings" not in first
    assert all(private in str(request["messages"]) for request in calls)
    ledger = (tmp_path / "model-usage.jsonl").read_text("utf-8")
    assert "private" not in ledger
    rows = [json.loads(line) for line in ledger.splitlines()]
    assert len(rows) == expected_calls
    if expected_calls == 2 and distortion != "archive-fence":
        code = {"wrong-id": "nonconsecutive_core_ids", "unknown-location": "unknown_source_id",
                "conflicting-links": "evidence_link_coverage", "missing-link": "evidence_link_coverage",
                "invalid-patch": "evidence_link_coverage", "block-missing-link": "evidence_link_coverage"}[distortion]
        assert rows[1]["previous_schema_failure"][0]["code"] == code
        repair = json.loads(calls[1]["messages"][-1]["content"])
        if code == "evidence_link_coverage":
            assert "previous_response" not in repair
            wanted = "paragraph" if distortion == "conflicting-links" else "groups.0.theorem"
            assert [r["block_id"] for r in repair["blocks"]] == [wanted]
            if distortion != "conflicting-links":
                assert "second passage" not in str(repair)
            assert rows[1]["operation"] == "EvidenceLinkPatch"
        else:
            assert code in calls[1]["messages"][-1]["content"]
    else:
        assert all("previous_schema_failure" not in r for r in rows)


def test_mixed_reply_preserves_valid_action_even_when_draft_is_invalid(monkeypatch):
    paper = {"passages": [{"id": "p1-1", "text": "Original page", "page": 1},
                          {"id": "p2-1", "text": "Another original page", "page": 2}]}
    valid = {"source_id": "p1-1", "target_id": "p2-1", "read_image": True,
             "question": "Does the displayed hypothesis require strict positivity?"}
    invalid = {"source_id": "p1-1", "target_id": "other-paper", "question": "Read this unrelated page"}
    mixed = {"source_checks": [valid, invalid, valid], "draft": {"title": "Unverified incomplete draft"}}
    calls = []

    def respond(req):
        calls.append(json.loads(req.content))
        return httpx.Response(200, json={"id": "mixed", "model": "fixture", "created": 0,
            "object": "chat.completion", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps(mixed)}}]})

    settings = dict(provider="openai", base_url="https://offline.invalid", text_model="fixture",
                    vision_model="fixture", api_key="test-key", **new_batch_profile())
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(models, "ChatOpenAI", lambda **opts: ChatOpenAI(http_client=client, **opts))
        response = models.Models(settings).structured(DirectResult, "Read", paper)
    assert len(calls) == 1
    assert [row.target_id for row in response.source_checks] == ["p2-1"]
    assert response.draft is None
    assert response._provisional == {"draft": mixed["draft"]}
    assert response._invalid_checks == [{"field": "source_checks.1", "code": "unknown_source_id"}]


def test_invalid_only_check_gets_one_focused_correction(monkeypatch):
    paper = {"passages": [{"id": "p1-1", "text": "Origin", "page": 1},
                          {"id": "p2-1", "text": "UNRELATED-SOURCE-NEVER-RESENT", "page": 2}]}
    bad = {"source_checks": [{"source_id": "p1-1", "target_id": "missing", "question": "Which direction?"}]}
    calls = []

    def respond(req):
        calls.append(json.loads(req.content))
        return httpx.Response(200, json={"id": "invalid", "model": "fixture", "created": 0,
            "object": "chat.completion", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps(bad)}}]})

    settings = dict(provider="openai", base_url="https://offline.invalid", text_model="fixture",
                    vision_model="fixture", api_key="test-key", **new_batch_profile())
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(models, "ChatOpenAI", lambda **opts: ChatOpenAI(http_client=client, **opts))
        with pytest.raises(models.StructuredResponseError, match="after two attempts"):
            models.Models(settings).structured(DirectResult, "Read", paper)
    assert len(calls) == 2
    correction = json.loads(calls[1]["messages"][-1]["content"])
    assert correction["previous_schema_failure"] == [{"field": "source_checks.0", "code": "unknown_source_id"}]
    assert "cores to []" in correction["correction"]
    assert [row["id"] for row in correction["passages"]] == ["p1-1"]
    assert "UNRELATED-SOURCE-NEVER-RESENT" not in str(calls[1])
