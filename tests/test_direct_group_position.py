"""Recover mislocated display fields without forgiving any mathematical or evidence error."""
from copy import deepcopy
import json

import httpx
import pytest
from langchain_openai import ChatOpenAI

from paper_extraction import models
from paper_extraction.block_response import expand_direct_blocks, place_paragraph_positions
from paper_extraction.lean_workflow import DirectResult
from paper_extraction.profile_inputs import GROUNDED_WORKFLOW, new_batch_profile


def native_draft():
    cores, groups = [], []
    for i in range(1, 7):
        claim = f"Le résultat numéro {i} contrôle la quantité correspondante."
        sources = [f"p{i}-1"]
        cores.append({"candidate_id": f"c{i}", "kind": "current", "source_number": f"Theorem {i}",
                      "claim": claim, "independent_question": "Quelle quantité est contrôlée ?",
                      "source_reason": "Le domaine est précisé.", "evidence_ids": sources})
        groups.append({"definitions": [], "theorem": {
            "text": claim, "candidate_id": f"c{i}", "evidence_ids": sources,
            "formal_source": {"source": rf"\cite[Theorem {i}]{{source}}"}},
            "paragraph": {"text": f"Précisions pour {i}.", "evidence_ids": sources},
            "paragraph_position": "before" if i % 2 else "after"})
    return {"cores": cores, "draft": {"title": "Une étude", "authors": ["A"],
            "paragraph": {"text": "Considérons ces six quantités.", "evidence_ids": ["p1-1"]},
            "references": [{"key": "source", "text": "A, Une étude."}],
            "history": [], "shared_definitions": [], "groups": list(reversed(groups))}}


@pytest.mark.parametrize("kind,expected_calls", [
    ("valid", 1), ("five-moves", 1), ("duplicates", 1), ("conflict", 2),
    ("invalid-enum", 2), ("unknown-key", 2), ("unknown-source", 2),
    ("unknown-binding", 2), ("pinned-previous", 2),
])
def test_block_positions_through_model_adapter(monkeypatch, kind, expected_calls):
    correct = native_draft()
    paper = {"passages": [{"id": row["evidence_ids"][0], "text": row["claim"]} for row in correct["cores"]]}
    answer = deepcopy(correct)
    for group in answer["draft"]["groups"][1:]:
        if kind != "valid":
            group["paragraph"]["paragraph_position"] = group["paragraph_position"]
            if kind != "duplicates":
                del group["paragraph_position"]
    first = answer["draft"]["groups"][1]
    if kind == "conflict":
        first["paragraph_position"] = "after" if first["paragraph"]["paragraph_position"] == "before" else "before"
    elif kind == "invalid-enum":
        first["paragraph"]["paragraph_position"] = "middle"
    elif kind == "unknown-key":
        first["paragraph"]["invented"] = 1
    elif kind == "unknown-source":
        answer["cores"][0]["evidence_ids"] = ["not-in-this-paper"]
    elif kind == "unknown-binding":
        first["theorem"]["candidate_id"] = "c99"
    if kind in {"five-moves", "duplicates"}:
        snapshot = deepcopy(answer)
        repaired, paths = place_paragraph_positions(answer)
        assert answer == snapshot and repaired == correct
        assert len(paths) == 5 and place_paragraph_positions(repaired) == (repaired, [])

    calls, events = [], []
    def respond(req):
        calls.append(req)
        reply = answer if len(calls) == 1 else correct
        return httpx.Response(200, json={"id": "fixture", "model": "fixture", "created": 0,
            "object": "chat.completion", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": json.dumps(reply)}}]})

    settings = dict(provider="openai", base_url="https://offline.invalid", text_model="fixture",
                    vision_model="fixture", api_key="test", **new_batch_profile())
    if kind == "pinned-previous":
        settings["workflow_revision"] = GROUNDED_WORKFLOW
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(models, "ChatOpenAI", lambda **opts: ChatOpenAI(http_client=client, **opts))
        adapter = models.Models(settings)
        adapter.on_direct_format_repair = events.append
        result = adapter.structured(DirectResult, "Read", paper)
    assert len(calls) == expected_calls
    assert result == DirectResult.model_validate(expand_direct_blocks(correct), context=paper)
    if kind in {"five-moves", "duplicates"}:
        assert events == [{"fields": paths, "schema_attempt": 1, "validated": True, "avoided_retry": True}]
    elif kind in {"valid", "pinned-previous"}:
        assert events == []
    else:
        assert events and all(not row["avoided_retry"] and not row["validated"] for row in events)
