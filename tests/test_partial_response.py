import json

import pytest

from paper_extraction.lean_workflow import EvidenceReview
from paper_extraction.models import AdmissionReview, StructuredResponseError
from paper_extraction.segment_evidence import ReadingAnswers
from test_agent_json import adapter


def wire(*responses):
    requests = []
    model = adapter([(json.dumps(x), "stop", 25) for x in responses], requests)
    model.block_workflow = True
    return model, requests


READ = {
    "source_questions": [{"question_id": "q1", "source_id": "p1"}, {"question_id": "q2", "source_id": "p2"}],
    "source_images": [], "affected_candidates": [],
    "passages": [{"id": "p1", "text": "closed known question"}, {"id": "p2", "text": "remaining evidence"}],
}
ANSWERS = [{"question_id": f"q{i}", "evidence_ids": [f"p{i}"], "finding": "source result", "status": "resolved"}
           for i in (1, 2)]


def test_partial_answers_keep_known_question_and_trip_count():
    model, calls = wire({"answers": ANSWERS[:1]}, {"answers": ANSWERS[1:]})
    result = model.structured(ReadingAnswers, "Check the hypotheses", READ)
    assert [row.model_dump() for row in result.answers] == ANSWERS
    assert len(calls) == 2
    correction = json.loads(calls[1]["messages"][1]["content"])
    assert [q["id"] for q in correction["passages"]] == ["p2"]


def test_invalid_partial_answer_uses_remaining_complete_retry():
    model, calls = wire({"answers": ANSWERS[:1]}, {"answers": [{**ANSWERS[1], "evidence_ids": ["p1"]}]},
                        {"answers": ANSWERS})
    result = model.structured(ReadingAnswers, "Check the hypotheses", READ)
    assert [row.model_dump() for row in result.answers] == ANSWERS
    assert len(calls) == 3
    assert json.loads(calls[2]["messages"][1]["content"]) == READ


def defect(field, source):
    return {"kind": "incorrect_claim", "draft_field": field, "current_excerpt": "x>0", "evidence_id": source,
            "source_excerpt": "x=0 is allowed", "required_change": "Include equality under D3.",
            "operation": "replace", "replacement_text": "x>=0"}


@pytest.mark.parametrize('schema', [AdmissionReview, EvidenceReview])
def test_partial_review_preserves_verdict_and_other_issue(schema):
    bad = defect('groups.1.theorem', 'p2')
    good = defect('groups.0.theorem', 'p1')
    original = {"route": "draft", "issues": [good, {**bad, "replacement_text": ""}], "evidence_ids": []}
    model, calls = wire({"verdict": original} if schema is EvidenceReview else original, {"issues": [bad]})
    packet = {"draft": {"groups": [{"theorem": "x>0"}, {"theorem": "x>0"}]}, "passages": READ["passages"]}
    response = model.structured(schema, "Review this draft", packet)
    verdict = response.verdict if schema is EvidenceReview else response
    assert verdict.route == 'draft' and [r.model_dump() for r in verdict.issues] == [good, bad]
    assert len(calls) == 2
    assert [p['id'] for p in json.loads(calls[1]['messages'][1]['content'])['passages']] == ['p2']
