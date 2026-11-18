"""Cost regressions for the source-bound long-paper extraction path (no provider calls)."""
import pytest
from pydantic import ValidationError

from paper_extraction.segment_evidence import (
    ReadingAnswers, SegmentLeads, attach_closed_readings, indexed_labels, label_sources, read_segment,
)


PAPER = {
    "s1": {"id": "s1", "file": "main.tex", "line": 1, "text": "Theorem A gives the general flag result."},
    "s2": {"id": "s2", "file": "main.tex", "line": 11, "text": r"\label{th:mult} Some positive multiple is representable.",
           "citation_statements": [{"label": "th:mult"}]},
    "s3": {"id": "s3", "file": "main.tex", "line": 16, "text": "The following boundary classes have different rules.",
           "citation_statements": [{"label": "th:mult"}]},
}
GAIN = "Extends the flag result with the stated qualifier."
LEAD = {"kind": "current", "source_number": "Theorem A", "claim": "There is a flag criterion.",
        "gain": GAIN, "evidence_ids": ["s1"]}
SECOND = {**LEAD, "source_number": "Theorem B", "claim": "The class is representable."}


class Fixed:
    def __init__(self, script):
        self.script = iter(script)
        self.packets = []

    def structured(self, schema, instructions, packet):
        self.packets.append(packet)
        expected, answer = next(self.script)
        assert schema is expected
        return schema.model_validate(answer, context=packet)


def read(model, passages=("s1",), ledger=None, resolve=None):
    def no_lookup(*args):
        raise AssertionError("A local answer should not trigger a lookup.")
    return read_segment(model, "Math rules", {"passages": [PAPER[k] for k in passages]},
                        {"authors": ["Author"], "title": "Paper"}, indexed_labels(PAPER),
                        ledger if ledger is not None else {}, [], resolve or no_lookup, lambda *_: None)


def test_metadata_and_real_tex_label_are_closed_locally():
    questions = [{"source_id": "s1", "field": "paper_authors", "subject": "author", "question": "Who wrote this paper?"},
                 {"source_id": "s1", "candidate_index": 0, "field": "tex_label", "subject": "th:mult",
                  "query": "th:mult", "question": "Where is this label?"}]
    model = Fixed([(SegmentLeads, {"candidates": [LEAD], "questions": questions})])
    ledger = {}
    out = read(model, ledger=ledger)
    assert model.packets[0]["paper_identity"]["authors"] == ["Author"]
    assert not out.uncertainties and out.candidates[0].evidence_ids == ["s1", "s2"]
    assert {(r["field"], r["status"]) for r in ledger.values()} == {
        ("tex_label", "resolved"), ("paper_authors", "resolved")}
    assert [r["id"] for r in label_sources("th:mult", indexed_labels(PAPER), PAPER)] == ["s2", "s3"]


def test_only_affected_candidate_sent_and_settled_reading_is_not_forwarded():
    question = {"source_id": "s1", "candidate_index": 1, "field": "assumptions", "subject": "positive multiple",
                "question": "Is the claim about a positive multiple?", "query": "th:mult"}
    script = [(SegmentLeads, {"candidates": [LEAD, SECOND], "questions": [question]}),
              (ReadingAnswers, {"answers": [{"question_id": "q1", "status": "resolved", "finding": "A positive multiple.",
                                              "evidence_ids": ["s2"]}],
                                "edits": [{"candidate_index": 1, "claim": "Some positive multiple is representable."}]}),
              (SegmentLeads, {"candidates": [LEAD, SECOND], "questions": [question]})]
    model = Fixed(script)
    ledger = {}
    def lookup(checks, origins):
        assert len(checks) == 1
        return {"source_questions": [{"source_id": "s1", "question": question["question"], "search_hits": ["s2","s3"]}],
                "passages": list(PAPER.values()), "source_images": []}
    out = read(model, ledger=ledger, resolve=lookup)
    assert [r["candidate_index"] for r in model.packets[1]["affected_candidates"]] == [1]
    assert [r["id"] for r in model.packets[1]["passages"]] == ["s1","s2","s3"]
    assert not out.uncertainties and out.candidates[1].claim == "Some positive multiple is representable."
    # Same owner, source and exact subject with a paraphrased question.
    out = read(model, ledger=ledger)
    assert out.candidates[1].claim.startswith("Some positive multiple")
    assert out.candidates[1].evidence_ids == ["s1", "s2"]


def test_unresolved_field_stays_bound_until_later_original_settles_it():
    question = {"source_id": "s1", "candidate_index": 0, "field": "assumptions", "subject": "flag generality",
                "target_id": "s2", "question": "Is a general flag necessary?"}
    model = Fixed([(SegmentLeads, {"candidates": [LEAD], "questions": [question]}),
                   (ReadingAnswers, {"answers": [{"question_id": "q1", "status": "unresolved",
                                                   "finding": "The supplied span does not identify the flag."}]}),
                   (SegmentLeads, {"candidates": [], "closed_readings": [{"question_id": "prior1", "status": "resolved",
                                      "finding": "The complete theorem requires a general flag.", "evidence_ids": ["s2"]}]})])
    def lookup(checks, origins):
        return {"source_questions": [{"source_id": "s1", "target_id": "s2"}], "source_images": [],
                "passages": [PAPER["s1"], PAPER["s2"]]}
    ledger = {}
    first = read(model, ledger=ledger, resolve=lookup)
    assert first.candidates[0].decision == "unresolved" and "s1" in first.uncertainties[0]
    second = read(model, passages=("s2",), ledger=ledger)
    assert not second.uncertainties
    recovered = attach_closed_readings([first.candidates[0].model_dump()], ledger)[0]
    assert recovered["decision"] == "admitted" and recovered["evidence_ids"] == ["s1","s2"]


def test_repeating_same_visible_passage_does_not_trigger_paid_round():
    question = {"source_id": "s1", "candidate_index": 0, "field": "definition", "subject": "genericity",
                "target_id": "s1", "question": "What does general mean here?"}
    model = Fixed([(SegmentLeads, {"candidates":[LEAD], "questions":[question]})])
    def lookup(*args):
        return {"source_questions": [{"source_id":"s1", "target_id":"s1"}], "source_images":[], "passages":[PAPER["s1"]]}
    out = read(model, resolve=lookup)
    assert out.candidates[0].decision == "unresolved" and len(model.packets) == 1


def test_resolved_answer_requires_available_original():
    packet = {"source_questions": [{"question_id": "q1"}], "passages": [PAPER["s1"]], "source_images": [],
              "affected_candidates": [{"candidate_index": 0}]}
    with pytest.raises(ValidationError):
        ReadingAnswers.model_validate({"answers": [{"question_id": "q1", "status":"resolved",
                                                     "finding":"All flags.", "evidence_ids":["missing"]}]}, context=packet)
