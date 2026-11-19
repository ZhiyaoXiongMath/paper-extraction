import shutil

import pytest

from paper_extraction import agent
from paper_extraction.admission import constrain_current_results, current_statement, verify_damaged_headings
from paper_extraction.lean_workflow import (
    AddedCores, AddedCoresWithComparisons, EvidenceReview, comparison_targets, selection_change,
)
from paper_extraction.models import AuthoritativeDraft
from paper_extraction.profile_inputs import new_batch_profile
from test_lean_workflow import APPLICATION, MAIN, Replay, field, manuscript
from test_nomination_delivery import formal_source


def test_recover_nomination_at_retained_theorem_appends_only_requested_result(tmp_path, monkeypatch):
    if not shutil.which("pdflatex"):
        pytest.skip("formal delivery requires pdfLaTeX")
    source = formal_source(tmp_path)
    model = Replay("omitted")
    original = model.structured

    def reply(schema, instructions, packet, image=None):
        if schema is EvidenceReview and model.review_count == 0:
            model.review_count += 1
            return schema(verdict={"route": "extract", "evidence_ids": [model.application], "issues": [{
                "kind": "missing_required_content", "draft_field": "groups.0.theorem", "current_excerpt": MAIN,
                "source_excerpt": APPLICATION, "evidence_id": model.application,
                "required_change": "Nominate Corollary 1.2 for selection as an independent application.",
                "operation": "recover", "replacement_text": ""}]})
        return original(schema, instructions, packet, image)

    model.structured = reply
    monkeypatch.setattr(agent, "Models", lambda _: model)
    result = agent.run_paper(tmp_path / "work", source, source.parent, new_batch_profile(), lambda *_: None)
    assert result["status"] == "completed"
    assert result["draft"]["groups"][0]["theorem"] == MAIN
    assert result["draft"]["groups"][1]["theorem"] == APPLICATION
    assert AddedCores in model.calls and AuthoritativeDraft not in model.calls
    assert not selection_change({"draft_field": "groups.0.theorem", "operation": "recover",
                                 "kind": "missing_required_content", "current_excerpt": MAIN,
                                 "required_change": "Recover the missing hypothesis of this assertion."},
                                result["draft"], targeted=True)


def test_relative_comparison_rebound_only_with_supported_identity():
    draft = manuscript([field(MAIN, "c1", "s1")])
    draft["groups"][0]["paragraph"] = "Unlike the preceding result, this conclusion is quantitative."
    path = "groups.0.paragraph"
    draft["evidence_links"].append({"block_id": path, "evidence_ids": ["s1", "s2"]})
    approved = {"c1": {"kind": "current", "evidence_ids": ["s1"]},
                "c2": {"kind": "application", "evidence_ids": ["s2"]}}
    assert comparison_targets(draft, {"c2": {**approved["c2"], "evidence_ids": ["elsewhere"]}}) == {}
    assert comparison_targets(draft, {"c2": approved["c2"]})[path]["possible_core_ids"] == ["c2"]
    result = AddedCoresWithComparisons(cores=[{"candidate_id": "c2", "text": APPLICATION,
        "evidence_ids": ["s2"], "formal_source": {"source": r"\cite[Corollary~1.2]{source}"}}],
        comparison_bindings=[{"paragraph": path, "phrase": "the preceding result", "core_id": "c2"}])
    group = result.merge(draft, approved)["groups"][0]
    assert group["paragraph"] == "Unlike Theorem~2, this conclusion is quantitative."


def test_damaged_ocr_selects_unique_caption_page_but_only_image_can_confirm():
    records = {key: {"id": key, "page": n, "text": text} for key,n,text in [
        ("p5-1", 5, "THIORÈML 4. — Si D est strictement convexe ..."),
        ("p36-1", 36, "Démonstration du théorème 4. — Fixons Z"),
    ]}
    row = {"candidate_id": "c4", "kind": "current", "decision": "admitted", "source_number": "4",
           "claim": "Independent boundary value application.", "evidence_ids": list(records)}
    assert current_statement(row, records) is None
    def unavailable(*args):
        raise ConnectionError("page reader unavailable")
    assert verify_damaged_headings([row], records, unavailable) == []
    def lookup(checks, origins):
        assert len(checks) == 1 and checks[0].read_image and list(origins) == ["p5-1"]
        return {"source_images": [{"source_id": "p5-1", "transcription": "THÉORÈME 4. — Si D est strictement convexe ..."}]}
    images = verify_damaged_headings([row, {**row, "candidate_id": "c6", "source_number": "Théorème 4"}], records, lookup)
    assert current_statement(row, records) is None
    records["p5-1"]["text"] += "\n" + images[0]["transcription"]
    constrain_current_results([row], records)
    assert row["decision"] == "admitted" and row["source_statement_id"].startswith("p5-1.")
    assert not verify_damaged_headings([row], records, lookup)
