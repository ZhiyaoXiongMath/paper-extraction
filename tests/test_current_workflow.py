"""Acceptance regressions for the current graph; all model replies are local fixtures."""
import shutil
from pathlib import Path

import pytest
import pypdfium2 as pdfium
from langchain_core.messages import AIMessage

from fixture_selection import selection_reply
from paper_extraction import agent
from paper_extraction.profile_variants import PREVIOUS_PROFILE
from paper_extraction.models import AdmissionReview, CurrentDraft, DraftPatch, Findings, Review, TextFindings
from paper_extraction.profile_inputs import current_rules, new_batch_profile, node_prompt
from paper_extraction.selection import WholePaperSelection
from paper_extraction.sources import SourceIndex
from paper_extraction.writer_patch import apply_patch, repair_targets
from test_profile import synthetic_draft


def claim(location):
    return {"kind": "current", "claim": "If x is positive, then x+x is positive.", "assumptions": "x>0",
            "significance": "Fixture conclusion", "evidence_ids": [location], "decision": "admitted",
            "decision_reason": "Main assertion in the supplied source", "dependencies": [], "source_number": "Theorem 2"}


class LocalModel:
    def __init__(self):
        self.text = self
        self.calls = []
        self.checks = []
        self.images = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        assert node_prompt("inspect") in messages[0].content
        self.calls.append("inspect")
        return AIMessage(content="The supplied passages support the draft.")

    def transcribe(self, png, page):
        self.images.append((png, page))
        return "The formula contains an upper bar."

    def structured(self, schema, instructions, packet, image=None):
        self.calls.append(schema)
        assert "MATHEMATICAL INTRODUCTION — v13" not in instructions
        if schema in {TextFindings, Findings}:
            assert node_prompt("extract") in instructions
            primary = packet["passages"][0]["id"]
            if schema is TextFindings:
                return schema(candidates=[claim(primary)], uncertainties=[], source_checks=self.checks)
            assert packet["initial_findings"]["candidates"]
            assert len(packet["source_images"]) == 1
            assert len(packet["source_questions"]) == 2
            assert [r["id"] for r in packet["passages"]].count(primary) == 1
            return schema(candidates=[claim(primary)], uncertainties=[])
        if schema is WholePaperSelection:
            assert instructions == current_rules() + "\n" + node_prompt("select")
            return schema.model_validate(selection_reply(packet["candidate_register"]))
        if schema is CurrentDraft:
            assert node_prompt("draft") in instructions
            draft = synthetic_draft()
            location = packet["candidates"][0]["evidence_ids"][0]
            draft.update(history=[], evidence_links=[{"block_id": block, "evidence_ids": [location]}
                                                     for block in ("paragraph", "groups.0.theorem")])
            return schema.model_validate(draft)
        if schema is AdmissionReview:
            assert node_prompt("review") in instructions
            return schema(route="pass", issues=[], evidence_ids=[])
        assert schema is Review and node_prompt("visual") in instructions
        assert image.startswith(b"\x89PNG")
        return Review(passed=True, issues=[])


@pytest.mark.skipif(not shutil.which("pdflatex"), reason="end-to-end delivery needs pdfLaTeX")
@pytest.mark.parametrize("round_limit", [False, True], ids=["first_draft", "third_draft"])
def test_current_no_history_is_deliverable(tmp_path, monkeypatch, round_limit):
    source = tmp_path / "source" / "positive.tex"
    source.parent.mkdir()
    source.write_text(r"\documentclass{article}\title{Positive real numbers}\author{Fixture Author}"
                      r"\begin{document}\maketitle Theorem 2. If $x>0$, then $x+x>0$.\end{document}")
    model = LocalModel()
    monkeypatch.setattr(agent, "Models", lambda settings: model)
    if round_limit:
        def layout_failure(*args):
            raise ValueError(f"Layout fixture finding in round {model.calls.count(CurrentDraft)}")
        monkeypatch.setattr(agent, "build_case", layout_failure)
    events = []
    result = agent.run_paper(tmp_path / "case", source, source.parent, new_batch_profile(PREVIOUS_PROFILE),
                             lambda stage, data: events.append((stage, data)))

    assert result["status"] == "completed", result.get("reason", result)
    assert result["draft"]["history"] == []
    assert (Path(result["result_dir"]) / "introduction.pdf").read_bytes().startswith(b"%PDF")
    assert model.calls[:2] == [TextFindings, WholePaperSelection]
    assert model.calls.count(TextFindings) == 1 and Findings not in model.calls
    assert not any(data.get("tool") == "check_prerequisites" for _, data in events)
    if round_limit:
        assert result["draft_round"] == model.calls.count(CurrentDraft) == 3
        assert model.calls[-1] is CurrentDraft
        assert model.calls.count(AdmissionReview) == 2
    else:
        assert result["draft_round"] == 1 and result["delivery"]["status"] == "reviewed"
        assert Review in model.calls


def test_digital_pdf_resolves_specific_doubts_using_one_crop(tmp_path, monkeypatch):
    model = LocalModel()
    model.checks = [
        {"source_id": "p1-1", "question": question, "query": "curvature", "read_image": True,
         "region": [0.1, 0.2, 0.9, 0.4]}
        for question in ("Is the mean curvature overlined?", "Does the overline include the subscript?")]
    calls = []

    class DigitalSource:
        def catalog(self):
            return [{"id": "p1-1", "text": "The printed mean curvature symbol is unclear.", "page": 1}]

        def search(self, query):
            calls.append(("search", query))
            return self.catalog()

        def image(self, page, region):
            calls.append(("image", page, region))
            return b"fixture cropped page"

    monkeypatch.setattr(agent, "Models", lambda settings: model)
    graph, _ = agent._make_graph(tmp_path, tmp_path / "source.pdf", None, {}, lambda *args: None, None, "",
                                 DigitalSource(), None, None, lambda: 0, current=True)
    state = graph.invoke({"cursor": 0, "candidates": [], "uncertainties": [], "attempt": 0,
                          "issues": [], "reviews": []}, interrupt_after=["extract"])
    assert state["candidates"][0]["evidence_ids"] == ["p1-1"]
    assert model.calls == [TextFindings, Findings]
    assert model.images == [(b"fixture cropped page", 1)]
    assert calls == [("search", "curvature"), ("image", 1, (0.1, 0.2, 0.9, 0.4))]


def test_scanned_page_keeps_automatic_ocr_without_a_tool_loop(tmp_path, monkeypatch):
    source = tmp_path / "scan.pdf"
    with pdfium.PdfDocument.new() as pdf:
        page = pdf.new_page(width=210, height=297)
        page.close()
        pdf.save(str(source))
    work = tmp_path / "case"
    work.mkdir()
    index = SourceIndex(source, None, work)
    model = LocalModel()
    monkeypatch.setattr(agent, "Models", lambda settings: model)
    graph, _ = agent._make_graph(work, source, None, {}, lambda *args: None, None, "", index,
                                 None, None, lambda: 0, current=True)
    graph.invoke({"cursor": 0, "candidates": [], "uncertainties": [], "attempt": 0, "issues": [], "reviews": []},
                 interrupt_after=["extract"])
    assert model.calls == [TextFindings]
    assert len(model.images) == 1 and model.images[0][0].startswith(b"\x89PNG")
    assert index.catalog()[0]["text"] == "The formula contains an upper bar."
    assert index.catalog()[0]["transcribed_from_image"]


def test_no_history_draft_accepts_source_bound_local_patch():
    draft = {**synthetic_draft(), "editorial_profile": agent.CURRENT_PROFILE, "history": [],
             "evidence_links": [{"block_id": key, "evidence_ids": ["p1-1"]}
                                for key in ("paragraph", "groups.0.theorem")]}
    source = {"id": "p1-1", "text": "Theorem 2 is stated for positive reals.", "page": 1}
    review = {"stage": "compile", "passed": False, "issues": ["overfull paragraph"],
              "repair_targets": [{"draft_field": "paragraph", "current_text": draft["paragraph"],
                                  "diagnostics": ["overfull paragraph"]}]}
    targets = repair_targets(draft, review, [source["id"]])
    assert [t["block_id"] for t in targets] == ["paragraph"]
    patch = DraftPatch(replacements=[{"block_id": "paragraph", "text": "Consider positive reals.",
                                      "evidence_ids": ["p1-1"]}])
    updated = apply_patch(draft, patch, {"repair_targets": targets, "passages": [source]}, {"p1-1": source})
    assert updated["history"] == [] and updated["groups"] == draft["groups"]
    assert updated["paragraph"] == "Consider positive reals."
