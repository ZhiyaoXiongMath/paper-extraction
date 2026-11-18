"""Compatibility with a full scan stopped by the old empty-draft admission check."""
import json
import pytest

import pypdfium2 as pdfium
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from paper_extraction import agent
from paper_extraction.models import AdmissionReview, Draft, Findings, Review
from paper_extraction.sources import SourceIndex
from paper_extraction.selection import WholePaperSelection
from fixture_selection import selection_reply


def _candidate(kind, claim, label):
    return {"kind": kind, "claim": claim, "assumptions": "source hypotheses", "significance": "comparison",
            "source_number": label, "dependencies": [], "decision": "admitted", "decision_reason": "supported",
            "evidence_ids": ["p1-1"]}


@pytest.mark.parametrize("sourced_history", [True, False])
def test_old_empty_draft_recovers_scan_and_targets_history(tmp_path, monkeypatch, sourced_history):
    source = tmp_path / "scanned.pdf"
    with pdfium.PdfDocument.new() as pdf:
        page = pdf.new_page(width=300, height=400)
        page.close()
        pdf.save(str(source))
    work = tmp_path / "case"
    work.mkdir()
    old = [_candidate("current", f"Primary result {i}", f"Theorem {i}") for i in range(39)]
    settings = {"provider": "offline", "text_model": "t", "vision_model": "v", "base_url": "http://localhost"}
    config = {"configurable": {"thread_id": agent._thread_id(source, None, settings, agent._rules())}}
    source_item = SourceIndex(source, None, work).catalog()[0]
    assert source_item["text"] == ""
    prior_page_reading = "The authors report the earlier comparison with stronger hypotheses."
    legacy_read = {**source_item, "text": prior_page_reading, "transcribed_from_image": True}

    with SqliteSaver.from_conn_string(str(work / "checkpoint.sqlite")) as saver:
        reader = StateGraph(agent.EvidenceState)
        reader.add_node("previous_reader", lambda s: {"rounds": 1})
        reader.add_edge(START, "previous_reader")
        reader.add_edge("previous_reader", END)
        sub = reader.compile()

        def former_extract(s):
            sub.invoke({"rounds": 0, "messages": [HumanMessage(content=json.dumps({"passages": [legacy_read]}))]})
            return {"cursor": 1, "candidates": old, "attempt": 0, "uncertainties": [], "issues": [],
                    "reviews": [], "target_ids": []}

        workflow = StateGraph(agent.PaperState)
        workflow.add_node("extract", former_extract)
        workflow.add_node("draft", lambda s: {"attempt": 1, "status": "failed",
            "reason": "Insufficient sourced history or current contributions for standard v13. The draft is empty."})
        workflow.add_edge(START, "extract")
        workflow.add_edge("extract", "draft")
        workflow.add_edge("draft", END)
        workflow.compile(checkpointer=saver).invoke({"cursor": 0}, config)

    invocations, events = [], []

    class Model:
        def __init__(self, settings):
            pass

        def structured(self, schema, instructions, payload, image=None):
            invocations.append(schema)
            if schema is WholePaperSelection:
                return WholePaperSelection.model_validate(selection_reply(
                    payload["candidate_register"], single_current=True))
            if schema is Findings:
                assert payload["missing_categories"] == ["historical"]
                assert payload["passages"][0]["text"] == prior_page_reading
                assert payload["passages"][0]["evidence_type"] == "machine_transcription"
                assert image.startswith(b"\x89PNG")
                if not sourced_history:
                    return Findings(candidates=[], uncertainties=["No attributable theorem in the supplied page."])
                return Findings(candidates=[_candidate("historical", "An attributed historical theorem.", "H")],
                                uncertainties=[])
            if schema is Draft:
                selected = payload["candidates"]
                assert {c["kind"] for c in selected} == {"historical", "current"}
                assert len(selected) == 2 and all(c["decision"] == "admitted" for c in selected)
                return Draft(title="Sample", authors=["Author"], paragraph="Setting.", history=["Prior claim."],
                    shared_definitions=[], groups=[{"theorem": "Main claim.", "definitions": []}],
                    references=[{"key": "source", "text": "Sample"}], evidence_links=[
                        {"block_id": k, "evidence_ids": ["p1-1"]}
                        for k in ("paragraph", "history.0", "groups.0.theorem")])
            if schema is AdmissionReview:
                assert payload["draft"]["groups"]
                return AdmissionReview(route="pass", issues=[], evidence_ids=[])
            assert schema is Review
            return Review(passed=True, issues=[])

    def build(work, *unused):
        dest = work / "compiled"
        dest.mkdir()
        for name in agent._DOCUMENTS:
            (dest / name).write_text("fixture")
        return dest

    monkeypatch.setattr(agent, "Models", Model)
    monkeypatch.setattr(agent, "_evidence_agent", lambda *args: None)
    monkeypatch.setattr(agent, "_inspect", lambda *args: "Original passages checked")
    monkeypatch.setattr(agent, "_render_pages", lambda path: [(1, b"candidate page")])
    monkeypatch.setattr(agent, "build_case", build)
    result = agent.run_paper(work, source, None, settings, lambda stage, data: events.append((stage, data)),
                             retry_legacy_review=True)
    if not sourced_history:
        assert result["status"] == "failed" and "历史定理" in result["reason"]
        assert "当前核心结果" not in result["reason"] and "草稿" not in result["reason"]
        assert len(result["candidates"]) == len(old)
        assert invocations.count(Findings) == 1 and Draft not in invocations
        return
    assert result["status"] == "completed", result
    assert len(result["candidates"]) == 40
    assert sum(c["decision"] == "admitted" for c in result["candidates"]) == 2
    assert invocations.count(Findings) == invocations.count(Draft) == 1
    assert not any(stage == "extract" for stage, _ in events)
    readback = SourceIndex(source, None, work).read("p1-1")
    assert readback["transcribed_from_image"] and readback["text"] == prior_page_reading

