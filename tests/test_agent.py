"""Run real graphs/checkpoints with deterministic, offline model replies."""
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from paper_extraction import agent
from paper_extraction.models import AdmissionReview, Draft, Findings, Review


class ToolChat:
    def __init__(self, owner):
        self.owner = owner

    def bind_tools(self, tools):
        assert {t.name for t in tools} == {"search_source", "read_span", "read_page_image"}
        return self

    def invoke(self, messages):
        if isinstance(messages[-1], ToolMessage):
            assert "source claim" in messages[-1].content
            return AIMessage(content="Read and checked t1-1.")
        return AIMessage(content="", tool_calls=[{"id": "call-1", "name": "read_span", "args": {"location": "t1-1"}}])


class OfflineModels:
    def __init__(self):
        self.text = ToolChat(self)
        self.extracts = 0
        self.drafts = 0
        self.visual_passes = True
        self.defect_once = False
        self.interrupt_review = False
        self.missing_application = False

    def structured(self, schema, instructions, payload, image=None):
        if schema is Findings:
            self.extracts += 1
            items = [{"kind": kind, "claim": "source claim", "assumptions": "compact",
                      "significance": "central", "evidence_ids": ["t1-1"], "decision": "admitted",
                      "decision_reason": "central in the source", "dependencies": [],
                      "source_number": kind} for kind in ("current", "historical")]
            if payload.get("targeted_issue"):
                items.append({**items[0], "kind": "application", "claim": "for every dimension n >= 10",
                              "source_number": "Corollary 2"})
            return Findings(candidates=items, uncertainties=[])
        if schema is Draft:
            self.drafts += 1
            return Draft(title="Test", authors=["Author"], paragraph="Scope.",
                history=[r"Historical claim \cite[p. 1]{source}."], shared_definitions=[],
                groups=[{"definitions": [], "theorem": r"Source claim \cite[p. 1]{source}."}],
                references=[{"key": "source", "text": "Author. Test."}],
                evidence_links=[{"block_id": k, "evidence_ids": ["t1-1"]}
                                for k in ("paragraph", "history.0", "groups.0.theorem")])
        if schema is AdmissionReview:
            if self.missing_application and not any(c["kind"] == "application" for c in payload["known_candidates"]):
                return AdmissionReview(route="extract", issues=["Missing independent application for every n >= 10"],
                                       evidence_ids=["t1-1"])
            return AdmissionReview(route="pass", issues=[], evidence_ids=[])
        assert schema is Review
        if self.interrupt_review and image is None:
            self.interrupt_review = False
            raise ConnectionError("transient interruption")
        if self.defect_once and image is None:
            self.defect_once = False
            return Review(passed=False, issues=["Theorem 1 must say every dimension n >= 10, not some dimension."])
        if image is not None and not self.visual_passes:
            return Review(passed=False, issues=["Formula clipping."])
        return Review(passed=True, issues=[])


@pytest.fixture
def paper(tmp_path, monkeypatch):
    models = OfflineModels()
    monkeypatch.setattr(agent, "Models", lambda _: models)
    source = tmp_path / "source.tex"
    source.write_text("The source claim is a statement on compact objects.", encoding="utf-8")
    work = tmp_path / "case"
    builds = []

    def build(directory, entry, source_root, draft):
        target = directory / str(len(builds))
        target.mkdir()
        for file in agent._DOCUMENTS:
            (target / file).write_text("offline fixture")
        builds.append(draft)
        return target

    monkeypatch.setattr(agent, "build_case", build)
    monkeypatch.setattr(agent, "_render_pages", lambda path: [(1, b"page1"), (2, b"page2")])
    settings = {"provider": "offline", "api_key": "unused", "base_url": "http://localhost",
                "text_model": "mock", "vision_model": "mock"}
    events = []

    def run():
        return agent.run_paper(work, source, None, settings, lambda name, data: events.append((name, data)))

    return models, run, builds, events


def test_direct_publish_uses_tools_and_survives_refresh(paper):
    model, run, builds, events = paper
    first = run()
    assert first["status"] == "completed", first
    assert first["links"] == first["draft"]["evidence_links"]
    assert len(builds) == 1
    assert [(n, d["page"]) for n, d in events if n == "visual"] == [("visual", 1), ("visual", 2)]
    assert any(n == "evidence" and d["tool"] == "read_span" for n, d in events)
    events.clear()
    assert run() == first
    assert not events and model.drafts == 1
    (Path(first["result_dir"]) / "introduction.pdf").write_text("tampered", encoding="utf-8")
    assert run()["status"] == "failed"


def test_source_defect_returns_to_writer_without_rescanning(paper):
    model, run, builds, events = paper
    model.defect_once = True
    result = run()
    assert result["status"] == "completed", result
    assert model.drafts == 2 and model.extracts == 1 and len(builds) == 1
    assert result["reviews"][0]["passed"] is False
    assert result["reviews"][0]["route"] == "draft"
    assert not any(data.get("targeted") for stage, data in events if stage == "extract")


def test_visual_failure_stops_after_two_repairs(paper):
    model, run, builds, events = paper
    model.visual_passes = False
    result = run()
    assert result["status"] == "failed", result
    assert "clipping" in result["reason"] and "result_dir" not in result
    assert len(builds) == 3
    assert not any(n == "publish" for n, _ in events)


def test_recovery_resumes_unfinished_checkpoint(paper):
    model, run, builds, events = paper
    model.interrupt_review = True
    first = run()
    assert first["status"] == "failed" and "transient interruption" in first["reason"]
    second = run()
    assert second["status"] == "completed", second
    assert model.extracts == model.drafts == len(builds) == 1



def test_missing_source_application_targets_evidence_then_redrafts(paper):
    model, run, builds, events = paper
    model.missing_application = True
    result = run()
    assert result["status"] == "completed", result
    assert result["reviews"][0]["route"] == "extract"
    assert [d["locations"] for n, d in events if n == "extract" and d["targeted"]] == [["t1-1"]]
    assert model.extracts == model.drafts == 2
    assert any(c["source_number"] == "Corollary 2" for c in result["candidates"])
    assert len(builds) == 1


def test_compile_failure_returns_to_writer(paper, monkeypatch):
    model, run, builds, events = paper
    original = agent.build_case
    steps = []

    def compile_case(*args):
        steps.append(None)
        if len(steps) == 1:
            raise ValueError("current theorem exceeds 12U")
        return original(*args)

    monkeypatch.setattr(agent, "build_case", compile_case)
    result = run()
    assert result["status"] == "completed", result
    assert model.drafts == len(steps) == 2
    assert model.extracts == 1
    assert any(r["stage"] == "compile" and not r["passed"] for r in result["reviews"])
