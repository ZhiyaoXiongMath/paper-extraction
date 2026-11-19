"""A scoped review nomination must not discard an independently usable draft."""
import json
from pathlib import Path
import shutil

import pytest

from paper_extraction import agent
from paper_extraction.lean_workflow import AddedCores, EvidenceReview
from paper_extraction.models import AuthoritativeDraft, DraftPatch
from paper_extraction.profile_inputs import new_batch_profile, using_profile
from paper_extraction.selection import LeanChoice, candidate_register
from paper_extraction.sources import SourceIndex
from test_lean_workflow import APPLICATION, MAIN, Replay


def test_scoped_register_does_not_renumber_a_saved_core():
    saved = {"candidate_id": "c4", "kind": "current", "claim": MAIN, "assumptions": "",
             "decision": "admitted", "decision_reason": "Already selected.", "evidence_ids": ["p1"]}
    observed = {"kind": "application", "claim": APPLICATION, "assumptions": "", "decision": "admitted",
                "decision_reason": "Requires paper-wide selection.", "evidence_ids": ["p2"]}
    rows = candidate_register([saved, observed], preserve_ids=True)
    assert [row["candidate_id"] for row in rows] == ["c4", "c5"]


def formal_source(tmp_path):
    source = tmp_path / "source" / "352.tex"
    source.parent.mkdir()
    source.write_text(
        "\\documentclass{article}\n\\usepackage{amsthm}\n"
        "\\newtheorem{maintheorem}{Theorem}[section]\n"
        "\\newtheorem{consequence}[maintheorem]{Corollary}\n"
        "\\begin{document}\n\\section{Main results}\n"
        "\\begin{maintheorem}\\label{thm:main}\n" + MAIN + "\n\\end{maintheorem}\n"
        "\\begin{consequence}\\label{cor:application}\n" + APPLICATION + "\n\\end{consequence}\n"
        "\\end{document}\n", encoding="utf-8",
    )
    return source


@pytest.mark.skipif(not shutil.which("pdflatex"), reason="requires local compiler")
@pytest.mark.parametrize("failure", [None, "selection", "writing"])
def test_scoped_nomination_keeps_prior_core_and_delivers(tmp_path, monkeypatch, failure):
    source = formal_source(tmp_path)
    model = Replay("omitted")
    original = model.structured

    def reply(schema, instructions, packet, image=None):
        if schema is LeanChoice:
            if failure == "selection":
                assert [row["candidate_id"] for row in packet["retained_cores"]] == ["c1"]
                raise ConnectionError("test: unavailable during scoped selection")
            choice = original(schema, instructions, packet, image)
            echo = {"candidate_id": "c1", "role": "current", "source_reason": "Retain the former structure.",
                    "evidence_ids": [model.main],
                    "b2_source_facts": [{"source_id": model.main, "fact": "Previously selected"}]}
            # The preexisting approval is outside the scope of this choice.
            # The echo intentionally has no new independent_question.
            return schema(selected=[echo, *choice.model_dump()["selected"]])
        if schema is AddedCores and failure == "writing":
            raise ConnectionError("test: unavailable during newly approved addition")
        return original(schema, instructions, packet, image)

    model.structured = reply
    monkeypatch.setattr(agent, "Models", lambda _: model)
    result = agent.run_paper(tmp_path / "work", source, source.parent, new_batch_profile(), lambda *_: None)
    assert result["status"] == "completed", result.get("reason")
    assert result["delivery"]["draft_round"] == (2 if failure is None else 1)
    assert result["delivery"]["status"] == ("reviewed" if failure is None else "delivered")
    assert len(result["draft"]["groups"]) == (2 if failure is None else 1)
    assert result["draft"]["core_bindings"]["groups.0.theorem"] == "c1"
    if failure == "writing":
        assert "1 条新选中的结果尚未写入" in result["delivery"]["note"]
    case = Path(result["result_dir"])
    assert (case / "introduction.tex").is_file()
    assert (case / "introduction.pdf").read_bytes().startswith(b"%PDF")


@pytest.mark.skipif(not shutil.which("pdflatex"), reason="requires local compiler")
def test_saved_draft_preserves_intersection_after_selection_changes(tmp_path, monkeypatch):
    source = formal_source(tmp_path)
    monkeypatch.setattr(agent, "Models", lambda _: Replay("first"))
    settings = new_batch_profile()
    work = tmp_path / "original"
    original = agent.run_paper(work, source, source.parent, settings, lambda *_: None)
    assert original["status"] == "completed"

    state = json.loads((work / "published.json").read_text("utf-8"))
    withdrawn, retained = state["candidates"]
    withdrawn["decision"] = "excluded"
    state["candidates"].append({**retained, "candidate_id": "c3", "claim": "A newly selected result."})
    restored = tmp_path / "recovered"
    restored.mkdir()
    index = SourceIndex(source, source.parent, restored, admission_statements=True)
    release = settings["extraction_profile"]
    with using_profile(release):
        output = agent._deliver_current(restored, source, source.parent, state, index, lambda *_: None,
                                        current=release, strict=True)
    assert output["status"] == "completed" and output["delivery"]["status"] == "delivered"
    assert output["delivery"]["compiler_verified"]
    assert output["delivery"]["checks"]["layout"] == "passed"
    assert "1 条新选中的结果尚未写入" in output["delivery"]["note"]
    assert "撤下 1 条" in output["delivery"]["note"]
    assert output["delivered_draft"]["core_bindings"] == {"groups.0.theorem": "c2"}
    assert output["delivered_draft"]["groups"][0]["theorem"] == APPLICATION
    assert not Path(output["result_dir"]).is_relative_to(work)


@pytest.mark.skipif(not shutil.which("pdflatex"), reason="requires local compiler")
@pytest.mark.parametrize("locatable", [True, False])
def test_nomination_and_material_edit_share_one_writing_round(tmp_path, monkeypatch, locatable):
    source = formal_source(tmp_path)
    model = Replay("omitted")
    original = model.structured
    operations, drafts = [], []
    revised = MAIN.replace("finite-time Calabi flow singularity", "Calabi flow singularity in finite time")
    excerpt = MAIN if locatable else "Reviewer paraphrase without an exact manuscript match."

    def reply(schema, instructions, packet, image=None):
        operations.append(schema)
        if schema is EvidenceReview and model.review_count == 0:
            model.review_count += 1
            return schema(verdict={"route": "draft", "issues": [
                {"kind": "missing_required_content", "draft_field": "groups", "current_excerpt": "",
                 "source_excerpt": APPLICATION, "evidence_id": model.application,
                 "required_change": "Select the independent dimension application separately.",
                 "operation": "insert", "replacement_text": APPLICATION},
                {"kind": "incorrect_claim", "draft_field": "groups.0.theorem",
                 "current_excerpt": excerpt, "source_excerpt": MAIN, "evidence_id": model.main,
                 "required_change": "State the time interval explicitly in the conclusion.",
                 "operation": "replace", "replacement_text": revised},
            ], "evidence_ids": []})
        if schema is DraftPatch:
            assert locatable
            target, = packet["repair_targets"]
            assert target["block_id"] == "groups.0.theorem" and "previous_draft" not in packet
            assert target["current_text"] == MAIN
            return schema(replacements=[{"block_id": target["block_id"], "text": revised,
                                         "evidence_ids": [model.main]}])
        if schema is AuthoritativeDraft:
            assert not locatable
            assert packet["identified_defects"][0]["current_excerpt"] == excerpt
            result = original(schema, instructions, packet, image)
            result.groups[0].theorem = revised
            return result
        return original(schema, instructions, packet, image)

    model.structured = reply
    monkeypatch.setattr(agent, "Models", lambda _: model)
    def notify(stage, info):
        if stage == "draft" and info.get("mode") in {"full", "patch"}:
            drafts.append(info["attempt"])

    result = agent.run_paper(tmp_path / "work", source, source.parent, new_batch_profile(), notify)
    assert result["status"] == "completed", result.get("reason")
    assert result["delivery"]["status"] == "reviewed" and result["delivery"]["draft_round"] == 2, (
        result["delivery"], [schema.__name__ for schema in operations], drafts, result["reviews"])
    assert result["draft"]["groups"][0]["theorem"] == revised
    assert result["draft"]["groups"][1]["theorem"] == APPLICATION
    if locatable:
        assert operations.count(AddedCores) == 1
        assert DraftPatch not in operations
        assert AuthoritativeDraft not in operations
    else:
        assert AuthoritativeDraft in operations
        assert DraftPatch not in operations and AddedCores not in operations
    assert drafts == [1, 2]
