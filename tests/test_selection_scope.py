from pathlib import Path
import shutil

import pytest

from paper_extraction import agent
from paper_extraction.admission import scoped_image_readings
from paper_extraction.lean_workflow import DirectResult, EvidenceReview
from paper_extraction.profile_inputs import GROUNDED_WORKFLOW, new_batch_profile
from paper_extraction.selection import select_contributions
from paper_extraction.selection_scope import (
    DEFINITION_SCOPE, DIRECT_SCOPE, PLANNED_SCOPE, coverage_obligations,
)
from test_lean_workflow import APPLICATION, Replay
from test_nomination_delivery import formal_source
from test_whole_selection import finding, pick


@pytest.mark.skipif(not shutil.which("pdflatex"), reason="requires local compiler")
@pytest.mark.parametrize("current", [True, False])
def test_direct_coverage_and_source_reach_the_actual_reviewer_and_manuscript(tmp_path, monkeypatch, current):
    source = formal_source(tmp_path)
    model = Replay("first")
    respond = model.structured
    settings = new_batch_profile()
    if not current:
        settings["workflow_revision"] = GROUNDED_WORKFLOW
    observed = []
    clause = APPLICATION.split(",")[0]

    def reply(schema, instructions, packet, image=None):
        if schema is DirectResult:
            assert (DIRECT_SCOPE in instructions) == current
            assert (DEFINITION_SCOPE in instructions) == current
            answer = respond(schema, instructions, packet)
            answer.cores[1].source_reason += f" A special dimension follows. Printed [{model.application}] «{clause}»"
            return answer
        if schema is EvidenceReview:
            observed.append(packet.get("coverage_to_verify"))
            assert (DEFINITION_SCOPE in instructions) == current
        return respond(schema, instructions, packet, image)

    monkeypatch.setattr(agent, "Models", lambda _: model)
    model.structured = reply
    result = agent.run_paper(tmp_path / "work", source, source.parent, settings, lambda *_: None)
    assert result["status"] == "completed" and result["delivery"]["status"] == "reviewed"
    assert clause in (Path(result["result_dir"]) / "introduction.tex").read_text(encoding="utf8")
    expected = {"candidate_id": "c2", "source_id": model.application, "scope": clause,
                "source_bound": True, "bound_field": "groups.1.theorem", "verbatim_field": "groups.1.theorem",
                "selection_reason": "A finite-time singularity in the specified dimension is established. A special dimension follows."}
    assert observed == ([[expected]] if current else [None])


def test_segmented_first_selection_carries_scope_and_can_still_admit_an_independent_fifth():
    claims = ["All finite signed measures converge weakly.", "Every positive probability measure converges weakly.",
              "Uniqueness holds for objects with a boundary.", "The limit realizes the prescribed spectrum.",
              "A uniform rate holds for compact subfamilies.", "Every such limit determines a boundary representation.",
              "A routine normalization changes the notation."]
    sources = [{"id": f"p{i}", "page": i, "text": f"Theorem {i}. {claim}"} for i, claim in enumerate(claims, 1)]
    candidates = [finding("current", f"Theorem {i}", claim, f"p{i}") for i, claim in enumerate(claims, 1)]

    class Selector:
        def structured(self, schema, instructions, packet):
            assert PLANNED_SCOPE in instructions and DIRECT_SCOPE not in instructions
            assert len(packet["candidate_register"]) == 7
            choices = [pick(f"c{i}", f"p{i}", independent_question=claims[i - 1]) for i in (1, 3, 4, 5, 6)]
            choices[0].update(source_reason="c2 follows by restriction. Scope [p1] «all finite signed measures»")
            choices[-1].update(source_reason="A convergence statement alone cannot assert the independent boundary representation.")
            return schema(selected=choices, exclusions=[{"candidate_ids": ["c2"], "source_reason": "Covered by c1."},
                                                         {"candidate_ids": ["c7"], "source_reason": "Routine notation."}])

    rows, summary = select_contributions(sources, candidates, Selector(), current=True, optimized=True, lean=True,
                                         strict_types=True, selection_settings=new_batch_profile())
    assert summary is None
    assert [row["candidate_id"] for row in rows if row["decision"] == "admitted"] == ["c1", "c3", "c4", "c5", "c6"]
    assert rows[-1]["decision_reason"] == "Routine notation."
    assert coverage_obligations(rows) == [{"candidate_id": "c1", "source_id": "p1",
                                          "scope": "all finite signed measures", "source_bound": True,
                                          "selection_reason": "c2 follows by restriction."}]


def test_reordered_draft_matches_only_own_theorem_or_explanation():
    candidate = {"candidate_id": "c1", "kind": "current", "decision": "admitted", "evidence_ids": ["p1"],
                 "decision_reason": "Printed [p1] «all signed inputs»"}
    draft = {"groups": [{"theorem": "all signed inputs"}, {"theorem": "positive inputs suffice",
                                                            "paragraph": "all signed inputs"}],
             "core_bindings": {"groups.0.theorem": "c2", "groups.1.theorem": "c1"}}
    assert coverage_obligations([candidate], draft)[0]["verbatim_field"] == "groups.1.paragraph"
    del draft["groups"][1]["paragraph"]
    assert coverage_obligations([candidate], draft)[0]["verbatim_field"] is None
    candidate["evidence_ids"] = ["p2"]
    assert not coverage_obligations([candidate])[0]["source_bound"]


def test_scoped_reading_keeps_statement_source_page_and_crop_identity():
    records = {"p1": {"id": "p1", "page": 1, "text": "Theorem 1. General claim.\nCorollary 1. Special claim."},
               "p2": {"id": "p2", "page": 2, "text": "Theorem 1. A different chapter."}}
    active = finding("current", "Theorem 1", "General claim.", "p1")
    full = {"source_id": "p1", "page": 1, "region": None, "transcription": "**Theorem 1** allows signed inputs."}
    crop = {**full, "region": [0, 0, .5, 1]}
    saved = {**active, "source_readings": [full, crop,
             {**full, "transcription": "Corollary 1. A separate claim."},
             {**full, "transcription": "Theorem 1. General claim.\nCorollary 1. Another heading."},
             {**full, "source_id": "p2", "page": 2}]}
    assert scoped_image_readings([active], [saved], records) == [full]
    assert scoped_image_readings([{**active, "source_readings": [crop]}], [saved], records) == [crop]
    assert not scoped_image_readings([active], [{**saved, "source_number": "Corollary 1"}], records)
    assert not scoped_image_readings([active], [{**saved, "evidence_ids": ["p2"]}], records)
