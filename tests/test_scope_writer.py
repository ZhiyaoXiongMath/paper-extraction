from copy import deepcopy
from unittest.mock import Mock

import pytest

from paper_extraction.citation_layout import named_prerequisite_citations
from paper_extraction.exact_revision import write_targeted_patch
from paper_extraction.grounded_writing import shared_repair_scope
from paper_extraction.models import AdmissionReview, DraftPatch
from paper_extraction.profile import render_tex
from paper_extraction.writer_patch import apply_patch, preview_exact_patch, repair_targets
from test_collection_edits import one_result


@pytest.mark.parametrize("fragment", [
    r"The conditions \cite[Hypotheses~2--3]{other} suffice.",
    r"The conditions \cite[Main Theorem~2]{source} suffice.",
    r"$\text{the conditions \cite[Hypothesis~2]{source}}$ suffice.",
    r"Assume (the conditions \cite[Definition~2]{source}).",
    r"Assume \emph{the conditions \cite[Definition~2]{source}}.",
    r"Unrelated prose. % conditions \cite[Definition~2]{source}",
])
def test_only_unambiguous_prerequisite_appositions_change(fragment):
    assert named_prerequisite_citations(fragment, {"source":r"\cite[Theorem~5]{source}"}) == fragment


def test_full_renderer_selects_local_repair_or_the_existing_scoped_writer():
    old = one_result()
    field = "groups.0.theorem"
    before = old["groups"][0]["theorem"]
    formal = old["formal_sources"][field]
    records = {"s1":{"id":"s1","label":"test","text":"a source"}}
    issue = dict(operation="replace", draft_field=field, current_excerpt=before,
                 replacement_text=r"The conditions \cite[Hypothesis~3]{source} imply $u=v$.",
                 required_change="State the source hypothesis.")
    row = dict(block_id=field,current_text=before,formal_source=formal,defects=[issue],
               bound_evidence_ids=["s1"],required_evidence_ids=["s1"], diagnostics=[])
    packet = dict(references=old["references"],repair_targets=[row],passages=list(records.values()),
                  context_blocks={})
    preview = lambda t,r: preview_exact_patch(old,t,r,packet,records)
    patch = write_targeted_patch(None,"",packet,records,preview=preview)
    fixed = apply_patch(old,patch,packet,records)
    assert fixed["formal_sources"][field] == formal
    assert "(Hypothesis~3 of the cited source)" in render_tex(fixed)
    assert old["groups"][0]["theorem"] == before
    issue["replacement_text"] = r"The conditions \cite[Theorem~7]{source} imply $u=v$."
    writer = Mock()
    writer.structured.return_value = DraftPatch(replacements=[dict(block_id=field,text="Fixed.",evidence_ids=["s1"])])
    write_targeted_patch(writer,"",packet,records,preview=preview)
    assert writer.structured.call_count == 1
    target, = writer.structured.call_args.args[2]["repair_targets"]
    assert target["diagnostics"] and target["current_text"] == before
    assert target["required_evidence_ids"] == ["s1"]


def test_shared_definition_implicit_users_keep_full_writer():
    manuscript = one_result()
    manuscript["shared_definitions"] = ["An admissible object."]
    path = "shared_definitions.0"
    row = dict(block_id=path,current_text=manuscript["shared_definitions"][0],defects=[{"draft_field":path}])
    headings = {path:"Definition 1"}
    assert not shared_repair_scope(manuscript,[row],headings)
    # Source locators do not count as local references.
    manuscript["groups"][0]["theorem"] = r"Apply \cite[Definition~1]{source}."
    assert not shared_repair_scope(manuscript,[row],headings)
    manuscript["groups"][0]["theorem"] = "Apply Definition~1."
    assert shared_repair_scope(manuscript,[row],headings)
    manuscript["groups"][0]["theorem"] = "Apply the condition above."
    heading_only = {**row,"defects":[{"draft_field":f"formal_sources.{path}.source"}]}
    assert shared_repair_scope(manuscript,[heading_only],headings)


@pytest.mark.parametrize("source_first", [True, False])
def test_new_definition_and_source_can_be_patched_without_rewriting_its_paper(source_first):
    old = one_result()
    field = "groups.0.definitions.0"
    edits = [dict(kind="missing_required_content",operation="insert",draft_field=field,current_excerpt="",
                  replacement_text="A closed path.",required_change="State the definition.",evidence_id="s1",
                  source_excerpt="A closed path."),
             dict(kind="missing_required_content",operation="insert",
                  draft_field=f'formal_sources["{field}"].source',current_excerpt="",
                  replacement_text=r"\cite[Definition~3]{source}",required_change="Give the source.",evidence_id="s1",
                  source_excerpt="Definition 3.")]
    if source_first:
        edits.reverse()
    def plan(changes, paired=True):
        review = AdmissionReview(route="draft",issues=changes,evidence_ids=[])
        verdict = dict(stage="review",route="draft",passed=False,issues=review.repair_issues(),
                       defects=changes,evidence_ids=[])
        return repair_targets(old,verdict,{"s1"},allow_collection_edits=True,resolve_fields=True,
                              pair_definition_sources=paired)
    assert not plan(edits,paired=False)
    assert not plan([next(edit for edit in edits if edit["draft_field"].startswith("formal_sources"))])
    row, = plan(edits)
    assert row["block_id"] == field and row["requires_formal_source"] and len(row["defects"]) == 2
    snapshot = deepcopy(old)
    records = {"s1":{"id":"s1","text":"A closed path.","label":"s1"}}
    patch = DraftPatch(replacements=[dict(block_id=field,text="A closed path.",evidence_ids=["s1"],
                                          formal_source={"source":r"\cite[Definition~3]{source}"})])
    prospective = apply_patch(old,patch,dict(repair_targets=[row],passages=list(records.values())),records)
    assert "Definition~3" in render_tex(prospective)
    assert old == snapshot
