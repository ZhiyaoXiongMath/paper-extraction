from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from paper_extraction.models import AdmissionReview, DraftPatch
from paper_extraction.editorial import reversed_edit_requests
from paper_extraction.profile import render_tex
from paper_extraction.store import saved_writer_targets
from paper_extraction.writer_patch import WRITER_VERSION, apply_patch, repair_targets
from test_saved_draft_review import old_terminal
from test_agent_json import adapter


def test_opposing_definition_feedback_stays_bound_to_the_field_and_original_location():
    def check(round_no, op, field, source):
        return dict(stage='review', attempt=round_no, defects=[dict(draft_field=field, evidence_id=source,
                    operation=op, required_change=f'{op} the definition')])
    field = 'shared_definitions.1'
    checks = [check(1, 'insert', field, 'p2-1'), check(2, 'remove', field, 'p2-1'),
              check(3, 'insert', field, 'p2-1'), check(3, 'remove', field, 'p1-1')]
    row, = reversed_edit_requests(checks, {field})
    assert row['source_id'] == 'p2-1' and [r['round'] for r in row['prior_requests']] == [2, 3]
    assert reversed_edit_requests(checks, {'history.2'}) == []


def test_move_group_paragraph_preserves_its_text_and_moves_it_before_the_theorem():
    draft = existing_edits()["draft"]
    field = "groups.0.paragraph"
    transition = "The next result addresses the compact case."
    draft["groups"][0]["paragraph"] = transition
    draft["evidence_links"].append({"block_id": field, "evidence_ids": ["t1-1"]})
    records = {"t1-1": {"id": "t1-1", "page": 1}}
    issue = "Place the transition before the theorem it introduces."
    verdict = {"stage": "visual", "passed": False, "issues": [issue], "repair_targets": [
        {"draft_field": field, "current_text": transition, "diagnostics": [issue]}]}
    targets = repair_targets(draft, verdict, records)
    patch = DraftPatch(replacements=[{"block_id": field, "text": transition, "paragraph_position": "before", "evidence_ids": ["t1-1"]}])
    revised = apply_patch(draft, patch, {"repair_targets": targets, "passages": list(records.values())}, records)
    tex = render_tex(revised)
    assert tex.index(transition) < tex.index(r"\begin{theorem}")
    assert draft["groups"][0].get("paragraph_position", "after") == "after"


def existing_edits():
    result = old_terminal("confirmation")
    issues = [{
        "kind": "incorrect_claim", "draft_field": "groups.0.theorem", "operation": "replace",
        "current_excerpt": "compact objects", "required_change": "The result requires connected compact objects.",
        "replacement_text": "connected compact objects", "evidence_id": "p2", "source_excerpt": "connected and compact",
    }, {
        "kind": "incorrect_attribution_or_locator", "draft_field": "groups.0.theorem", "operation": "replace",
        "current_excerpt": "p.~1", "required_change": "Give the theorem number in the locator.",
        "replacement_text": "Theorem~1", "evidence_id": "p3", "source_excerpt": "Theorem 1",
    }]
    review = AdmissionReview(route="draft", issues=issues, evidence_ids=[])
    result["reviews"][-1].update(defects=issues, issues=review.repair_issues())
    return result


def test_only_reported_blocks_and_their_sources_change():
    result = existing_edits()
    original = result["draft"]
    # An unrelated old formal block has no citation. Patch-local binding must not modify it.
    original["history"][0] = "Earlier result."
    records = {k: {"id": k, "page": 2} for k in ("p2", "p3", "t1-1")}
    targets = repair_targets(original, result["reviews"][-1], records.keys())
    assert [t["block_id"] for t in targets] == ["groups.0.theorem"]
    assert targets[0]["required_evidence_ids"] == ["p2", "p3"] and len(targets[0]["defects"]) == 2
    packet = {"passages": list(records.values()), "repair_targets": targets}
    patch = DraftPatch(replacements=[dict(block_id="groups.0.theorem",
                                          text=r"Connected compact objects qualify. \cite[Theorem~1]{source}",
                                          evidence_ids=["p2", "p3"])])
    saved = deepcopy(original)
    revised = apply_patch(original, patch, packet, records)
    expected = deepcopy(saved)
    expected["groups"][0]["theorem"] = patch.replacements[0].text
    expected["evidence_links"][-1]["evidence_ids"] = ["p2", "p3"]
    assert revised == expected and original == saved


@pytest.mark.parametrize("problem", ["unrequested", "missing_location", "invented_location", "unchanged",
                                     "duplicate_target", "extra_metadata"])
def test_provider_cannot_drop_corrections_or_rewrite_other_blocks(problem):
    original = existing_edits()
    records = {k: {"id": k} for k in ("p2", "p3", "t1-1")}
    targets = repair_targets(original["draft"], original["reviews"][-1], records.keys())
    row = {"block_id": "groups.0.theorem", "text": "Connected and compact.", "evidence_ids": ["p2", "p3"]}
    rows = [row]
    if problem == "unrequested":
        row["block_id"] = "history.0"
    elif problem == "missing_location":
        row["evidence_ids"] = ["p2"]
    elif problem == "invented_location":
        row["evidence_ids"].append("other-paper-12")
    elif problem == "unchanged":
        row["text"] = targets[0]["current_text"]
    elif problem == "extra_metadata":
        row["title"] = "Changed"
    else:
        rows.append(row)
    with pytest.raises(ValidationError):
        DraftPatch.model_validate({"replacements": rows},
                                  context={"repair_targets": targets, "passages": list(records.values())})


@pytest.mark.parametrize("change", [
    {"draft_field": "groups.1.theorem"}, {"draft_field": "references.0.text"}, {"draft_field": "title"},
    {"operation": "remove", "replacement_text": ""}, {"draft_field": "groups.0.paragraph"},
    {"evidence_id": "unregistered"}, {"current_excerpt": "different previous draft"}, {"kind": "improper_selection", "operation": "remove", "replacement_text": ""},
])
def test_structural_or_unbound_verdict_uses_whole_writer(change):
    result = existing_edits()
    last = result["reviews"][-1]
    last["defects"][1].update(change)
    last["issues"] = AdmissionReview(route="draft", issues=last["defects"], evidence_ids=[]).repair_issues()
    assert repair_targets(result["draft"], last, {"p2", "p3", "t1-1"}) == []


def test_terminal_upgrade_requires_prepatch_writer_version_and_exact_edits():
    class Index:
        def catalog(self):
            return [{"id": k} for k in ("p2", "p3", "t1-1")]
    result = existing_edits()
    assert saved_writer_targets(result, Index())
    result["writer_version"] = WRITER_VERSION
    assert not saved_writer_targets(result, Index())
    result.pop("writer_version")
    result["reviews"][-1]["issues"].append("Also restructure the introduction")
    assert not saved_writer_targets(result, Index())


@pytest.mark.parametrize(('field', 'operation', 'excerpt', 'replacement'), [
    ('groups.0.paragraph', 'insert', '', 'Connectedness is necessary for the following result.'),
    ('references.0.text', 'replace', 'Author, Sample.', 'Author, Sample (2007).'),
])
def test_paragraph_insertion_and_bibliography_edit_keep_other_blocks(field, operation, excerpt, replacement):
    result = existing_edits()
    original = result['draft']
    if operation == 'insert':
        original['groups'][0].pop('paragraph', None)
    defects = result['reviews'][-1]['defects']
    defects[1].update(draft_field=field, operation=operation, current_excerpt=excerpt, replacement_text=replacement)
    if operation == 'insert':
        defects[1]['kind'] = 'missing_required_content'
    review = AdmissionReview(route='draft', issues=defects, evidence_ids=[])
    verdict = dict(result['reviews'][-1], issues=review.repair_issues(), defects=defects)
    records = {key: {'id': key, 'page': 2} for key in ('p2', 'p3', 't1-1')}
    targets = repair_targets(original, verdict, records.keys())
    assert [row['block_id'] for row in targets] == ['groups.0.theorem', field]
    updated_theorem = r'Connected compact objects qualify. \cite[Theorem~1]{source}'
    patch = DraftPatch(replacements=[dict(block_id='groups.0.theorem', text=updated_theorem, evidence_ids=['p2']),
                                    dict(block_id=field, text=replacement, evidence_ids=['p3'])])
    packet = dict(passages=list(records.values()), repair_targets=targets)
    before = deepcopy(original)
    revised = apply_patch(original, patch, packet, records)
    assert original == before
    expected = deepcopy(before)
    expected['groups'][0]['theorem'] = updated_theorem
    next(row for row in expected['evidence_links'] if row['block_id'] == 'groups.0.theorem')['evidence_ids'] = ['p2']
    if operation == 'insert':
        expected['groups'][0]['paragraph'] = replacement
        expected['evidence_links'].append(dict(block_id=field, evidence_ids=['p3']))
    else:
        expected['references'][0]['text'] = replacement
    assert revised == expected


def test_provider_repairs_unbound_patch_inside_original_writer_call():
    result = existing_edits()
    ids = {"p2", "p3", "t1-1"}
    targets = repair_targets(result["draft"], result["reviews"][-1], ids)
    row = dict(block_id="groups.0.theorem", text="Connected compact objects qualify.", evidence_ids=["p2"])
    replies = [{"replacements": [row]}, {"replacements": [{**row, "evidence_ids": ["p2", "p3"]}]}]
    requests = []
    fake = adapter([(json.dumps(reply), "stop", 210) for reply in replies], requests)
    corrected = fake.structured(DraftPatch, "Edit the existing theorem.", {
        "passages": [{"id": k} for k in ids], "repair_targets": targets})
    assert corrected.replacements[0].evidence_ids == ["p2", "p3"]
    assert len(requests) == 2 and "all required locations" in requests[1]["messages"][-1]["content"]

@pytest.mark.parametrize("field", ["shared_definitions.0", "groups.0.definitions.0"])
def test_append_next_definition_preserves_other_blocks_and_source(field):
    result = existing_edits()
    draft, verdict = result["draft"], result["reviews"][-1]
    defect = dict(verdict["defects"][0], kind="missing_required_content", operation="insert", draft_field=field,
                  current_excerpt="", replacement_text="A compact object has finite covers.", evidence_id="p2")
    def request(path):
        row = dict(defect, draft_field=path)
        review = AdmissionReview(route="draft", issues=[row], evidence_ids=[])
        return {**verdict, "defects":[row], "issues":review.repair_issues()}
    records = {k:{"id":k,"label":k} for k in ("p2","p3","t1-1")}
    assert repair_targets(draft, request(field[:-1]+"2"), records.keys()) == []
    targets = repair_targets(draft, request(field), records.keys())
    assert [r["block_id"] for r in targets] == [field]
    patch = DraftPatch(replacements=[dict(block_id=field,text=defect["replacement_text"],evidence_ids=["p2"])])
    updated = apply_patch(draft, patch, {"repair_targets":targets,"passages":list(records.values())}, records)
    assert updated["groups"][0]["theorem"] == draft["groups"][0]["theorem"]
    assert any(r["block_id"]==field and r["evidence_ids"]==["p2"] for r in updated["evidence_links"])
