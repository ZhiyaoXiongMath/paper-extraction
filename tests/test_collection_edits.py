"""Local additions, removals and their original source bindings."""
import pytest
from pydantic import ValidationError

from paper_extraction.models import AdmissionReview, DraftPatch
from paper_extraction.profile_variants import CURRENT_PROFILE
from paper_extraction.writer_patch import apply_patch, repair_targets, write_block_patch


def one_result():
    return {"title": "A result", "authors": ["A. Author"], "paragraph": "Consider a finite set.",
            "groups": [{"theorem": "The set is finite.", "definitions": []}], "history": [],
            "shared_definitions": [], "editorial_profile": CURRENT_PROFILE,
            "core_bindings": {"groups.0.theorem": "c1"},
            "formal_sources": {"groups.0.theorem": {"source": r"\cite[Theorem~1]{source}"}},
            "references": [{"key": "source", "text": "The source."}, {"key": "other", "text": "Another paper."}],
            "evidence_links": [{"block_id": field, "evidence_ids": ["s1"]}
                               for field in ("paragraph", "groups.0.theorem")]}


def targets_for(draft, field, operation, excerpt, edit):
    issue = {"kind": "missing_required_content" if operation == "insert" else "incorrect_attribution_or_locator",
             "draft_field": field, "current_excerpt": excerpt, "operation": operation,
             "replacement_text": edit, "required_change": "Apply this edit.", "evidence_id": "s1",
             "source_excerpt": "The source states the result."}
    review = AdmissionReview(route="draft", issues=[issue], evidence_ids=[])
    verdict = {"stage": "review", "route": "draft", "passed": False, "issues": review.repair_issues(),
               "defects": [issue], "evidence_ids": []}
    return repair_targets(draft, verdict, {"s1"}, allow_collection_edits=True)


def test_definition_collection_append_keeps_source_and_other_blocks():
    draft = one_result()
    targets = targets_for(draft, "groups.0.definitions", "insert", "", "A definition.")
    assert [row["block_id"] for row in targets] == ["groups.0.definitions.0"]
    records = {"s1": {"id": "s1", "label": "Theorem 1", "text": "A definition."}}
    packet = {"repair_targets": targets, "passages": list(records.values())}
    replacement = {"block_id": targets[0]["block_id"], "text": "A definition.", "evidence_ids": ["s1"]}
    with pytest.raises(ValidationError, match="formal source"):
        DraftPatch.model_validate({"replacements": [replacement]}, context=packet)
    replacement["formal_source"] = {"source": r"\cite[Definition~2]{source}"}
    updated = apply_patch(draft, DraftPatch(replacements=[replacement]), packet, records)
    assert updated["groups"][0]["definitions"] == ["A definition."]
    assert updated["core_bindings"] == draft["core_bindings"]
    assert updated["evidence_links"][-1]["block_id"] == "groups.0.definitions.0"


def test_unused_reference_can_be_removed_locally_but_an_attributed_key_is_retained():
    draft = one_result()
    targets = targets_for(draft, "references.1", "remove", "Another paper.", "")
    assert targets[0]["unused_reference"]
    records = {"s1": {"id": "s1", "label": "Theorem 1", "text": "A result."}}
    packet = {"repair_targets": targets, "passages": list(records.values()), "references": draft["references"]}
    patch = write_block_patch(None, "", packet)
    assert apply_patch(draft, patch, packet, records)["references"] == draft["references"][:1]
    draft["formal_sources"]["groups.0.theorem"]["source"] = r"\cite[Theorem~2]{other}"
    assert targets_for(draft, "references.1", "remove", "Another paper.", "") == []
