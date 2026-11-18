from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from paper_extraction.models import AdmissionReview, DraftPatch
from paper_extraction.store import saved_writer_targets
from paper_extraction.writer_patch import WRITER_VERSION, apply_patch, repair_targets
from test_saved_draft_review import old_terminal
from test_agent_json import adapter


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
    {"evidence_id": "unregistered"}, {"current_excerpt": "different previous draft"}, {"kind": "improper_selection"},
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
