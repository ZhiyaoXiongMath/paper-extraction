from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from paper_extraction.grounded_writing import AdditionRequest, definition_uses
from paper_extraction.lean_workflow import AddedCores
from paper_extraction.models import AdmissionReview, DraftPatch
from paper_extraction.writer_patch import apply_patch, repair_targets
from test_collection_edits import one_result


def new_core(key, count):
    return {"candidate_id": key, "text": "A source-supported result.", "evidence_ids": ["s1"],
            "formal_source": {"source": r"\cite[Theorem~7]{source}"},
            "definitions": [{"text": f"Definition {i}.", "evidence_ids": ["s1"],
                             "formal_source": {"source": r"\cite[Section~1]{source}"}} for i in range(count)]}


def budget_request(remaining, check):
    draft = one_result()
    existing = ["shared_definitions.0", "shared_definitions.1", "groups.0.definitions.0", "groups.0.definitions.1",
                "groups.1.definitions.0", "groups.1.definitions.1"]
    draft["groups"].append({"theorem": "A second result.", "definitions": []})
    draft["core_bindings"]["groups.1.theorem"] = "retained2"
    draft["formal_sources"]["groups.1.theorem"] = {"source": r"\cite[Theorem~2]{source}"}
    draft["evidence_links"].append({"block_id": "groups.1.theorem", "evidence_ids": ["s1"]})
    for field in existing[:len(existing) - remaining]:
        words = "Given the whole original statement, its exact context matters."
        if field.startswith("shared"):
            draft["shared_definitions"].append(words)
        else:
            draft["groups"][int(field.split(".")[1])]["definitions"].append(words)
        draft["evidence_links"].append({"block_id": field, "evidence_ids": ["s2"]})
        draft["formal_sources"][field] = {"source": r"\cite[Section~1]{source}"}
    wanted = {k: {"kind": "current", "evidence_ids": ["s1"]} for k in ["a", "b"]}
    packet = {"requested_cores": wanted, "passages": [{"id": "s1", "text": "Original text."}],
              "references": draft["references"], "shared_definitions": draft["shared_definitions"]}
    return AdditionRequest(packet, draft, wanted, check)


@pytest.mark.parametrize("remaining,counts", [(0, (1, 0)), (1, (1, 1))])
def test_shared_and_local_definition_slots_aggregate_before_merge(remaining, counts):
    merged = []
    packet = budget_request(remaining, merged.append)
    original = deepcopy(packet.draft)
    assert packet["definition_budget"]["remaining"] == remaining
    rows = packet["definition_budget"]["retained"]
    assert any(r["owner"] == "retained2" for r in rows) and all("concept_preview" in r for r in rows)
    assert "A second result." not in json.dumps(packet)
    bad = {"cores": [new_core("a", counts[0]), new_core("b", counts[1])]}
    with pytest.raises(ValidationError, match="remaining allowance"):
        AddedCores.model_validate(bad, context=packet)
    correction = packet.budget_correction(bad)
    keys = [row["candidate_id"] for row in correction.payload["previous_rows"]]
    assert keys == (["a"] if not remaining else ["a", "b"])
    fixed = correction.merge({"cores": [new_core(k, 0) for k in keys]})
    AddedCores.model_validate(fixed, context=packet)
    assert len(merged) == 1 and len(merged[0]["groups"]) == 4
    assert original == packet.draft


def test_invalid_prospective_merge_leaves_source_intact():
    def refuse(_):
        raise ValueError("A required heading locator is missing")
    packet = budget_request(0, refuse)
    prior = deepcopy(packet.draft)
    with pytest.raises(ValidationError, match="heading locator"):
        AddedCores.model_validate({"cores": [new_core("a", 0), new_core("b", 0)]}, context=packet)
    assert packet.draft == prior


def test_reviewed_definition_swap_can_release_a_slot_inside_the_same_revision():
    packet = budget_request(0, lambda _: None)
    draft = packet.draft
    draft["groups"][1]["theorem"] = r"Apply Definition~3 and the source \cite[Definition~4]{source}."
    specific = r"An unrelated concept."
    old = draft["groups"][0]["definitions"][0]
    issue = dict(kind="missing_required_content", evidence_id="s1", source_excerpt="A named object.",
                 required_change="Replace the unused term with the necessary complete definition.")
    changes = [dict(issue, operation="remove", current_excerpt=old, replacement_text="",
                    draft_field="draft.groups[0].definitions[0]"),
               dict(issue, operation="insert", current_excerpt="", replacement_text=specific,
                    draft_field="groups.0.definitions")]
    review = AdmissionReview(route="draft", issues=changes, evidence_ids=[])
    verdict = {"stage": "review", "route": "draft", "passed": False, "evidence_ids": [],
               "issues": review.repair_issues(), "defects": changes}
    assert repair_targets(draft, verdict, {"s1", "s2"}, allow_collection_edits=True) == []
    assert repair_targets(draft, verdict, {"s1", "s2"}, allow_collection_edits=True, resolve_fields=True) == []
    changes.append(dict(issue, operation="replace", draft_field="groups.1.theorem", current_excerpt="Definition~3",
                        replacement_text="the equivalent pointwise criterion"))
    review = AdmissionReview(route="draft", issues=changes, evidence_ids=[])
    verdict.update(issues=review.repair_issues(), defects=changes)
    targets = repair_targets(draft, verdict, {"s1", "s2"}, allow_collection_edits=True, resolve_fields=True)
    assert [row["block_id"] for row in targets] == ["groups.0.definitions.0", "groups.0.definitions.2", "groups.1.theorem"]
    assert definition_uses(draft, targets, {"groups.0.definitions.0": "Definition 3"}) == {
        "groups.0.definitions.0": {"scope": "c1", "explicit_users": ["groups.1.theorem"]}}


@pytest.mark.parametrize("field", ["draft.formal_sources.groups.0.theorem.comparison",
                                   'formal_sources["groups.0.theorem"].comparison'])
def test_structural_attribution_path_repairs_only_its_owner(field):
    draft = one_result()
    old, new = r"\cite[Lemma~7]{source}", r"\cite[Theorem~8]{source}"
    block = "groups.0.theorem"
    draft["formal_sources"][block]["comparison"] = old
    issue = dict(kind="incorrect_attribution_or_locator", draft_field=field, current_excerpt=old, evidence_id="s1",
                 source_excerpt="Theorem 8.", required_change="Correct the comparison locator.", operation="replace",
                 replacement_text=new)
    review = AdmissionReview(route="draft", issues=[issue], evidence_ids=[])
    verdict = {"stage": "review", "route": "draft", "passed": False, "evidence_ids": [],
               "issues": review.repair_issues(), "defects": [issue]}
    records = {"s1": {"id": "s1", "text": "Theorem 8.", "label": "s1"}}
    assert repair_targets(draft, verdict, records) == []
    targets = repair_targets(draft, verdict, records, resolve_fields=True)
    assert [t["block_id"] for t in targets] == [block]
    response = dict(block_id=block, evidence_ids=["s1"], text=draft["groups"][0]["theorem"],
                    formal_source={**draft["formal_sources"][block], "comparison": new})
    packet = {"repair_targets": targets, "passages": list(records.values())}
    repaired = apply_patch(draft, DraftPatch(replacements=[response]), packet, records)
    assert repaired["formal_sources"][block]["comparison"] == new and repaired["groups"] == draft["groups"]
    with pytest.raises(ValidationError, match="heading-only"):
        DraftPatch.model_validate({"replacements": [{**response, "text": "A different theorem."}]}, context=packet)
