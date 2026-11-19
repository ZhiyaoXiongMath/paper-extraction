from copy import deepcopy

import pytest
from pydantic import ValidationError

from paper_extraction.profile import BuildFailure, draft_layout_targets, render_tex, saved_build_targets
from paper_extraction.reference_patch import ReferencePatch, clean_comparison_locators, reference_packet
from paper_extraction.writer_patch import apply_patch, repair_targets
from test_heading_sources import paper_with_comparison
from test_writer_patch import existing_edits


def test_comparison_citation_reports_the_original_paragraph():
    draft = paper_with_comparison()
    draft["groups"][1]["paragraph"] += r" \cite{energy}."
    with pytest.raises(BuildFailure, match=r"groups\.1\.paragraph.*energy") as failure:
        render_tex(draft)
    targets = failure.value.repair_targets
    assert [r["draft_field"] for r in targets] == ["groups.1.paragraph"]
    assert saved_build_targets(draft, [str(failure.value)]) == targets
    assert [r["draft_field"] for r in draft_layout_targets(draft)
            if any("energy" in issue for issue in r["diagnostics"])] == ["groups.1.paragraph"]


def test_reference_and_block_are_applied_together_from_verified_bibliography():
    draft = existing_edits()["draft"]
    field = "groups.0.paragraph"
    value = r"The bounded case is described elsewhere. \cite{energy}."
    entry = "U. CEGRELL, Pluricomplex energy, Acta Mathematica, 180:2 (1998), 187-217."
    draft["groups"][0]["paragraph"] = value
    draft["evidence_links"].append(dict(block_id=field, evidence_ids=["p16-1"]))
    records = {"p16-1": {"id": "p16-1", "page": 16, "text": "The comparison was extended in [7]."},
               "p21-1": {"id": "p21-1", "page": 21, "text": "[7] " + entry},
               "t1-1": {"id": "t1-1", "page": 1}}
    check = {"stage": "compile", "passed": False, "issues": ["missing key"],
             "repair_targets": [{"draft_field":field, "current_text": value, "missing_reference_keys":["energy"],
                               "diagnostics":["Missing bibliography key energy."]}]}
    targets = repair_targets(draft, check, records)
    targets[0]["defects"] = [{"required_change":"Add the bibliography entry from p21-1."}]
    packet = {"repair_targets": targets, "references": draft["references"], "passages": [records["p16-1"]]}
    data = dict(replacements=[dict(block_id=field, evidence_ids=["p16-1"], text=value)],
                additional_references=[dict(key="energy", text=entry, source_id="p21-1")])
    context = reference_packet(packet, records)
    patch = ReferencePatch.model_validate(data, context=context)
    result = apply_patch(draft, patch, packet, records)
    assert result["references"][-1] == {"key": "energy", "text": entry}
    assert next(r["evidence_ids"] for r in result["evidence_links"] if r["block_id"] == field) == ["p16-1", "p21-1"]
    assert result["groups"] == draft["groups"]
    assert r"\bibitem{energy}" in render_tex(result)
    assert all(r["key"] != "energy" for r in draft["references"])
    data["additional_references"][0]["text"] = entry.replace("1998", "1997")
    with pytest.raises(ValidationError, match="verbatim"):
        ReferencePatch.model_validate(data, context=context)


def test_only_unresolvable_secondary_locator_is_removed():
    citation = r"\cite[Theorem~1.3 and Eq.~(\ref{unused})]{source}"
    draft = {"groups": [{"theorem": citation, "paragraph": citation}],
             "evidence_links": [{"block_id": "groups.0.paragraph", "evidence_ids": ["t1-1"]}]}
    records = {"t1-1": {"id": "t1-1", "line": 12, "text": "Source.",
               "source_statements": [{"id": "t.s1", "label": "th", "kind": "theorem", "number": "1.3",
                                      "unnumbered": False}]}}
    original = deepcopy(draft)
    cleaned = clean_comparison_locators(draft, records)
    assert cleaned["groups"][0]["paragraph"] == r"\cite[Theorem~1.3]{source}"
    assert cleaned["groups"][0]["theorem"] == citation
    assert draft == original
    records["t1-1"]["text"] += r" \label{unused}"
    assert clean_comparison_locators(draft, records) is draft
    records["t1-1"].update(text="Paper.", source_statements=[])
    assert clean_comparison_locators(draft, records) is draft
