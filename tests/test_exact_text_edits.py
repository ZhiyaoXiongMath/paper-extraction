from paper_extraction.exact_revision import _exact_row, write_targeted_patch


def reviewed_field(text="The standing hypotheses include completeness. Assume a subunit path."):
    return {"block_id": "shared_definitions.1", "current_text": text,
            "required_evidence_ids": ["p2"], "bound_evidence_ids": ["p1"], "diagnostics": [],
            "defects": [{"operation": "replace", "draft_field": "shared_definitions.1",
                         "current_excerpt": "Assume a subunit path.", "replacement_text": "Assume a Lipschitz subunit path.",
                         "required_change": "State the path regularity from the source."}]}


def test_preserve_definition_name_outside_reviewed_excerpt_without_model():
    target = reviewed_field()
    packet = {"references": [], "repair_targets": [target], "passages": [{"id": "p1"}, {"id": "p2"}]}
    patch = write_targeted_patch(None, "", packet, {})
    edit, = patch.replacements
    assert edit.text == "The standing hypotheses include completeness. Assume a Lipschitz subunit path."
    assert edit.evidence_ids == ["p1", "p2"]


def test_ambiguous_overlapping_or_referenced_edit_goes_through_writer():
    assert _exact_row(reviewed_field("Assume a subunit path. Assume a subunit path."), set()) is None
    row = reviewed_field()
    row["defects"] *= 2
    assert _exact_row(row, set()) is None
    row = reviewed_field()
    row["defects"][0]["replacement_text"] = r"See \cite{new-work}."
    assert _exact_row(row, {"source"}) is None
    row["defects"][0]["replacement_text"] = r"Use Eq. (\ref{unused})."
    assert _exact_row(row, set()) is None
    row = reviewed_field()
    row["diagnostics"] = ["Definition 2 exceeds its height budget."]
    assert _exact_row(row, set()) is None
