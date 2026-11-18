"""Deterministic whole-paper selection used only by offline provider fixtures."""


def selection_reply(register, *, single_current=False):
    selected, excluded, metadata = [], [], []
    definitions = [r["candidate_id"] for r in register if r["kind"] == "definition"]
    has_current = False
    for row in register:
        kind, key = row["kind"], row["candidate_id"]
        if kind == "definition":
            continue
        if kind == "setting":
            metadata.append(key)
            continue
        if kind == "current" and has_current and single_current:
            excluded.append(key)
            continue
        deps = definitions if kind == "current" and not has_current else []
        selected.append({"candidate_id": key, "independent_question": row["claim"],
                         "b2_source_facts": [{"source_id": row["evidence_ids"][0], "fact": row["claim"]}],
                         "source_reason": "This fixture selects the cited original assertion.",
                         "evidence_ids": row["evidence_ids"], "definition_ids": deps, "variant_ids": []})
        has_current = has_current or kind == "current"
    if not has_current:
        excluded.extend(definitions)
    return {"selected": selected, "unresolved": [], "metadata_ids": metadata,
            "exclusions": [{"candidate_ids": excluded,
                            "source_reason": "Support for the fixture's main result."}] if excluded else []}

