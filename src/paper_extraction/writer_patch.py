"""Source-bound edits to existing v13 Introduction blocks."""
import re
from copy import deepcopy

from pydantic import ValidationError

from .models import AdmissionReview, Draft, DraftPatch
from .profile import bind_formal_citations, saved_build_targets

WRITER_VERSION = "block-patch-v1"
PATCH_GUIDANCE = (
    "For each block_id in repair_targets, return one replacement with its complete revised text "
    "and registered evidence_ids. Use TeX fragments in text. Return replacements in the DraftPatch schema."
)
_BLOCK_PATH = re.compile(r"paragraph|history\.\d+|shared_definitions\.\d+|references\.\d+\.text|"
                         r"groups\.\d+\.(?:theorem|paragraph|definitions\.\d+)")


def _address(draft, block):
    value = draft
    pieces = [int(k) if k.isdigit() else k for k in block.split(".")]
    for piece in pieces[:-1]:
        value = value[piece]
    return value, pieces[-1]


def repair_targets(draft, verdict, registered):
    """Plan a patch only when every unresolved issue maps to an existing writable block."""
    if not draft or verdict.get("passed") is not False or not verdict.get("issues"):
        return []
    try:
        parsed = Draft.model_validate(draft)
    except ValidationError:
        return []
    links = {link.block_id: link.evidence_ids for link in parsed.evidence_links}
    registered = set(registered)
    if any(not set(locations) <= registered for locations in links.values()):
        return []
    targets = {}

    additions = []

    def target(block, *, allow_append=False):
        if not isinstance(block, str) or not _BLOCK_PATH.fullmatch(block):
            return None
        if block not in targets:
            try:
                parent, key = _address(draft, block)
                append = (allow_append and isinstance(parent, list) and key == len(parent) and
                          bool(re.fullmatch(r"(?:shared_definitions|groups\.\d+\.definitions)\.\d+", block)))
                current = "" if append else parent.get(key, "") if key == "paragraph" else parent[key]
            except (IndexError, KeyError, TypeError):
                return None
            is_empty_paragraph = key == "paragraph" and block.startswith("groups.") and not current
            if not isinstance(current, str) or (block not in links and not append and not is_empty_paragraph and not block.startswith("references.")):
                return None
            if append:
                additions.append(block)
            targets[block] = dict(block_id=block, current_text=current, bound_evidence_ids=links.get(block, []),
                                  required_evidence_ids=[], defects=[], diagnostics=[])
            if is_empty_paragraph or (key == "paragraph" and block.startswith("groups.")):
                targets[block]["paragraph_position"] = parent.get("paragraph_position", "after")
        return targets[block]

    if verdict.get("stage") == "review" and verdict.get("route") == "draft":
        try:
            review = AdmissionReview(route="draft", issues=verdict.get("defects"), evidence_ids=[])
        except ValidationError:
            return []
        if review.repair_issues() != verdict["issues"] or verdict.get("evidence_ids"):
            return []
        for defect in review.issues:
            row = target(defect.draft_field, allow_append=defect.operation == "insert" and not defect.current_excerpt.strip())
            if (row is None or defect.operation not in {"replace", "insert"}
                    or (defect.kind == "improper_selection" and defect.operation != "replace") or defect.evidence_id not in registered
                    or (defect.current_excerpt.strip() and defect.current_excerpt not in row["current_text"])):
                return []
            row["defects"].append(defect.model_dump())
            if defect.evidence_id not in row["required_evidence_ids"]:
                row["required_evidence_ids"].append(defect.evidence_id)
    elif verdict.get("stage") in {"compile", "visual"}:
        errors = verdict.get("repair_targets")
        if errors is None and verdict.get("stage") == "compile":
            errors = saved_build_targets(draft, verdict["issues"])
        if not errors:
            return []
        for error in errors:
            if not isinstance(error, dict):
                return []
            row = target(error.get("draft_field"))
            diagnostics = error.get("diagnostics")
            if (row is None or error.get("current_text") != row["current_text"] or not isinstance(diagnostics, list)
                    or not diagnostics or any(not isinstance(msg, str) or not msg.strip() for msg in diagnostics)):
                return []
            row["diagnostics"].extend(diagnostics)
            row["required_evidence_ids"] = list(row["bound_evidence_ids"])
    if additions:
        prospective = deepcopy(draft)
        for block in additions:
            parent, _ = _address(prospective, block)
            parent.append("Definition")
            prospective["evidence_links"].append({"block_id": block, "evidence_ids": targets[block]["required_evidence_ids"]})
        try:
            Draft.model_validate(prospective)
        except ValidationError:
            return []
    return list(targets.values())


def apply_patch(draft, patch, packet, records):
    """Merge only requested text and its evidence links, preserving all other serialized fields."""
    checked = DraftPatch.model_validate(patch.model_dump(), context=packet)
    result = deepcopy(draft)
    links = {link["block_id"]: link for link in result["evidence_links"]}
    for row in checked.replacements:
        parent, key = _address(result, row.block_id)
        if isinstance(parent, list) and key == len(parent):
            parent.append(row.text)
        else:
            parent[key] = row.text
        if row.paragraph_position is not None:
            parent["paragraph_position"] = row.paragraph_position
        if row.block_id in links:
            links[row.block_id]["evidence_ids"] = row.evidence_ids
        elif not row.block_id.startswith("references."):
            result["evidence_links"].append({"block_id": row.block_id, "evidence_ids": row.evidence_ids})
    bound = bind_formal_citations(result, records)
    for row in checked.replacements:
        parent, key = _address(result, row.block_id)
        normalized, _ = _address(bound, row.block_id)
        previous, _ = _address(draft, row.block_id)
        before = ("" if isinstance(previous, list) and key == len(previous)
                  else previous.get(key, "") if key == "paragraph" else previous[key])
        moved = row.paragraph_position is not None and row.paragraph_position != previous.get("paragraph_position", "after")
        if before == normalized[key] and not moved:
            raise ValueError(f"{row.block_id}: the requested change is missing after binding the original citation")
        parent[key] = normalized[key]
    Draft.model_validate(result)
    if any(key not in records for link in result["evidence_links"] for key in link["evidence_ids"]):
        raise ValueError("The repaired draft contains unregistered original evidence")
    return result
