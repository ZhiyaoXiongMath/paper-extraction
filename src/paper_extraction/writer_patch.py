"""Source-bound edits to existing v13 Introduction blocks."""
import re
from copy import deepcopy

from pydantic import ValidationError

from .models import AdmissionReview, Draft, DraftPatch
from .profile import bind_formal_citations, saved_build_targets

WRITER_VERSION = "block-patch-v1"
PATCH_GUIDANCE = (
    "Revise only the blocks listed in repair_targets. Return their complete replacement text. "
    "Resolve every listed edit against its quoted original passage, including exact mathematical "
    "hypotheses and attribution. A proposed replacement_text can refer to a short excerpt rather "
    "than the entire block. Retain the surrounding correct content, citations and original TeX "
    "notation. Submit block content without environment wrappers or document commands. For layout "
    "diagnostics preserve the mathematical statement while fitting the stated v13 limit. "
    "Keep the original IDs supporting preserved claims, include each required correction ID, "
    "and take any additional evidence from the supplied original passages."
)
_BLOCK_PATH = re.compile(r"paragraph|history\.\d+|shared_definitions\.\d+|"
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
    if any(not set(locations) <= registered for locations in links.values()):
        return []
    targets = {}

    def target(block):
        if not isinstance(block, str) or block not in links or not _BLOCK_PATH.fullmatch(block):
            return None
        if block not in targets:
            parent, key = _address(draft, block)
            targets[block] = dict(block_id=block, current_text=parent[key], bound_evidence_ids=links[block],
                                  required_evidence_ids=[], defects=[], diagnostics=[])
        return targets[block]

    if verdict.get("stage") == "review" and verdict.get("route") == "draft":
        try:
            review = AdmissionReview(route="draft", issues=verdict.get("defects"), evidence_ids=[])
        except ValidationError:
            return []
        if review.repair_issues() != verdict["issues"] or verdict.get("evidence_ids"):
            return []
        for defect in review.issues:
            row = target(defect.draft_field)
            if (row is None or defect.operation not in {"replace", "insert"}
                    or defect.kind == "improper_selection" or defect.evidence_id not in registered
                    or (defect.current_excerpt.strip() and defect.current_excerpt not in row["current_text"])):
                return []
            row["defects"].append(defect.model_dump())
            if defect.evidence_id not in row["required_evidence_ids"]:
                row["required_evidence_ids"].append(defect.evidence_id)
    elif verdict.get("stage") == "compile":
        errors = verdict.get("repair_targets")
        if errors is None:
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
    return list(targets.values())


def apply_patch(draft, patch, packet, records):
    """Merge only requested text and its evidence links, preserving all other serialized fields."""
    checked = DraftPatch.model_validate(patch.model_dump(), context=packet)
    result = deepcopy(draft)
    links = {link["block_id"]: link for link in result["evidence_links"]}
    for row in checked.replacements:
        parent, key = _address(result, row.block_id)
        parent[key] = row.text
        links[row.block_id]["evidence_ids"] = row.evidence_ids
    bound = bind_formal_citations(result, records)
    for row in checked.replacements:
        parent, key = _address(result, row.block_id)
        normalized, _ = _address(bound, row.block_id)
        previous, _ = _address(draft, row.block_id)
        if previous[key] == normalized[key]:
            raise ValueError(f"{row.block_id}: the requested change is missing after binding the original citation")
        parent[key] = normalized[key]
    Draft.model_validate(result)
    if any(key not in records for link in result["evidence_links"] for key in link["evidence_ids"]):
        raise ValueError("The repaired draft contains unregistered original evidence")
    return result
