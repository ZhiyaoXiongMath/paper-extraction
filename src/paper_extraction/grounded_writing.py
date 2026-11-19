"""Inputs shared by content review, scoped writing, and local merge validation."""
import re

from pydantic import BaseModel, Field

from .lean_workflow import NewCore
from .models import MAX_DEFINITIONS, MAX_GROUP_DEFINITIONS


REVIEW_SCOPE = (
    "\nFor each material defect you already found, verify the entire affected assertion against its source: "
    "include the hypotheses, quantifier or parameter range, direction, exceptions and attribution that actually "
    "matter to this correction. Give one complete replacement per affected passage, with its real draft_field, "
    "exact current excerpt and supporting location. Coordinate dependent edits in this same verdict. "
    "Omit unaffected fields and stylistic preferences. Request missing evidence rather than guessing."
)

ADDITION_GUIDANCE = (
    "\nThe definition_budget is for the merged manuscript after the existing edits. All new cores together "
    "share its remaining slots. At zero, every definitions list must be empty. Reuse matching definitions, "
    "or state complete necessary conditions in the theorem when supported. A preview identifies a concept only. "
    "Keep scope, direction, exceptions and normalization. The retained blocks are not writable in this addition."
)


def add_definition_budget(packet, draft):
    """Count every retained definition after projection, not just the ones shared by all results."""
    links = {row["block_id"]: row["evidence_ids"] for row in draft["evidence_links"]}
    relevant_ids = {row["id"] for row in packet["passages"]}
    catalog = []
    def describe(block_id, text, owner):
        evidence = links[block_id]
        row = {"block_id": block_id, "owner": owner, "evidence_ids": evidence}
        # Never present a clipped mathematical condition as a complete definition.
        if relevant_ids.intersection(evidence):
            row["text"] = text
        else:
            term = re.search(r"\\(?:emph|textit|textbf)\{([^{}]+)\}", text)
            row["concept_preview"] = term[1] if term else " ".join(text.split()[:12])
        catalog.append(row)

    for i, text in enumerate(draft["shared_definitions"]):
        describe(f"shared_definitions.{i}", text, "shared")
    for i, group in enumerate(draft["groups"]):
        for j, text in enumerate(group["definitions"]):
            describe(f"groups.{i}.definitions.{j}", text, draft["core_bindings"][f"groups.{i}.theorem"])
    packet.pop("shared_definitions", None)
    packet["definition_budget"] = {"limit": MAX_DEFINITIONS, "remaining": MAX_DEFINITIONS - len(catalog),
                                   "per_new_group": MAX_GROUP_DEFINITIONS, "retained": catalog}


def definition_uses(draft, targets, visible_headings):
    """Expose scope and explicit local references when a requested edit changes a retained definition."""
    from .lean_workflow import block_text
    result = {}
    for target in targets:
        path = target["block_id"]
        if not (re.fullmatch(r"(?:shared_definitions|groups\.\d+\.definitions)\.\d+", path) and target["current_text"]):
            continue
        heading = visible_headings.get(path, "")
        if not re.fullmatch(r"Definition \d+", heading):
            continue
        number = heading.split()[-1]
        # A bibliography locator names a definition of the original paper, independently of our numbering.
        local_reference = re.compile(r"\bDefinition[ ~]+" + number + r"(?!\d|\.\d)", re.I)
        dependents = []
        for link in draft["evidence_links"]:
            field = link["block_id"]
            if field == path:
                continue
            body = re.sub(r"\\cite(?:\[[^\]]*\])*\{[^}]*\}", "", block_text(draft, field))
            if local_reference.search(body):
                dependents.append(field)
        local_group = path.rsplit(".definitions.", 1)[0]
        owner = draft["core_bindings"].get(f"{local_group}.theorem")
        result[path] = {"scope": owner or "shared", "explicit_users": dependents}
    return result


class AdditionRequest(dict):
    """The wire receives only new cores and source excerpts. Validate against the full draft locally."""

    def __init__(self, packet, draft, approved, validate_merge):
        super().__init__(packet)
        add_definition_budget(self, draft)
        self.draft = draft
        self.approved = approved
        self.validate_merge = validate_merge

    def check_additions(self, additions):
        # The response schema invokes this inside the ordinary bounded correction slot.
        # A rejected output has no effect on the checkpoint or the last usable draft.
        prospective = additions.merge(self.draft, self.approved)
        self.validate_merge(prospective)

    def budget_correction(self, document):
        """Reuse one existing schema correction slot for the over-budget rows and their evidence."""
        if not isinstance(document, dict) or not isinstance(document.get("cores"), list):
            return None
        try:
            # Only project structurally sound rows; malformed unrelated fields use the ordinary correction.
            from .lean_workflow import AddedCores
            AddedCores.model_validate(document)
        except ValueError:
            return None
        rows = document["cores"]
        if any(not isinstance(row, dict) or not isinstance(row.get("definitions", []), list) for row in rows):
            return None
        keys = [row.get("candidate_id") for row in rows]
        if (not all(isinstance(key, str) for key in keys) or set(keys) != set(self["requested_cores"])
                or len(keys) != len(set(keys))):
            return None
        affected = [row for row in rows if row.get("definitions")]
        if sum(len(row["definitions"]) for row in affected) <= self["definition_budget"]["remaining"]:
            return None
        return CoreBudgetCorrection(self, document, affected)


class CorrectedCoreRows(BaseModel):
    cores: list[NewCore] = Field(min_length=1)


class CoreBudgetCorrection:
    schema = CorrectedCoreRows

    def __init__(self, packet, document, affected):
        self.document = document
        self.keys = {row["candidate_id"] for row in affected}
        needed = {key: val for key, val in packet["requested_cores"].items() if key in self.keys}
        sources = {loc for core in needed.values() for loc in core["evidence_ids"]}
        for row in affected:
            for block in [row, *row["definitions"]]:
                if isinstance(block, dict):
                    sources.update(x for x in block.get("evidence_ids", []) if isinstance(x, str))
            sources.update(x for x in row.get("paragraph_evidence_ids", []) if isinstance(x, str))
        self.payload = {"requested_cores": needed, "previous_rows": affected, "definition_budget": packet["definition_budget"],
                        "references": [*packet["references"], *document.get("additional_references", [])],
                        "passages": [p for p in packet["passages"] if p["id"] in sources]}
        if images := [im for im in packet.get("checked_source_images", []) if im["source_id"] in sources]:
            self.payload["checked_source_images"] = images

    def merge(self, document):
        revised = self.schema.model_validate(document).cores
        rows = {row.candidate_id: row.model_dump() for row in revised}
        if set(rows) != self.keys or len(revised) != len(rows):
            raise ValueError("budget_correction_membership")
        shown = {p["id"] for p in self.payload["passages"]}
        if any(not set([*row.evidence_ids, *row.paragraph_evidence_ids,
                        *(loc for definition in row.definitions for loc in definition.evidence_ids)]) <= shown for row in revised):
            raise ValueError("budget_correction_unseen_source")
        return {**self.document, "cores": [rows.get(row["candidate_id"], row) for row in self.document["cores"]]}


def source_address(draft, field):
    """Normalize explicit JSON paths from the reviewer, including dotted formal-source keys."""
    path = field.removeprefix("draft.").strip()
    path = re.sub(r"\[(\d+)\]", r".\1", path)
    for block_id in draft.get("formal_sources", {}):
        for attr in ("source", "prior", "reporting", "comparison"):
            if path in (f"formal_sources.{block_id}.{attr}",
                        f'formal_sources["{block_id}"].{attr}', f"formal_sources['{block_id}'].{attr}"):
                return block_id, attr
    return path, None
