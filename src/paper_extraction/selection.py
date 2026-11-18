"""Attribute final v13 admission at paper scope, after local candidate discovery."""
from __future__ import annotations

from importlib.resources import files
from typing import Literal
import re

from pydantic import BaseModel, Field

from .editorial import evidence_packet
from .scope_reconciliation import reconcile_scope

SELECTION_VERSION = "v13-whole-paper-3"


class CoreChoice(BaseModel):
    candidate_id: str
    role: Literal["historical", "current", "application"] | None = None
    independent_question: str | None = None
    source_reason: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    definition_ids: list[str] = Field(default_factory=list)
    variant_ids: list[str] = Field(default_factory=list)


class OpenChoice(BaseModel):
    candidate_id: str
    source_reason: str = Field(min_length=1)


class Exclusion(BaseModel):
    candidate_ids: list[str] = Field(min_length=1)
    source_reason: str = Field(min_length=1)


class WholePaperSelection(BaseModel):
    selected: list[CoreChoice]
    unresolved: list[OpenChoice] = Field(default_factory=list)
    exclusions: list[Exclusion] = Field(default_factory=list)
    metadata_ids: list[str] = Field(default_factory=list)
    central_open_ids: list[str] = Field(default_factory=list)


SELECTION_FIELDS = """candidate_id, definition_ids, variant_ids and metadata_ids refer to candidate_register IDs.
evidence_ids refer to source_locations IDs. For an exclusion group, c7..c23 denotes the inclusive ID range.
central_open_ids refers to unresolved candidate IDs for the B4 central-admission status."""
CORRECTION_FIELDS = """Return replacements only for affected_selected_ids. replacement rows use the same
candidate and source ID namespaces. Omit an affected row to remove that row; other rows remain in place."""


def _source_row(row):
    return {k: row[k] for k in ("candidate_id", "source_number", "claim", "assumptions",
                                 "dependencies", "local_kind", "local_evidence_ids") if k in row}


def candidate_register(candidates):
    """Preserve source-qualified variants and pool only exact repeated observations."""
    rows, by_signature = [], {}
    for c in candidates:
        role = c.get("local_kind", c["kind"])
        signature = tuple(" ".join(str(v or "").split()) for v in
                          (role, c.get("source_number"), c["claim"], c["assumptions"]))
        existing = by_signature.get(signature)
        if existing is not None:
            original_ids = c.get("local_evidence_ids", c["evidence_ids"])
            existing["evidence_ids"] = list(dict.fromkeys(existing["evidence_ids"] + original_ids))
            existing["local_evidence_ids"] = list(existing["evidence_ids"])
            existing["dependencies"] = list(dict.fromkeys(existing["dependencies"] + c.get("dependencies", [])))
            continue
        original_ids = c.get("local_evidence_ids", c["evidence_ids"])
        row = {**c, "candidate_id": f"c{len(rows) + 1}", "local_kind": role, "kind": role,
               "local_decision": c.get("local_decision", c["decision"]),
               "local_reason": c.get("local_reason", c["decision_reason"]),
               "evidence_ids": list(original_ids), "local_evidence_ids": list(original_ids),
               "dependencies": list(c.get("dependencies", []))}
        for prior in ("represented_by", "independent_question", "selection_evidence_ids", "needed_by", "context_for",
                      "scope_evidence_id", "scope_excerpt", "scope_role", "scope_group_id", "scope_central_open",
                      "scope_evidence_version", "scope_reason_type"):
            row.pop(prior, None)
        by_signature[signature] = row
        rows.append(row)
    return rows


def selection_packet(catalog, register):
    # Source excerpts and locations are data, without previous candidates' decisions.
    overview = evidence_packet(catalog, [])["passages"]
    exposed = {r["id"] for r in overview}
    exposed.update(loc for row in register for loc in row["local_evidence_ids"])
    passages = [r for r in catalog if r["id"] in exposed]
    count = max(len(register), 1)
    claim_budget = max(180, min(1100, 100_000 // count))
    navigators = []
    for row in register:
        claim = row["claim"]
        if len(claim) > claim_budget:
            claim = claim[:claim_budget] + "… (see evidence_ids)"
        navigators.append({"candidate_id": row["candidate_id"], "kind": row["local_kind"],
                           "source_number": row.get("source_number"), "claim": claim,
                           "assumptions": row["assumptions"][:claim_budget//2],
                           "dependencies": row.get("dependencies", []), "evidence_ids": row["evidence_ids"]})
    return {"passages": passages,
            "source_locations": [{"id": r["id"], "label": r.get("label", r["id"])} for r in catalog],
            "candidate_register": navigators}


def exclusion_notes(candidates):
    """Brief paper-wide reasons for the reviewer to verify or rebut from the original."""
    categories = {}
    for row in candidates:
        if not row.get("candidate_id") or row["decision"] not in {"excluded", "unresolved"}:
            continue
        key = (row["decision"], row["decision_reason"])
        categories.setdefault(key, []).append({"id": row["candidate_id"], "label": row.get("source_number"),
            "claim": row["claim"][:120], "source_id": next(iter(row.get("evidence_ids", [])), None)})
    return [{"decision": decision, "reason": reason, "observations": items}
            for (decision, reason), items in categories.items()]


class SelectionConflict(ValueError):
    def __init__(self, message, *owners):
        super().__init__(message)
        self.owners = set(owners)


class SelectionCorrection(BaseModel):
    """Entire replacement for the affected positive rows only; [] withdraws them."""
    selected: list[CoreChoice]


def apply_selection(register, choice: WholePaperSelection, catalog):
    """Resolve each row at paper scope, preserving strong claims and shared definitions."""
    rows = {r["candidate_id"]: {**r, "decision": "unresolved",
            "decision_reason": "The paper-wide selection did not settle this observation."} for r in register}
    source_ids = {r["id"] for r in catalog}
    core_kinds = {"historical", "application", "current"}
    choices, supporting = {}, {}
    for item in choice.selected:
        key = item.candidate_id
        if key not in rows:
            raise SelectionConflict(f"Unknown selected candidate ID {key}", key)
        role = item.role or rows[key]["local_kind"]
        if role not in core_kinds:
            # Authors' definitions or setting may be selected as necessary material.
            # Their local type occupies no current/historical theorem slot.
            verified = set(item.evidence_ids) <= source_ids and bool(set(item.evidence_ids).intersection(rows[key]["evidence_ids"]))
            if verified or key not in supporting:
                supporting[key] = item, verified
            continue
        if key in choices:
            existing = choices[key]
            if (existing.role or rows[key]["local_kind"]) != role:
                raise SelectionConflict(f"{key} received incompatible core roles", key)
            changes = {field: list(dict.fromkeys(getattr(existing, field) + getattr(item, field)))
                       for field in ("evidence_ids", "definition_ids", "variant_ids")}
            choices[key] = existing.model_copy(update=changes)
        else:
            choices[key] = item

    definitions, variants, context = {}, {}, {}
    assigned = set(choices)
    for key, pick in choices.items():
        row = rows[key]
        known = set(row["evidence_ids"])
        for other in dict.fromkeys(pick.variant_ids):
            if other not in rows or other == key:
                continue
            if other in choices:
                raise SelectionConflict(f"{other} is both its own selected core and an equivalent of {key}", key, other)
            # A definition or shared setting with the same label can provide context
            # without being another formulation of the theorem.
            if rows[other]["local_kind"] not in core_kinds:
                context.setdefault(other, []).append(key)
                continue
            variants.setdefault(other, []).append(key)
            known.update(rows[other]["evidence_ids"])
        if not set(pick.evidence_ids) <= source_ids or not known.intersection(pick.evidence_ids):
            raise SelectionConflict(f"{key} requires registered source IDs including evidence for the selected assertion", key)
        row.update(kind=pick.role or row["local_kind"], decision="admitted", decision_reason=pick.source_reason,
                   independent_question=pick.independent_question, selection_evidence_ids=pick.evidence_ids,
                   evidence_ids=list(dict.fromkeys(row["evidence_ids"] + pick.evidence_ids)))
        for dep in dict.fromkeys(pick.definition_ids):
            if dep in rows and dep != key:
                definitions.setdefault(dep, []).append(key)
    for key, owners in definitions.items():
        row = rows[key]
        row["needed_by"] = owners
        if key not in assigned:
            row.update(kind="definition", decision="admitted", decision_reason="Required to state " + ", ".join(owners))
            assigned.add(key)
    for key, (item, verified) in supporting.items():
        if key in assigned:
            continue
        row = rows[key]
        if verified:
            row.update(decision="admitted", decision_reason=item.source_reason, selection_evidence_ids=item.evidence_ids,
                       evidence_ids=list(dict.fromkeys(row["evidence_ids"] + item.evidence_ids)))
        else:
            row.update(decision="unresolved", decision_reason="The proposed supporting item could not be bound to its original source passage.")
        assigned.add(key)
    for key in dict.fromkeys(choice.metadata_ids):
        if key in rows and key not in assigned:
            rows[key].update(kind="setting", decision="admitted", decision_reason="Source identity or paper-wide setting.")
            assigned.add(key)
    for key, owners in variants.items():
        if key not in assigned:
            rows[key].update(decision="excluded", represented_by=owners[0],
                             decision_reason="Equivalent formulation covered by " + ", ".join(owners))
            assigned.add(key)
    for key, owners in context.items():
        rows[key]["context_for"] = owners
    for item in choice.unresolved:
        if item.candidate_id in rows and item.candidate_id not in assigned:
            rows[item.candidate_id]["decision_reason"] = item.source_reason
            assigned.add(item.candidate_id)
    for group in choice.exclusions:
        for item in dict.fromkeys(group.candidate_ids):
            interval = re.fullmatch(r"c(\d+)\s*(?:\.\.|[-–—])\s*c(\d+)", item)
            if interval:
                first, last = map(int, interval.groups())
                keys = (f"c{n}" for n in range(max(1, first), min(last, len(rows)) + 1))
            else:
                keys = [item]
            for key in keys:
                if key in rows and key not in assigned:
                    rows[key].update(decision="excluded", decision_reason=group.source_reason)
                    assigned.add(key)
    for key in choice.central_open_ids:
        row = rows.get(key)
        if row and row["kind"] in {"current", "application"} and row["decision"] == "unresolved":
            loc = next((loc for loc in row["evidence_ids"] if loc in source_ids), None)
            if loc is not None:
                row.update(scope_central_open=True, scope_evidence_id=loc)
    return list(rows.values())


def _repair_choice(conflict, choice, register, catalog, models, rubric):
    affected = [p for p in choice.selected if p.candidate_id in conflict.owners]
    related = set(conflict.owners)
    for item in affected:
        related.update(item.variant_ids)
        related.update(item.definition_ids)
    exact = [_source_row(row) for row in register if row["candidate_id"] in related]
    locations = {loc for row in exact for loc in row["local_evidence_ids"]}
    bindings = {"candidate_id", "role", "evidence_ids", "definition_ids", "variant_ids"}
    payload = {"conflict": str(conflict), "affected_selected_ids": sorted(conflict.owners),
               "previous_bindings": [p.model_dump(include=bindings) for p in affected],
               "candidate_rows": exact,
               "original_passages": [r for r in catalog if r["id"] in locations]}
    if any(key not in {row["candidate_id"] for row in exact} for key in conflict.owners):
        payload["registered_candidates"] = [{"candidate_id": row["candidate_id"], "source_number": row["source_number"],
                                             "claim": row["claim"][:160], "evidence_ids": row["evidence_ids"]}
                                            for row in register]
    repair = models.structured(SelectionCorrection, rubric + "\n" + CORRECTION_FIELDS, payload)
    replacements = [p for p in repair.selected if p.candidate_id in conflict.owners or
                    p.candidate_id not in {q.candidate_id for q in choice.selected}]
    retained = [p for p in choice.selected if p.candidate_id not in conflict.owners]
    return choice.model_copy(update={"selected": retained + replacements})


def select_contributions(catalog, candidates, models):
    register = candidate_register(candidates)
    if not register:
        return []
    packet = selection_packet(catalog, register)
    rubric = files("paper_extraction").joinpath("profile_assets", "intro-rules-v13.txt").read_text(encoding="utf-8")
    instructions = rubric + "\n" + SELECTION_FIELDS
    choice = models.structured(WholePaperSelection, instructions, packet)
    for attempt in range(3):
        try:
            rows = apply_selection(register, choice, catalog)
            break
        except SelectionConflict as exc:
            if attempt == 2:
                raise
            choice = _repair_choice(exc, choice, register, catalog, models, rubric)
    return reconcile_scope(catalog, rows, models, rubric)
