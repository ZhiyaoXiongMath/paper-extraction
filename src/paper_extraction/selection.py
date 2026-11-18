"""Attribute final v13 admission at paper scope, after local candidate discovery."""
from __future__ import annotations

from importlib.resources import files
from typing import Literal
import re

from pydantic import BaseModel, Field

from .editorial import evidence_packet
from .scope_reconciliation import organization_context, primary_uses, reconcile_scope

SELECTION_VERSION = "v13-whole-paper-2"


class CoreChoice(BaseModel):
    candidate_id: str
    role: Literal["historical", "current", "application"] | None = None
    independent_question: str = Field(min_length=1)
    source_reason: str = Field(min_length=1, description="For a current core, state positive ORIGINAL evidence of an independent substantial question or consequence beyond its use in this proof.")
    evidence_ids: list[str] = Field(min_length=1, description="Original locations supporting the assertion and the stated independent significance.")
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


GUIDANCE = """Make the final editorial selection for ONE whole original paper, using v13 B1–B5 and C1–C4.
The register is a list of local observations. Treat every local importance judgment as provisional.
Read the original passages FIRST. Identify the paper's stated questions, comparisons, main assertions and
important consequences before consulting the register of local observations. Consider where the author uses each
candidate. Turning a lemma into a yes/no question or listing its assumptions does not establish significance.
For a result that supplies estimates, an intermediate representation, an inverse, or a construction used by
another result, seek a positive source indication of a separate substantial problem, comparison or important
independent consequence. A proof's organization and the author's use of a result are evidence of its role.
Read original_organization and original_uses to distinguish where claims are announced and where they are used.
Separately named, stated and proved is not an independence argument. A local step whose iteration, correction or
combination yields the main theorem remains a proof ingredient when that is the only positive role the source gives.
Do not manufacture independence by restating the step as a local existence question. Record its specific role among
the exclusions. Conversely, retain a result that also settles an independently significant problem outside that role,
with positive original evidence for that independent conclusion. There is no quota of main results or applications.
For each current choice give the independently answerable mathematical question and cite the source facts
showing its importance. Check whether the claim still answers a recognizable question when the author's
proof construction is removed. A method's tuning parameters, intermediate estimates, witnesses, auxiliary
sequence properties, construction choices and restatements do not earn new slots by themselves.
Choose only substantial independent results and important consequences. Choose historical comparisons whose
hypotheses, conclusion and attribution the supplied original actually supports. Use role only to correct a
misclassified result according to that source. State the actual historical comparison in independent_question.
Do not strengthen source claims or demote a substantial independent application for length. If more genuinely
admitted contributions than the standard profile allows are unavoidable, retain them; the program reports scope.
For competing extractions of the same assertion choose ONE faithfully scoped representative; variant_ids hold
equivalent formulations only. Distinct clauses are separate only if they answer genuinely independent questions.
Give necessary defined concepts via definition_ids. Use metadata_ids for exact paper identity, authors and necessary
shared setting. Historical attributions and genuine scope limitations must survive. List unresolved IDs explicitly.
Group routine/proof-only/explanatory observations in exclusions, by a source-specific reason. In an exclusion
group you may compress consecutive IDs as c7..c23. Individually selected and unresolved rows take priority. Cover the register,
avoiding repeated IDs. Use only the supplied candidate IDs and original evidence IDs. Omitted observations are
recorded as unresolved; a compact output lists IDs and reasons, without copying long statements a second time."""


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
                      "scope_evidence_id", "scope_excerpt", "scope_role", "scope_evidence_version", "scope_reason_type"):
            row.pop(prior, None)
        by_signature[signature] = row
        rows.append(row)
    return rows


def selection_packet(catalog, register):
    # Sample original passages by editorial function, independently of local admission labels.
    neutral = [{**r, "decision": "unresolved"} for r in register]
    passages = evidence_packet(catalog, neutral)["passages"]
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
    return {"passages": passages, "original_organization": organization_context(catalog),
            "original_uses": primary_uses(catalog, register), "candidate_register": navigators}


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
    return list(rows.values())


def _repair_choice(conflict, choice, register, catalog, models):
    affected = [p for p in choice.selected if p.candidate_id in conflict.owners]
    related = set(conflict.owners)
    for item in affected:
        related.update(item.variant_ids)
        related.update(item.definition_ids)
    exact = [row for row in register if row["candidate_id"] in related]
    locations = {loc for row in exact for loc in row["evidence_ids"]}
    payload = {"conflict": str(conflict), "affected_selected_ids": sorted(conflict.owners),
               "previous_selection": choice.model_dump(), "candidate_rows": exact,
               "original_passages": [r for r in catalog if r["id"] in locations]}
    if any(key not in {row["candidate_id"] for row in exact} for key in conflict.owners):
        payload["registered_candidates"] = [{"candidate_id": row["candidate_id"], "source_number": row["source_number"],
                                             "claim": row["claim"][:160], "evidence_ids": row["evidence_ids"]}
                                            for row in register]
    repair = models.structured(SelectionCorrection,
        "Repair ONLY the positive rows in affected_selected_ids using the exact register and original source. "
        "Return their replacement selections. Omit a row to withdraw it. Keep distinct substantial assertions, "
        "or one representative for a real equivalence. For an incorrectly typed local observation, give an "
        "explicit source-supported role. A definition with a matching number is supporting context, not a "
        "theorem. State a valid evidence_id for each core. Do not return unaffected rows or reclassify the register.",
        payload)
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
    instructions = rubric[rubric.index("B1."):rubric.index("C3.")] + "\n" + GUIDANCE
    choice = models.structured(WholePaperSelection, instructions, packet)
    for attempt in range(3):
        try:
            rows = apply_selection(register, choice, catalog)
            break
        except SelectionConflict as exc:
            if attempt == 2:
                raise
            choice = _repair_choice(exc, choice, register, catalog, models)
    return reconcile_scope(catalog, rows, models, rubric[rubric.index("B1."):rubric.index("C1.")])
