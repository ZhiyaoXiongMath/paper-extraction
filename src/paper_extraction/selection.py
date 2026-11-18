"""Attribute final v13 admission at paper scope, after local candidate discovery."""
from __future__ import annotations

from importlib.resources import files
from typing import Literal
import re

from pydantic import BaseModel, Field, ValidationInfo, model_validator

from .editorial import evidence_packet
from .focused_evidence import projected_sources
from .models import SourceCheck
from .scope_reconciliation import reconcile_scope
from .profile_variants import COUNT_OVERRIDE
from .lean_workflow import IndependentRole, ScreeningNote, independent_role_supported

SELECTION_VERSION = "v13-whole-paper-3"


class SourceFact(BaseModel):
    source_id: str = Field(min_length=1)
    fact: str = Field(min_length=1)


class CoreChoice(BaseModel):
    candidate_id: str
    role: Literal["historical", "current", "application"] | None = None
    independent_question: str | None = None
    b2_source_facts: list[SourceFact]
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


class LeanCoreChoice(CoreChoice):
    independent_role: IndependentRole | None = None


class LeanChoice(WholePaperSelection):
    selected: list[LeanCoreChoice]
    central_checks: dict[str, SourceCheck] = Field(default_factory=dict, description="For unresolved central candidates only: candidate ID -> precise source question and statement page when known")
    summary: ScreeningNote | None = Field(default=None, description="Chinese source-grounded note if there is no qualifying current result")

    @model_validator(mode="after")
    def verify_outcome(self, info: ValidationInfo):
        if not info.context:
            return self
        roles = {r["candidate_id"]: r["kind"] for r in info.context["candidate_register"]}
        current = any((s.role or roles.get(s.candidate_id)) in {"current", "application"} for s in self.selected)
        current |= any(c["kind"] in {"current", "application"} for c in info.context.get("retained_cores", []))
        if current == bool(self.summary):
            raise ValueError("Return a Chinese screening note only when the final selection contains no current theorem")
        known = {r["id"] for r in info.context["source_locations"]}
        if self.summary and (not set(self.summary.evidence_ids) <= known or
                             (self.summary.kind == "no_qualifying_result" and not self.summary.evidence_ids)):
            raise ValueError("Use registered source evidence for the screening note")
        return self


class SourceSelection(LeanChoice):
    """An optional read of the unabridged originals before deciding membership."""
    selected: list[LeanCoreChoice] = Field(default_factory=list)
    read_source_ids: list[str] = Field(default_factory=list, description="Request original passages in one batch before deciding. Leave other fields empty for this action.")

    @model_validator(mode="after")
    def verify_outcome(self, info: ValidationInfo):
        if not self.read_source_ids:
            return super().verify_outcome(info)
        if self.selected or self.summary or self.unresolved or self.exclusions:
            raise ValueError("Request the needed originals or commit the selection")
        if info.context and set(self.read_source_ids) - {r["id"] for r in info.context["source_locations"]}:
            raise ValueError("Request registered source IDs")
        return self


SELECTION_FIELDS = """candidate_id, definition_ids, variant_ids and metadata_ids refer to candidate_register IDs.
evidence_ids identify the source locations for the assertion. b2_source_facts records B2(iii)
as pairs of fact and source_id, using source_locations IDs. For an exclusion group, c7..c23 denotes the inclusive ID range.
central_open_ids refers to unresolved candidate IDs for the B4 central-admission status."""
CORRECTION_FIELDS = """Return replacements only for affected_selected_ids. replacement rows use the same
candidate and source ID namespaces, including source_id in b2_source_facts. Omit an affected row to remove that row; other rows remain in place."""


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
            if c.get("source_readings"):
                old = existing.get("source_readings", [])
                existing["source_readings"] = old + [r for r in c["source_readings"] if r not in old]
            continue
        original_ids = c.get("local_evidence_ids", c["evidence_ids"])
        row = {**c, "candidate_id": f"c{len(rows) + 1}", "local_kind": role, "kind": role,
               "local_decision": c.get("local_decision", c["decision"]),
               "local_reason": c.get("local_reason", c["decision_reason"]),
               "evidence_ids": list(original_ids), "local_evidence_ids": list(original_ids),
               "dependencies": list(c.get("dependencies", []))}
        for prior in ("represented_by", "independent_question", "b2_source_facts", "selection_evidence_ids", "needed_by", "context_for",
                      "scope_evidence_id", "scope_excerpt", "scope_role", "scope_group_id", "scope_central_open",
                      "scope_evidence_version", "scope_reason_type"):
            row.pop(prior, None)
        by_signature[signature] = row
        rows.append(row)
    return rows


def selection_packet(catalog, register, *, optimized=False):
    # Source excerpts and locations are data, without previous candidates' decisions.
    overview = evidence_packet(catalog, [])["passages"]
    exposed = {r["id"] for r in overview}
    exposed.update(loc for row in register for loc in row["local_evidence_ids"])
    passages = [r for r in catalog if r["id"] in exposed]
    if optimized:
        # Local observations identify plausible results; the selector can read
        # any full location, including a proof, appendix or a compared work.
        intro = {r["id"] for r in catalog if re.search(r"introduction|abstract", r.get("section", ""), re.I)
                 or r"\begin{abstract}" in r["text"]}
        if catalog:
            intro.add(catalog[0]["id"])
        primary = {loc for row in register for loc in row["local_evidence_ids"][:1]}
        # Always carry every identified statement span, including continuation
        # spans. A single TeX range may straddle an exceptional case or equation.
        formal = {r["id"] for r in passages if r.get("citation_statements")}
        passages = projected_sources(catalog, intro | primary | formal, abridge=True)
    count = max(len(register), 1)
    claim_budget = max(180, min(1100, 100_000 // count))
    navigators = []
    for row in register:
        claim = row["claim"]
        if not optimized and len(claim) > claim_budget:
            claim = claim[:claim_budget] + "… (see evidence_ids)"
        navigators.append({"candidate_id": row["candidate_id"], "kind": row["local_kind"],
                           "source_number": row.get("source_number"), "claim": claim,
                           "assumptions": row["assumptions"] if optimized else row["assumptions"][:claim_budget//2],
                           "dependencies": row.get("dependencies", []), "evidence_ids": row["evidence_ids"]})
    if optimized:
        for row in navigators:
            if not row["assumptions"] or row["assumptions"] in row["claim"]:
                del row["assumptions"]
            # Dependency descriptions accompany admitted statements when writing;
            # retain them here to identify required definitions, but don't repeat
            # an identical description verbatim for multiple candidates.
        repeated = {}
        for row in navigators:
            for item in row["dependencies"]:
                repeated[item] = repeated.get(item, 0) + 1
        dictionary = {item: f"d{i}" for i, item in enumerate(
            (item for item, count in repeated.items() if count > 1 and len(item) >= 24), 1)}
        for row in navigators:
            if row["dependencies"]:
                row["dependencies"] = [dictionary.get(v, v) for v in row["dependencies"]]
            else:
                del row["dependencies"]
    return {"passages": passages, **({"dependency_terms": {v:k for k,v in dictionary.items()}} if optimized and dictionary else {}),
            "source_locations": [{"id": r["id"], "label": r.get("label", r["id"])} for r in catalog],
            "candidate_register": navigators}


def review_navigator(candidates):
    """Give the independent review source coordinates for every observation, including legacy rows."""
    return [{"candidate_id": row.get("candidate_id"), "source_number": row.get("source_number"),
             "evidence_ids": list(dict.fromkeys([
                 *row.get("local_evidence_ids", []), *row.get("evidence_ids", []),
                 *row.get("selection_evidence_ids", []),
                 *(fact["source_id"] for fact in row.get("b2_source_facts", [])),
             ]))} for row in candidates]


class SelectionConflict(ValueError):
    def __init__(self, message, *owners):
        super().__init__(message)
        self.owners = set(owners)


class SelectionCorrection(BaseModel):
    """Entire replacement for the affected positive rows only; [] withdraws them."""
    selected: list[CoreChoice]


class LeanCorrection(SelectionCorrection):
    selected: list[LeanCoreChoice]


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
        unregistered = sorted({fact.source_id for fact in item.b2_source_facts} - source_ids)
        if unregistered:
            raise SelectionConflict(f"{key} has unregistered b2_source_facts.source_id: {', '.join(unregistered)}", key)
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
            changes["b2_source_facts"] = list({
                (fact.source_id, fact.fact): fact for fact in [*existing.b2_source_facts, *item.b2_source_facts]
            }.values())
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
        if not any(fact.fact.strip() for fact in pick.b2_source_facts):
            raise SelectionConflict(f"{key}: b2_source_facts is empty; enter the source fact and source_id for B2(iii)", key)
        row.update(kind=pick.role or row["local_kind"], decision="admitted", decision_reason=pick.source_reason,
                   independent_question=pick.independent_question, selection_evidence_ids=pick.evidence_ids,
                   b2_source_facts=[fact.model_dump() for fact in pick.b2_source_facts],
                   evidence_ids=list(dict.fromkeys([*row["evidence_ids"], *pick.evidence_ids,
                                                    *(fact.source_id for fact in pick.b2_source_facts)])))
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


def _repair_choice(conflict, choice, register, catalog, models, rubric, fields):
    affected = [p for p in choice.selected if p.candidate_id in conflict.owners]
    related = set(conflict.owners)
    for item in affected:
        related.update(item.variant_ids)
        related.update(item.definition_ids)
    exact = [_source_row(row) for row in register if row["candidate_id"] in related]
    locations = {loc for row in exact for loc in row["local_evidence_ids"]}
    if catalog:
        locations.add(catalog[0]["id"])
    locations.update(loc for item in affected for loc in item.evidence_ids)
    locations.update(fact.source_id for item in affected for fact in item.b2_source_facts)
    bindings = {"candidate_id", "role", "evidence_ids", "definition_ids", "variant_ids"}
    payload = {"conflict": str(conflict), "affected_selected_ids": sorted(conflict.owners),
               "previous_bindings": [{**p.model_dump(include=bindings),
                                      "b2_source_ids": [f.source_id for f in p.b2_source_facts]} for p in affected],
               "candidate_rows": exact,
               "original_passages": [r for r in catalog if r["id"] in locations]}
    if any(key not in {row["candidate_id"] for row in exact} for key in conflict.owners):
        payload["registered_candidates"] = [{"candidate_id": row["candidate_id"], "source_number": row["source_number"],
                                             "claim": row["claim"][:160], "evidence_ids": row["evidence_ids"]}
                                            for row in register]
    repair = models.structured(LeanCorrection if isinstance(choice, LeanChoice) else SelectionCorrection,
                               rubric + "\n" + fields, payload)
    replacements = [p for p in repair.selected if p.candidate_id in conflict.owners or
                    p.candidate_id not in {q.candidate_id for q in choice.selected}]
    retained = [p for p in choice.selected if p.candidate_id not in conflict.owners]
    return choice.model_copy(update={"selected": retained + replacements})


def select_contributions(catalog, candidates, models, previous_failure=None, rules=None, *, unlimited_current=False,
                         current=False, lean=False, optimized=False, nominations=(), nomination_sources=(), uncertainties=(),
                         resolve_central=None, only_ids=(), checked_images=()):
    register = candidate_register(candidates)
    if not register and not lean:
        return []
    preserved = {}
    if lean and nomination_sources:
        # A review is a nomination, not a new global vote. Previously approved
        # cores outside the requested change retain their identity and decision.
        origins = set(nomination_sources)
        previous = {c["candidate_id"]: c for c in candidates if "candidate_id" in c}
        affected = {d.get("core_id") for d in nominations}
        for row in register:
            key = row["candidate_id"]
            if not only_ids and (key not in previous or (previous[key]["decision"] != "admitted" and
                                        origins.intersection(row["local_evidence_ids"]))):
                affected.add(key)
        affected.update(only_ids)
        preserved = {key: row for key, row in previous.items() if key not in affected}
        register = [r for r in register if r["candidate_id"] not in preserved]
    packet = selection_packet(catalog, register, optimized=optimized)
    readings = {(r["source_id"], r["transcription"]): r for c in register for r in c.get("source_readings", [])}
    readings.update(((r["source_id"], r["transcription"]), r) for r in checked_images)
    if lean and readings:
        packet["checked_source_images"] = list(readings.values())
    if lean and uncertainties:
        packet["unresolved_readings"] = list(dict.fromkeys(uncertainties))
    if lean and nomination_sources:
        located = set(nomination_sources)
        located.update(loc for row in register for loc in row["local_evidence_ids"])
        packet["passages"] = [r for r in catalog if r["id"] in located]
        packet.update(nominations=list(nominations),
                      retained_cores=[{"candidate_id": k, "source_number": row.get("source_number"), "kind": row["kind"], "claim": row["claim"]}
                                      for k, row in preserved.items() if row["decision"] == "admitted" and
                                      row["kind"] in {"current", "historical", "application"}])
    if previous_failure:
        from .retry_context import attach_previous_failure
        attach_previous_failure(packet, previous_failure, catalog)
    if current:
        from .profile_inputs import current_rules, node_prompt
        rubric = current_rules()
        fields, correction = node_prompt("select"), node_prompt("correct_selection")
    else:
        rubric = rules or files("paper_extraction").joinpath("profile_assets", "intro-rules-v13.txt").read_text(encoding="utf-8")
        if unlimited_current:
            rubric += "\n" + COUNT_OVERRIDE
        fields, correction = SELECTION_FIELDS, CORRECTION_FIELDS
    instructions = rubric + "\n" + fields
    missing = {r["id"] for r in catalog} - {r["id"] for r in packet["passages"] if not r.get("abridged")}
    if optimized and missing and not nomination_sources:
        choice = models.structured(SourceSelection, instructions + "\nread_source_ids retrieves the exact full text of any registered location. "
                                   "Request all originals needed to resolve missing conditions, an application or a comparison together before choosing.", packet)
        if choice.read_source_ids:
            requested = set(choice.read_source_ids)
            full = {r["id"]: r for r in projected_sources(catalog, requested)}
            shown = {r["id"]: r for r in packet["passages"]}
            shown.update(full)
            packet = {**packet, "passages": [shown[r["id"]] for r in catalog if r["id"] in shown]}
            choice = models.structured(LeanChoice, instructions, packet)
    else:
        choice = models.structured(LeanChoice if lean else WholePaperSelection, instructions, packet)
    for attempt in range(3):
        try:
            if lean:
                invalid = [item.candidate_id for item in choice.selected
                           if (item.role or next((r["kind"] for r in register if r["candidate_id"] == item.candidate_id), None))
                           in {"current", "historical", "application"} and not (item.independent_question or "").strip()]
                if invalid:
                    raise SelectionConflict("State the independent question or withdraw this formal result", *invalid)
            rows = apply_selection(register, choice, catalog)
            break
        except SelectionConflict as exc:
            if attempt == 2:
                raise
            choice = _repair_choice(exc, choice, register, catalog, models, rubric, correction)
    if lean:
        records = {r["id"]: r for r in catalog}
        for reading in readings.values():
            key = reading["source_id"]
            if key in records:
                records[key] = {**records[key], "text": records[key]["text"] + "\n" + reading["transcription"]}
        selected_rows = {r.candidate_id: r for r in choice.selected}
        for row in rows:
            if row["decision"] != "admitted" or row["kind"] not in {"current", "historical", "application"}:
                continue
            selected_row = selected_rows[row["candidate_id"]]
            role = selected_row.independent_role
            if independent_role_supported(row["kind"], row.get("source_number"), row["claim"],
                                          row.get("local_evidence_ids") or row["evidence_ids"], role, records):
                if role:
                    row["independent_role"] = role.model_dump()
            else:
                row.update(decision="excluded", decision_reason="The supplied source supports the assertion or its proof use, but does not establish its independent role in this paper.")
        rows = sorted([*preserved.values(), *rows], key=lambda row: int(row["candidate_id"][1:]))
        admitted_ids = {r["candidate_id"] for r in rows if r["decision"] == "admitted" and r["kind"] in {"current", "application", "historical"}}
        for row in rows:
            if not row.get("needed_by") or row["kind"] != "definition":
                continue
            row["needed_by"] = [k for k in row["needed_by"] if k in admitted_ids]
            if row["needed_by"]:
                row["decision_reason"] = "Required to state " + ", ".join(row["needed_by"])
            else:
                row.update(decision="excluded", decision_reason="Only required by results outside the final selection.")
        if checked_images:
            for row in rows:
                if row["candidate_id"] in only_ids:
                    row["source_readings"] = [r for r in readings.values() if r["source_id"] in row["evidence_ids"]]
        if resolve_central:
            open_rows = [row for row in rows if row.get("scope_central_open") and row["decision"] == "unresolved"
                         and not row.get("central_checked") and row["candidate_id"] not in preserved]
            checks = []
            checked_ids = []
            for row in open_rows:
                item = _central_check(row, choice.central_checks.get(row["candidate_id"]), records)
                if item:
                    row["central_checked"] = True
                    checks.append(item)
                    checked_ids.append(row["candidate_id"])
            if checks:
                # The selector explicitly marked this independent result as both central
                # and unresolved. Check its actual statement once, then adjudicate only
                # these IDs. No recursion back into the evidence lookup on that pass.
                origins = list(dict.fromkeys(q.source_id for q in checks))
                extra = resolve_central(checks, {key: records[key] for key in origins})
                locs = list(dict.fromkeys([*origins, *(q.target_id for q in checks if q.target_id)]))
                return select_contributions(catalog, rows, models, current=True, lean=True, optimized=optimized, only_ids=checked_ids,
                    nominations=[{"candidate_id": key, "source_question": q.question} for key, q in zip(checked_ids, checks)],
                    nomination_sources=locs, checked_images=extra["source_images"])
        note = choice.summary
        if not note and not any(r["decision"] == "admitted" and r["kind"] in {"current", "application"} for r in rows):
            open_cores = [r for r in rows if r.get("scope_central_open") and r["decision"] == "unresolved"]
            ids = list(dict.fromkeys(i for r in (open_cores or rows) for i in r["evidence_ids"]))
            labels = "、".join(r.get("source_number") or r["candidate_id"] for r in open_cores)
            note = ScreeningNote(kind="insufficient_evidence" if open_cores else "no_qualifying_result",
                question="原文是否给出了值得单列的当前贡献？", evidence_ids=ids or [r["id"] for r in catalog[:1]],
                reason=(f"原文提及 {labels}，现有证据不足以完整核实其条件和结论。" if open_cores else
                        "原文支持所列陈述或其证明用途，但没有确立能够单独列入的主要成果。"))
        return rows, note
    return rows if unlimited_current else reconcile_scope(catalog, rows, models, rubric, previous_failure)


def _central_check(row, requested, records):
    locs = [loc for loc in row["evidence_ids"] if loc in records]
    if requested and requested.source_id in locs and (not requested.target_id or requested.target_id in records):
        return requested
    number = re.search(r"\d+(?:\.\d+)*", row.get("source_number") or "")
    if not number:
        return None
    # Prefer the printed statement itself to the introduction that advertises
    # it or later proofs that invoke it. No search over unrelated source pages.
    heading = re.compile(r"(?im)^\s*(?:theorem|corollary|lemma|proposition|théorème|corollaire|lemme|satz|定理|推论|引理)[\s.~:]*"
                         + re.escape(number[0]) + r"(?!\d|\.\d)")
    located = next((loc for loc in locs if heading.search(records[loc]["text"])), None)
    if not located or not records[located].get("page"):
        return None
    return SourceCheck(source_id=located, read_image=True,
        question=f"Read the complete hypotheses, quantifiers and conclusion of {row['source_number']} on this original page. {row['decision_reason']}")
