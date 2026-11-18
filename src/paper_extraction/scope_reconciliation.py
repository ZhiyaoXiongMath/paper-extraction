"""Resolve a provisional v13 scope conflict against the author's original mathematical questions."""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from .editorial import evidence_packet

MAX_CURRENT_GROUPS = 4
SCOPE_EVIDENCE_VERSION = "source-roles-v2"
SCOPE_FIELDS = """Return assessments for requested_ids. candidate_id and retained_id are candidate IDs.
source_id is an ID in original_overview or original_statements_and_context. A supplied source_excerpt is
optional literal text from source_id. linked_clause uses retained_id to identify its current theorem group.
central_open_ids identifies unresolved candidate IDs for the B4 central-admission status."""


class ScopeItem(BaseModel):
    candidate_id: str
    outcome: Literal["admitted", "excluded", "equivalent", "linked_clause", "unresolved"]
    source_id: str
    source_excerpt: str | None = None
    source_reason: str = Field(min_length=1)
    independent_question: str | None = None
    retained_id: str | None = None


class ScopeReconciliation(BaseModel):
    assessments: list[ScopeItem]
    central_open_ids: list[str] = Field(default_factory=list)


def _literal_excerpt(row, proposed):
    if not proposed or row.get("transcribed_from_image"):
        return None
    pieces = proposed.strip().split()
    if pieces:
        exact = re.search(r"\s+".join(re.escape(piece) for piece in pieces), row["text"])
        if exact:
            return exact.group()


def _source_packet(catalog, current):
    original = evidence_packet(catalog, [], max_chars=24000)["passages"]
    selected_ids = {loc for r in current for loc in (r["selection_evidence_ids"] + r["local_evidence_ids"])}
    head_ids = {r["id"] for r in original}
    proposals = [{k: r[k] for k in ("candidate_id", "source_number", "kind", "claim", "assumptions",
                                      "selection_evidence_ids", "local_evidence_ids")}
                 for r in current]
    return {"original_overview": [r for r in catalog if r["id"] in head_ids],
            "original_statements_and_context": [r for r in catalog if r["id"] in selected_ids - head_ids],
            "candidate_rows": proposals, "proposed_current_group_count": len(current),
            "standard_current_group_max": MAX_CURRENT_GROUPS}


def reconcile_scope(catalog, rows, models, rubric, previous_failure=None):
    current = [row for row in rows if row["decision"] == "admitted" and row["kind"] in {"current", "application"}]
    if len(current) <= MAX_CURRENT_GROUPS:
        return rows
    packet = _source_packet(catalog, current)
    if previous_failure:
        from .retry_context import attach_previous_failure
        attach_previous_failure(packet, previous_failure, catalog, "original_statements_and_context")
    supplied = {r["id"] for field in ("original_overview", "original_statements_and_context")
                for r in packet[field]}
    original = {r["id"]: r for r in catalog if r["id"] in supplied}
    positions = {r["candidate_id"]: r for r in current}
    decisions = {}
    pending = list(positions)
    central_open = set()
    for attempt in range(2):
        response = models.structured(ScopeReconciliation, rubric + "\n" + SCOPE_FIELDS, {**packet, "requested_ids": pending})
        central_open.difference_update(pending)
        central_open.update(k for k in response.central_open_ids if k in pending)
        problems = {}
        for item in response.assessments:
            key = item.candidate_id
            if key not in pending:
                continue
            if item.source_id not in original:
                problems[key] = f"{item.outcome}: source ID {item.source_id!r} was not supplied. Cite an original passage in this packet."
            elif item.outcome == "admitted" and not (item.independent_question or "").strip():
                problems[key] = f"admitted at {item.source_id}: independent_question is missing."
            else:
                decisions[key] = item
        for key in positions:
            if key not in decisions:
                problems.setdefault(key, "No assessment or source ID was returned for this candidate.")
        for key, item in decisions.items():
            owner = decisions.get(item.retained_id)
            if item.outcome in {"equivalent", "linked_clause"} and (owner is None or owner.outcome != "admitted" or item.retained_id == key):
                explanation = f"{item.outcome} at {item.source_id}: retained_id {item.retained_id or '[missing]'} does not reference a distinct admitted row."
                if not attempt:
                    problems[key] = explanation + " Correct retained_id or outcome."
                elif key not in problems:
                    decisions[key] = item.model_copy(update={"outcome": "unresolved", "retained_id": None,
                        "source_reason": explanation})
        if not problems:
            break
        if attempt:
            detail = " | ".join(f"{key}: {why}" for key, why in problems.items())
            raise ValueError("Scope reconciliation could not bind its assessments to the original: " + detail)
        pending = list(problems)
        bindings = {"candidate_id", "outcome", "source_id", "retained_id"}
        packet = {**packet, "corrections": problems,
                  "previous_bindings": [r.model_dump(include=bindings) for r in response.assessments]}

    retained = {k for k, v in decisions.items() if v.outcome in {"admitted", "linked_clause"}}
    representative = {k: v.retained_id for k, v in decisions.items() if v.outcome == "equivalent"}
    for key, item in decisions.items():
        row = positions[key]
        row.update(decision_reason=item.source_reason, scope_evidence_id=item.source_id, scope_role=item.outcome,
                   scope_central_open=(item.outcome == "unresolved" and key in central_open),
                   scope_reason_type="paraphrase", scope_evidence_version=SCOPE_EVIDENCE_VERSION)
        quote = _literal_excerpt(original[item.source_id], item.source_excerpt)
        if quote:
            row["scope_excerpt"] = quote
        else:
            row.pop("scope_excerpt", None)
        if item.outcome in {"admitted", "linked_clause"}:
            row["scope_group_id"] = item.retained_id if item.outcome == "linked_clause" else key
            row["independent_question"] = item.independent_question or row.get("independent_question")
            row["selection_evidence_ids"] = list(dict.fromkeys(row["selection_evidence_ids"] + [item.source_id]))
            row["evidence_ids"] = list(dict.fromkeys(row["evidence_ids"] + [item.source_id]))
        else:
            row["decision"] = "unresolved" if item.outcome == "unresolved" else "excluded"
            if item.outcome == "equivalent":
                row["represented_by"] = item.retained_id
            if item.outcome == "excluded" and item.retained_id in retained:
                row["context_for"] = [item.retained_id]
    for equivalent, owner_id in representative.items():
        owner = positions[owner_id]
        loc = decisions[equivalent].source_id
        owner["evidence_ids"] = list(dict.fromkeys(owner["evidence_ids"] + [loc]))
        owner["selection_evidence_ids"] = list(dict.fromkeys(owner["selection_evidence_ids"] + [loc]))
    for row in rows:
        # Keep shared or historical definitions. Release definitions needed only
        # by proof ingredients or inherit their owners when cores were equivalent.
        if "needed_by" in row:
            owners = [representative.get(key, key) for key in row["needed_by"]
                      if key not in positions or key in retained or key in representative]
            row["needed_by"] = list(dict.fromkeys(owners))
            if not owners and row["kind"] == "definition":
                row.update(decision="excluded", decision_reason="The definition served only provisional results omitted after source reconciliation.")
        owner = row.get("represented_by")
        if owner in representative:
            row["represented_by"] = representative[owner]
        elif owner in positions and owner not in retained:
            row.pop("represented_by", None)
            row.update(decision=positions[owner]["decision"],
                       decision_reason=f"Alternate formulation of {owner}. " + positions[owner]["decision_reason"])
    return rows


def scope_failure_reason(candidates, *, unlimited_current=False):
    current = [row for row in candidates if row["decision"] == "admitted" and row["kind"] in {"current", "application"}]
    if not current:
        unsettled = [r for r in candidates if r.get("scope_central_open")]
        if unsettled:
            details = [f'{r.get("source_number") or r["candidate_id"]} [{r["scope_evidence_id"]}] {r["decision_reason"]}' for r in unsettled]
            return "v13 B4：当前核心成果的独立准入仍未确定。\n" + "\n".join(details)
    if unlimited_current:
        return ""
    groups = {r.get("scope_group_id") or r["candidate_id"] for r in current}
    if len(groups) <= MAX_CURRENT_GROUPS:
        return ""
    details = [f'{r.get("source_number") or r["candidate_id"]} [{r["scope_evidence_id"]}] — {r["independent_question"]} {r["decision_reason"]}'
               for r in current]
    return f"v13 范围例外：原文所选陈述需 {len(groups)} 个当前定理组，超出标准稿的四项上限。\n" + "\n".join(details)



def stale_scope_result(result):
    current = [r for r in result.get("candidates", []) if r.get("decision") == "admitted"
               and r.get("kind") in {"current", "application"} and r.get("scope_role") in {"independent", "admitted"}]
    return len(current) > MAX_CURRENT_GROUPS and any(r.get("scope_evidence_version") != SCOPE_EVIDENCE_VERSION for r in current)


def saved_count_exception(result, expected_selection):
    """Reuse source-checked candidates when a saved draft was blocked only by the old count limit."""
    selected = result.get("candidates", [])
    return (result.get("status") == "failed" and not result.get("draft") and
            result.get("selection_version") == expected_selection and
            result.get("reason", "").startswith("v13 范围例外：") and
            any(r.get("kind") == "historical" and r.get("decision") == "admitted" for r in selected) and
            len({r.get("scope_group_id") or r.get("candidate_id") for r in selected if r.get("decision") == "admitted"
                 and r.get("kind") in {"current", "application"}}) > MAX_CURRENT_GROUPS)
