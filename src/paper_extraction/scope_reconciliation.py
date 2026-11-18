"""Resolve a provisional v13 scope conflict against the author's original mathematical questions."""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from .editorial import evidence_packet

MAX_CURRENT_GROUPS = 4
SCOPE_EVIDENCE_VERSION = "source-roles-v2"
_ORGANIZATION = re.compile(
    r"\b(?:the (?:paper|article|proof) is (?:organized|organised|structured)|"
    r"outline of the (?:paper|proof)|in [Ss]ection\s+(?:\d+|[IVX]+)\s+we|"
    r"[Ss]ection\s+\d+\s+(?:collects|constructs|develops|provides|establishes))\b", re.I)


class ScopeItem(BaseModel):
    candidate_id: str
    outcome: Literal["independent", "ingredient", "equivalent", "unresolved"]
    source_id: str = Field(description="An original passage in the supplied packet.")
    source_excerpt: str | None = Field(default=None, description="Optional short verbatim phrase from the cited passage; omit if uncertain or transcribed.")
    source_reason: str = Field(min_length=1, description="Explain what the source establishes about the role and significance.")
    independent_question: str | None = Field(default=None, description="For an independent result, name the mathematical question it resolves.")
    retained_id: str | None = Field(default=None, description="For an equivalent formulation, identify the retained independent core. Optional for an ingredient.")


class ScopeReconciliation(BaseModel):
    assessments: list[ScopeItem]


GUIDANCE = """Reconcile the provisional over-budget selection within extraction under v13 B1–B6.
Read the original opening, motivation, author comparisons, structure of the proof and main results or
applications BEFORE the previously proposed questions. Those proposals are hypotheses to check.
Identify what recognizable question the paper sets out to answer and which other questions receive
a substantial independent answer or consequence. Turning an equation, lemma or estimate into a yes/no
question does not demonstrate independent mathematical importance.
For each possibly auxiliary result look for POSITIVE original evidence of a separately motivated problem,
a comparison it advances, a result on a recognizable class with independent significance, or an actual
important application. Check the surrounding text for how the author uses it. A standard estimate,
coordinate representation, intermediate existence assertion, inverse acting on an earlier construction,
or conditional final step used to prove the central theorem can be a necessary proof ingredient.
A generic usefulness claim, the numbering of the assertion, the ability to state it separately,
or simply quoting its formula cannot justify occupying an independent contribution slot. A section preview
or statement of an equivalence may show that the claim appears in the paper. Read the surrounding purpose and
its downstream uses. If a criterion only reformulates a hypothesis needed by another selected result, being
presented in its own section and enabling that result are insufficient evidence of separate importance. For a candidate
used in the proof, explain the positive source evidence for significance beyond that role before retaining
it independently. This evidence may be an explicit independently important mathematical consequence in
the source itself; the author need not label it important or mention it in the introduction.
Use 'ingredient' when source context establishes a purely supporting role; 'equivalent' for a duplicate
formulation of a retained core; 'unresolved' if the supplied source does not establish significance.
Do not weaken a retained theorem, discard an important application, or aim for any particular count.
Five or more genuinely independent contributions must all survive. Return one assessment per requested ID.
For each assessment supply the original source ID and paraphrase the decisive source fact. A short literal
excerpt is optional. For an independent item explain the connection between the source fact and its own question.
A theorem label, an announcement that a criterion will be proved, or the criterion's use to state another main
theorem cannot by themselves demonstrate a separate major consequence."""


def _literal_excerpt(row, proposed):
    if not proposed or row.get("transcribed_from_image"):
        return None
    pieces = proposed.strip().split()
    if pieces:
        exact = re.search(r"\s+".join(re.escape(piece) for piece in pieces), row["text"])
        if exact:
            return exact.group()


_INVOKED = re.compile(r"\b(?:apply|applying|using|iterate|iterating|by|invoking|from|combine)\s+(?:(?:theorem|proposition|lemma|corollary)\s*)?$", re.I)
_ROLE = re.compile(r"\b(?:iterate[ds]?|iteration|iterative step|apply|applying|application of|"
                   r"used (?:in|to|only)|using|by (?:theorem|proposition|lemma)|"
                   r"proof of|it suffices|we first (?:prove|construct)|as a consequence)\b", re.I)


def organization_context(catalog):
    end = max(4, min(18, len(catalog) // 5))
    passages = []
    for row in catalog[:end]:
        match = _ORGANIZATION.search(row["text"]) or _ROLE.search(row["text"])
        if match:
            passages.append({"id": row["id"], "text": row["text"][max(0, match.start() - 150):match.end() + 750]})
    return passages[:5]


def _source_uses(catalog, current, *, limit=4, radius=280):
    uses = []
    for item in current:
        number = item.get("source_number") or ""
        labels = re.findall(r"[\w]+(?:[.:-][\w]+)+", number)
        terms = [r"\\(?:auto|c|C|eq)?ref\s*\{[^{}]*?(?<![\w.:/-])" + re.escape(label) + r"(?![\w.:/-])[^{}]*\}"
                 for label in labels]
        named = re.search(r"\b(theorem|proposition|corollary|lemma)\s+([A-Z]|\d+(?:\.\d+)*)\b", number, re.I)
        if named:
            terms.append(r"\b" + named[1] + r"\s+" + re.escape(named[2]) + r"(?!\w|\.\d)")
        bound = set(item.get("local_evidence_ids", item.get("evidence_ids", [])))
        references = []
        pattern = re.compile("|".join(terms), re.I) if terms else None
        for order, row in enumerate(catalog):
            t = row["text"]
            occurrences = list(pattern.finditer(t)) if pattern else []
            if not occurrences and row["id"] in bound:
                match = _ROLE.search(t)
                if match:
                    occurrences = [match]
            for match in occurrences:
                snippet = t[max(0, match.start() - radius):match.end() + radius + 35]
                indicative = bool(_ROLE.search(t[max(0, match.start() - 140):match.end() + 140]))
                numbered_use = bool(pattern and pattern.fullmatch(match.group()))
                invoked = numbered_use and bool(_INVOKED.search(t[max(0, match.start()-65):match.start()]))
                priority = (int(not invoked), int(not (indicative and numbered_use)), int(not numbered_use), int(not indicative),
                            int(row["id"] in bound), order, match.start())
                references.append((priority, row["id"], snippet))
        picked = set()
        for _, loc, text in sorted(references):
            if loc not in picked:
                uses.append({"candidate_id": item["candidate_id"], "id": loc, "text": text})
                picked.add(loc)
            if len(picked) == limit:
                break
    return uses


def primary_uses(catalog, register):
    # Several chunk observations often identify the same numbered source statement.
    # Share its actual use passage so all those navigators fit one model context.
    numbered, owners = {}, {}
    for row in register:
        if row["kind"] not in {"current", "application", "historical"}:
            continue
        key = (row.get("source_number") or "", row["kind"])
        if not key[0] or row["kind"] == "historical":
            key = (*key, tuple(row["evidence_ids"]))
        if key not in numbered:
            numbered[key] = {**row, "local_evidence_ids": list(row["evidence_ids"])}
            owners[row["candidate_id"]] = []
        anchor = numbered[key]
        anchor["local_evidence_ids"] = list(dict.fromkeys(anchor["local_evidence_ids"] + row["evidence_ids"]))
        owners[anchor["candidate_id"]].append(row["candidate_id"])
    radius = max(80, min(240, 12000 // max(1, len(numbered))))
    return [{"candidate_ids": owners[use["candidate_id"]], "id": use["id"], "text": use["text"]}
            for use in _source_uses(catalog, list(numbered.values()), limit=1, radius=radius)]


def _source_packet(catalog, current):
    original = evidence_packet(catalog, [], max_chars=24000)["passages"]
    selected_ids = {loc for r in current for loc in (r["selection_evidence_ids"] + r["local_evidence_ids"])}
    head_ids = {r["id"] for r in original}
    head_ids.update(r["id"] for r in organization_context(catalog))
    proposals = [{k: r[k] for k in ("candidate_id", "source_number", "kind", "claim", "assumptions",
                                      "independent_question", "decision_reason", "selection_evidence_ids", "local_evidence_ids")}
                 for r in current]
    return {"original_overview": [r for r in catalog if r["id"] in head_ids],
            "original_statements_and_context": [r for r in catalog if r["id"] in selected_ids - head_ids],
            "original_uses": _source_uses(catalog, current), "provisional_contributions": proposals}


def reconcile_scope(catalog, rows, models, rubric):
    current = [row for row in rows if row["decision"] == "admitted" and row["kind"] in {"current", "application"}]
    if len(current) <= MAX_CURRENT_GROUPS:
        return rows
    packet = _source_packet(catalog, current)
    supplied = {r["id"] for field in ("original_overview", "original_statements_and_context", "original_uses")
                for r in packet[field]}
    original = {r["id"]: r for r in catalog if r["id"] in supplied}
    positions = {r["candidate_id"]: r for r in current}
    decisions = {}
    pending = list(positions)
    for attempt in range(2):
        response = models.structured(ScopeReconciliation, rubric + "\n" + GUIDANCE, {**packet, "requested_ids": pending})
        problems = {}
        for item in response.assessments:
            key = item.candidate_id
            if key not in pending:
                continue
            if item.source_id not in original:
                problems[key] = f"{item.outcome}: source ID {item.source_id!r} was not supplied. Cite an original passage in this packet."
            elif item.outcome == "independent" and not (item.independent_question or "").strip():
                problems[key] = f"independent at {item.source_id}: the mathematical question resolved is missing."
            else:
                decisions[key] = item
        for key in positions:
            if key not in decisions:
                problems.setdefault(key, "No assessment or source ID was returned for this candidate.")
        for key, item in decisions.items():
            owner = decisions.get(item.retained_id)
            if item.outcome == "equivalent" and (owner is None or owner.outcome != "independent" or item.retained_id == key):
                explanation = f"equivalent at {item.source_id}: {item.retained_id or '[missing]'} is not a retained independent representative."
                if not attempt:
                    problems[key] = explanation + " Identify its representative or correct its role."
                elif key not in problems:
                    decisions[key] = item.model_copy(update={"outcome": "unresolved", "retained_id": None,
                        "source_reason": explanation + " Proposed relationship: " + item.source_reason})
        if not problems:
            break
        if attempt:
            detail = " | ".join(f"{key}: {why}" for key, why in problems.items())
            raise ValueError("Scope reconciliation could not bind its assessments to the original: " + detail)
        pending = list(problems)
        packet = {**packet, "corrections": problems, "previous_assessments": [r.model_dump() for r in response.assessments]}

    retained = {k for k, v in decisions.items() if v.outcome == "independent"}
    representative = {k: v.retained_id for k, v in decisions.items() if v.outcome == "equivalent"}
    for key, item in decisions.items():
        row = positions[key]
        row.update(decision_reason=item.source_reason, scope_evidence_id=item.source_id, scope_role=item.outcome,
                   scope_reason_type="paraphrase", scope_evidence_version=SCOPE_EVIDENCE_VERSION)
        quote = _literal_excerpt(original[item.source_id], item.source_excerpt)
        if quote:
            row["scope_excerpt"] = quote
        else:
            row.pop("scope_excerpt", None)
        if item.outcome == "independent":
            row["independent_question"] = item.independent_question
            row["selection_evidence_ids"] = list(dict.fromkeys(row["selection_evidence_ids"] + [item.source_id]))
            row["evidence_ids"] = list(dict.fromkeys(row["evidence_ids"] + [item.source_id]))
        else:
            row["decision"] = "unresolved" if item.outcome == "unresolved" else "excluded"
            if item.outcome == "equivalent":
                row["represented_by"] = item.retained_id
            if item.outcome == "ingredient" and item.retained_id in retained:
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


def scope_failure_reason(candidates):
    current = [row for row in candidates if row["decision"] == "admitted" and row["kind"] in {"current", "application"}]
    if not current:
        unsettled = [r for r in candidates if r.get("scope_role") == "unresolved"]
        if unsettled:
            details = [f'{r.get("source_number") or r["candidate_id"]} [{r["scope_evidence_id"]}] {r["decision_reason"]}' for r in unsettled]
            return "v13 B4：当前核心成果的独立准入仍未确定。\n" + "\n".join(details)
    if len(current) <= MAX_CURRENT_GROUPS:
        return ""
    details = [f'{r.get("source_number") or r["candidate_id"]} [{r["scope_evidence_id"]}] — {r["independent_question"]} {r["decision_reason"]}'
               for r in current]
    return f"v13 范围例外：原文支持 {len(current)} 项独立贡献，超出标准稿的四项上限。\n" + "\n".join(details)



def stale_scope_result(result):
    current = [r for r in result.get("candidates", []) if r.get("decision") == "admitted"
               and r.get("kind") in {"current", "application"} and r.get("scope_role") == "independent"]
    return len(current) > MAX_CURRENT_GROUPS and any(r.get("scope_evidence_version") != SCOPE_EVIDENCE_VERSION for r in current)
