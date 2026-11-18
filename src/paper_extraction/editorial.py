"""Build a compact, primary-source packet for one v13 editorial decision."""
from __future__ import annotations

from collections import defaultdict
from itertools import zip_longest
import re


EDIT_CONTRACT_VERSION = "actionable-edits-v1"
_NO_EDIT = re.compile(
    r"(?:^|[.;:,!?()]\s*)(?:there (?:is|are) )?"
    r"(?:no (?:actionable |actual |mathematical |visible |visual |layout )?(?:defects?|issues?|errors?)"
    r"(?: (?:remains?|found))?|no (?:edits?|changes?|corrections?|revisions?)"
    r"(?: (?:is |are )?(?:needed|required|necessary))?|(?:leave|keep) (?:it |this |the draft )?unchanged)"
    r"(?=\s*(?:[.;:,!?()]|$))", re.I)
_CONFIRMATION_CLAUSE = re.compile(
    r"(?:equivalent|(?:the (?:two )?(?:assertions|statements|formulations) are equivalent)"
    r"(?: and the draft (?:copies|matches|agrees with) the source)?"
    r"|the (?:wording|phrase|phrases|notation)(?: <quoted>(?: and <quoted>)*)? "
    r"(?:do|does) not (?:alter|change) (?:the|this) (?:claim|meaning|statement)"
    r"|(?:retained|included|listed) only to (?:record|note) that (?:it|this) was checked)", re.I)


def says_no_edit(text: str) -> bool:
    """Explicit self-contradiction in a proposed edit, not an inference of correctness."""
    return bool(_NO_EDIT.search(text))


def confirmation_only(text: str) -> bool:
    """Recognize only unambiguously affirmative clauses in an old rejected verdict."""
    plain = re.sub(r'"[^"]*"|“[^”]*”', "<quoted>", text)
    clauses = [p.strip(" ()\n\t") for p in re.split(r"[.;!?]+", plain)]
    return says_no_edit(plain) and all(
        not p or _NO_EDIT.fullmatch(p) or _CONFIRMATION_CLAUSE.fullmatch(p) for p in clauses)


EXTRACTION_GUIDANCE = (
    "For each entry in candidates, fill kind, source_number, claim, assumptions, dependencies, "
    "significance, decision, decision_reason and evidence_ids. Copy evidence_ids from the supplied "
    "source records. Use null for an unnumbered source statement. Record source uncertainties in uncertainties."
)

DRAFT_GUIDANCE = (
    "Populate the Draft fields title, authors, paragraph, history, shared_definitions, groups, "
    "references and evidence_links. Use TeX fragments for block text. Use the key source for the "
    "supplied paper. For each block, pair its block_id with registered evidence_ids. "
    "The renderer supplies environments, numbering and bibliography items."
)

REVIEW_GUIDANCE = (
    "Populate route, issues and evidence_ids. route names the next graph edge: pass, extract or draft. "
    "For each reported issue, identify its draft_field and source evidence_id. Use current_excerpt and "
    "source_excerpt for the corresponding passages, and required_change for the edit, referring to the "
    "applicable v13 clause. Use operation and replacement_text to encode the proposed edit. "
    "For pass, return empty issues and evidence_ids."
)


def reversed_edit_requests(reviews, fields=None):
    """Find opposite structural edits requested against the same original passage."""
    latest = {}
    for review in reviews:
        if review.get("stage") != "review":
            continue
        for issue in review.get("defects", []):
            if not isinstance(issue, dict) or issue.get("operation") not in {"insert", "remove"}:
                continue
            field, source = issue.get("draft_field"), issue.get("evidence_id")
            if not field or not source or (fields is not None and field not in fields):
                continue
            latest.setdefault((field, source), {})[issue["operation"]] = {
                "round": review.get("attempt"), "operation": issue["operation"], "requested_change": issue.get("required_change", "")}
    return [{"draft_field": field, "source_id": source, "prior_requests": sorted(operations.values(), key=lambda row: row["round"] or 0)}
            for (field, source), operations in latest.items() if len(operations) == 2]


_HISTORY = re.compile(
    r"\b(?:previously|earlier|formerly|classical|was known|known (?:when|under|for|in)|"
    r"showed that|proved that|established that|asserted that|attributed to|due to|"
    r"generalizes? (?:a |the |results?)|theorem of|result of)\b", re.I)
_MAIN = re.compile(
    r"\b(?:main (?:theorem|results?)|principal (?:theorem|results?)|we "
    r"(?:prove|establish|show|characterize|construct)|our (?:result|theorem))\b", re.I)
_APPLICATION = re.compile(
    r"\b(?:as (?:a|an) (?:direct |important )?(?:consequence|application)|"
    r"applications?|corollary\s+[\dA-Z])\b|\\begin\s*\{(?:corollary|cor)\}", re.I)
_STATEMENT = re.compile(r"(?m)^\s*(?:Theorem|Proposition)\s+[\dA-Z]|\\begin\s*\{(?:theorem|thm)\}", re.I)
_INTRO = re.compile(r"\b(?:abstract|introduction|background|historical|context)\b", re.I)
_PROOF_SECTION = re.compile(r"\b(?:proof|construction|estimate|calculation)\b", re.I)


def _distinct(values):
    return list(dict.fromkeys(values))


def _normalized_label(label):
    value = " ".join((label or "").strip().casefold().split())
    return re.sub(r"^(thm|prop|cor|defn|lem)\.?\s+", lambda m: {
        "thm": "theorem", "prop": "proposition", "cor": "corollary",
        "defn": "definition", "lem": "lemma",
    }[m[1]] + " ", value)


def _candidate_register(candidates):
    """Pool repeated locations without equating differently qualified assertions."""
    register = []
    seen = {}
    by_label = {}
    for original in candidates:
        row = {**original, "evidence_ids": _distinct(original.get("evidence_ids", []))}
        label = _normalized_label(row.get("source_number"))
        # Mathematical case and word boundaries are significant.
        signature = (row.get("kind"), row.get("decision"), label,
                     " ".join(row.get("claim", "").split()), " ".join(row.get("assumptions", "").split()))
        if signature in seen:
            previous, group = seen[signature]
            previous["evidence_ids"] = _distinct(previous["evidence_ids"] + row["evidence_ids"])
            previous["dependencies"] = _distinct(previous.get("dependencies", []) + row.get("dependencies", []))
            if group is not previous:
                group["evidence_ids"] = _distinct(group["evidence_ids"] + row["evidence_ids"])
            continue
        # Group repeated numbered current statements while retaining distinct clauses.
        # A label alone is ambiguous for prior works by different authors.
        numbered = re.fullmatch(r"(?:theorem|proposition|corollary|lemma|definition)\s+[\da-z][\da-z.-]*", label)
        group_key = (row.get("kind"), row.get("decision"), label)
        if numbered and row.get("kind") in {"current", "application", "definition"}:
            if group_key in by_label:
                group = by_label[group_key]
                variant = {k: v for k, v in row.items() if k not in {"kind", "decision", "source_number"}}
                group.setdefault("other_reported_clauses", []).append(variant)
                group["evidence_ids"] = _distinct(group["evidence_ids"] + variant["evidence_ids"])
                seen[signature] = (variant, group)
                continue
            by_label[group_key] = row
        register.append(row)
        seen[signature] = (row, row)
    return register


def evidence_packet(catalog, candidates, draft=None, *, max_chars=48000):
    """Return original passages, candidate register and draft for one model context.

    The text budget samples optional surrounding passages. The first source passage
    and all registered locations cited by the draft are always included verbatim.
    Additional locations stay available through the paper's read-only tools.
    """
    records = {str(r["id"]): r for r in catalog}
    available = list(records.values())
    register = _candidate_register(candidates)
    links = (draft or {}).get("evidence_links", [])
    bound = _distinct(str(loc) for link in links for loc in link.get("evidence_ids", []))
    required = _distinct(([str(available[0]["id"])] if available else []) + bound)
    picked = {loc for loc in required if loc in records}
    used = sum(len(records[loc].get("text", "")) for loc in picked)

    kinds = defaultdict(set)
    for row in register:
        if row.get("decision") != "excluded":
            for loc in row["evidence_ids"]:
                kinds[str(loc)].add(row.get("kind"))
    lanes = {k: [] for k in ("opening", "history", "main", "application")}
    for order, r in enumerate(available):
        loc = str(r["id"])
        if loc in picked:
            continue
        text = r.get("text", "")
        section = r.get("section", "")
        opening = order < 2 or r.get("page") == 1 or bool(_INTRO.search(section))
        proof = bool(_PROOF_SECTION.search(section))
        if opening or "setting" in kinds[loc]:
            lanes["opening"].append((0 if order < 2 or r.get("page") == 1 else 1, order, loc))
        for lane, kind, pattern in (("history", "historical", _HISTORY),
                                    ("main", "current", _MAIN),
                                    ("application", "application", _APPLICATION)):
            explicit = bool(pattern.search(text) or pattern.search(section))
            source_statement = lane == "main" and not proof and bool(_STATEMENT.search(text))
            if kind in kinds[loc] or explicit or source_statement:
                priority = (0 if explicit and opening else 1 if kind in kinds[loc] and explicit
                            else 2 if kind in kinds[loc] else 3 if explicit else 4)
                lanes[lane].append((priority + int(proof), order, loc))
    # Draw from each purpose so dense theorem or introductory sections cannot
    # consume all room before a historical comparison or a late application.
    queues = [[loc for _, _, loc in sorted(items)] for items in lanes.values()]
    for group in zip_longest(*queues):
        for loc in group:
            if loc is None or loc in picked:
                continue
            cost = len(records[loc].get("text", ""))
            if used + cost <= max_chars:
                picked.add(loc)
                used += cost
    packet = {"passages": [r for r in available if str(r["id"]) in picked],
              "candidates": register, "unregistered_ids": [loc for loc in bound if loc not in records]}
    if draft is not None:
        packet["draft"] = draft
    return packet

