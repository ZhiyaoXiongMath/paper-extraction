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


EXTRACTION_GUIDANCE = """Read the original as a mathematical editor. Identify the
setting, exact title and ordered authors, substantive historical comparisons, and
the paper's independent contributions and major applications. For each candidate
separate its core assertion, necessary assumptions and definitions from proof
devices and refinements. Admission requires a substantial solution of a central
problem, a substantial independent theorem or an important independent consequence.
Record the original label, attribution and location. A
historical result may be stated indirectly in a comparison ("known when ..."):
recover its objects, assumptions and conclusion from the surrounding source.
Questions, predictions and statements that a question was emphasized belong to
the motivation. Cite them as such. If the earlier theorem itself is missing, say
which evidence is missing. Treat repeated formulations and proof restatements as
evidence for the same candidate. Judge a separately sourced important consequence
on its own mathematical content."""

DRAFT_GUIDANCE = """Use the original v13 profile and write a selective English
Introduction. State each selected core with exactly the hypotheses, parameter
ranges, quantifiers, objects and strength it needs. Use the paper's exact title and
all authors. Attribute historical statements as reported in this manuscript
unless another source was supplied. A named problem is useful setting; a formal
historical theorem needs a sourced mathematical assertion. Retain major
independent applications, including the full parameter range of a selected
existence consequence. A proof's witness, representative, computation of a
constant or construction detail belongs in the statement only when the selected
assertion depends on its specification. Use numbered definitions for genuine
concepts, local TeX displays for essential formulas, and the original v13 block
and page budgets. Give the source theorem, section or page supporting each formal
block. A source theorem's own statement is an appropriate location. Use
source-supported standard TeX expressions for paper-specific shorthand.

On repair, edit the affected JSON fields and their dependent citations or links.
Preserve the other supported statements and their qualifiers. For a history
block, give an earlier result with hypotheses and conclusion, and attribute it
as reported in the supplied paper. Place a conjecture motivating the paper in
the setting paragraph. References are key/text objects and history and groups
are content fields. The renderer supplies their environments, headings, item
commands, numbering and page breaks."""

REVIEW_GUIDANCE = """Make one editorial decision on this whole draft using v13,
especially A2–A3, B3–B6, C1, D5 and F2. Consult original passages for the title,
authors, historical comparisons, main statements and independent applications.
For a proposed omission, identify the independent question it resolves and the
source evidence of its significance before requesting its inclusion. The
extraction model's admission label alone does not establish editorial importance.
Use additional read-only source tools when the relevant text crosses a passage
boundary or a symbol, attribution or important hypothesis remains unclear.
The candidate register is a reading aid. Decide what was actually established
from the cited manuscript, including historical assertions it explicitly reports.

Judge the selected assertion on its own terms. Check whether its scope is stated
accurately, indispensable assumptions and definitions are present, and its
conclusion preserves the paper's quantifiers, topology, parameters, normalization
and strength. Accept a self-contained supported existence or dimension
consequence without reconstructing the proof's representative, coefficient
formula or witness. The location of the original statement suffices when it
supports the selected claim. An alternative valid historical choice, a paraphrase
or a different ordering within the profile can pass. Keep questions and earlier
authors' emphasis distinct from theorems. A report of a manuscript, or earlier
work described by the original, needs the corresponding attribution.

Return one verdict on the CURRENT JSON draft. The history array already denotes
formal historical blocks. References already denote bibliography items. TeX
wrappers, headings, numbering and page breaks belong to the renderer. Review
their mathematical content, citations and attribution here. A locator to the
supplied paper which reports an earlier result is valid for indirect attribution.

For each issue quote the decisive original wording at its registered location
and the affected draft words, using an empty draft excerpt only for a genuine
omission. Compare the whole current assertion, including its definitions and
citations, with the selected source assertion. State exactly how the two differ
and the smallest edit that repairs the difference. For a changed hypothesis or
quantifier identify a case the two assertions treat differently, accounting for
implications of the other hypotheses. If you cannot substantiate a discrepancy,
leave it out of issues. An existing qualifier or locator needs no reminder.

For a missing result, explain the independent question it resolves and the
source evidence for its importance. If all four slots are occupied, identify
a proof-only or lower-tier item to replace, or a genuine B6 scope conflict.
For a missing detail, show why the selected assertion as written actually
requires it. A possible strengthening, a proof of the consequence, an optional
quantitative refinement of an earlier comparison, or a conjecture proposed for
a history slot does not meet that test. Conversely, correct a conjecture actually
presented as proved or a proof-only result taking an independent theorem slot.

Pass when there are no substantiated defects. Route draft when existing evidence
suffices to fix the affected fields. Route extract for a substantive missing or
misclassified result needing further targeted source extraction, and supply the
original location IDs. Read-only observations, candidate decisions and repair
suggestions from earlier attempts are leads; the current draft and original are
the evidence for this verdict. For a local correction, identify the operation
and give the exact short replacement or insertion. Removal needs the precise
existing excerpt. Use recover for a missing assertion whose original evidence
still needs reading. A verified equivalent formulation belongs to the pass
decision, with no defect entry."""

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

