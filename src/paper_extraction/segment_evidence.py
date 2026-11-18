"""Register principal leads in a long source and resolve only outstanding readings."""
from __future__ import annotations

import hashlib
import re
from typing import ClassVar, Literal

from pydantic import BaseModel, Field, ValidationInfo, model_validator

from .models import Candidate, Findings, SourceCheck, json_text
from .sources import uncomment


SCAN_INSTRUCTIONS = """Register leads for whole-paper selection. Read all supplied passages, including
results and applications first stated outside the introduction. Register principal advances,
independent applications/counterexamples and the history or definitions needed to understand them.
Proof ingredients and standard cases stay retrievable in the source index; omit a separate row
unless the paper gives a separate contribution. Preserve scope, exceptions and original labels.
Use a short identification of the assertion and its gain, without repeating them across fields.
Read paper_identity as supplied front matter. Return a question only when its answer can change
a retained assertion, condition, dependency, source or eligibility. Attach its candidate index
(zero-based) and originating passage. For a missing central assertion that cannot yet be
identified use a null index. Distinguish the subject of each uncertainty (e.g. exponent versus
sign). Use an exact source phrase or TeX label for lookup, or request the specific PDF image.
If no further source can answer it, omit the lookup and state the remaining uncertainty.
If these passages settle a question in previously_open, return its ID, finding and source
in closed_readings. A similar-looking claim with different hypotheses cannot close it.
Keep a verified partial assertion when another clause is genuinely unresolved. Empty results are valid."""

CHECK_INSTRUCTIONS = """Resolve the listed questions from the supplied source excerpts and images.
Return one answer per question ID. Cite source IDs for a resolved answer; if the evidence does
not settle it, state exactly what remains open. Supply edits only to the named affected
candidates and only to fields that must change. If a verified reading adds or changes a
hypothesis, formula or mathematical range, edit that field so the registered statement itself
contains it. Retain all other qualifiers and exceptions.
For a genuinely separate contribution newly located in these excerpts, return a short new lead."""


class Lead(BaseModel):
    kind: Literal["current", "application", "historical", "definition", "setting"]
    source_number: str | None = None
    claim: str = Field(min_length=1)
    assumptions: str = ""
    dependencies: list[str] = Field(default_factory=list)
    gain: str = Field(min_length=1, description="Source-grounded advance or concrete use of this definition/comparison.")
    evidence_ids: list[str] = Field(min_length=1)

    def candidate(self, pending):
        return Candidate(**self.model_dump(exclude={"gain"}), significance=self.gain,
                         decision="unresolved" if pending else "admitted",
                         decision_reason="; ".join(pending) if pending else self.gain)


class ReadingQuestion(BaseModel):
    candidate_index: int | None = Field(default=None, ge=0)
    source_id: str
    field: Literal["claim", "assumptions", "definition", "role", "paper_authors", "paper_title", "tex_label"]
    subject: str = Field(min_length=1, description="Exact symbol, condition, paper metadata field or locator at issue.")
    question: str = Field(min_length=1)
    target_id: str | None = None
    query: str | None = None
    read_image: bool = False
    region: tuple[float, float, float, float] | None = None

    def lookup(self):
        if not (self.target_id or self.query or self.read_image or self.region):
            return None
        return SourceCheck.model_validate(self.model_dump())

    def key(self, lead):
        # Exclude the model's paraphrase of the question. Keep case-sensitive
        # mathematics, scope and target so different hypotheses cannot coalesce.
        identity = [lead.model_dump(exclude={"gain", "evidence_ids"}) if lead else None, self.source_id, self.field,
                    " ".join(self.subject.split()), self.target_id, self.query, self.region, self.read_image]
        return hashlib.sha256(json_text(identity).encode()).hexdigest()


class ReadingAnswer(BaseModel):
    question_id: str
    status: Literal["resolved", "unresolved"]
    finding: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class SegmentLeads(BaseModel):
    candidates: list[Lead]
    questions: list[ReadingQuestion] = Field(default_factory=list)
    closed_readings: list[ReadingAnswer] = Field(default_factory=list)
    retry_with_previous: ClassVar[bool] = True

    @model_validator(mode="after")
    def check_sources(self, info: ValidationInfo):
        origins = {r["id"] for r in info.context["passages"]} if info.context is not None else None
        if origins is not None and any(not set(c.evidence_ids) <= origins for c in self.candidates):
            raise ValueError("Use only supplied passage IDs for registered leads.")
        for question in self.questions:
            if question.candidate_index is not None and question.candidate_index >= len(self.candidates):
                raise ValueError("The reading question must name an existing zero-based candidate index.")
            if origins is not None and question.source_id not in origins:
                raise ValueError("The reading question must originate in a supplied passage.")
            question.lookup()
        if info.context is not None:
            requested = {r["question_id"] for r in info.context.get("previously_open", [])}
            closed = [r.question_id for r in self.closed_readings]
            if len(closed) != len(set(closed)) or not set(closed) <= requested:
                raise ValueError("Only close an outstanding question supplied in previously_open.")
            if any(r.status != "resolved" or not r.evidence_ids or not set(r.evidence_ids) <= origins
                   for r in self.closed_readings):
                raise ValueError("To close a prior question, cite the supplied passage that settles it.")
        return self


class LeadEdit(BaseModel):
    candidate_index: int = Field(ge=0)
    claim: str | None = Field(default=None, min_length=1)
    assumptions: str | None = None
    dependencies: list[str] | None = None
    gain: str | None = Field(default=None, min_length=1)
    source_number: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class ReadingAnswers(BaseModel):
    retry_with_previous: ClassVar[bool] = True
    answers: list[ReadingAnswer]
    edits: list[LeadEdit] = Field(default_factory=list)
    additions: list[Lead] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_answers(self, info: ValidationInfo):
        if info.context is None:
            return self
        packet = info.context
        asked = {row["question_id"] for row in packet["source_questions"]}
        ids = [a.question_id for a in self.answers]
        if set(ids) != asked or len(ids) != len(asked):
            raise ValueError("Return one resolved or unresolved answer for each supplied question_id.")
        locations = {r["id"] for r in packet["passages"]}
        locations.update(r["source_id"] for r in packet["source_images"])
        for item in [*self.answers, *self.additions, *self.edits]:
            if not set(item.evidence_ids) <= locations:
                raise ValueError("Only cite evidence supplied for these questions.")
        if any(a.status == "resolved" and not a.evidence_ids for a in self.answers):
            raise ValueError("Name the source that resolves each answered question.")
        allowed = {r["candidate_index"] for r in packet["affected_candidates"]}
        edits = [e.candidate_index for e in self.edits]
        if len(edits) != len(set(edits)) or not set(edits) <= allowed:
            raise ValueError("Only edit each affected candidate once.")
        return self


def indexed_labels(records):
    """Keep the real definition span and line. A reference to a label is not its definition."""
    hits = {}
    for r in records.values():
        if not r.get("file") or not r.get("line"):
            continue
        for line, text in enumerate(uncomment(r["text"]).splitlines(), start=r["line"]):
            for key in re.findall(r"\\label\s*\{([^{}]+)\}", text):
                hits.setdefault(key, []).append((r["id"], r["file"], line))
    return hits


def _label_key(value):
    value = (value or "").strip()
    wrapper = re.fullmatch(r"\\(?:label|ref|eqref|autoref)\s*\{([^{}]+)\}", value)
    return wrapper[1] if wrapper else value


def label_sources(query, labels, records):
    """If a unique TeX label matches, retain its full indexed statement across splits."""
    key = _label_key(query)
    hits = labels.get(key, ())
    if len(hits) != 1:
        return []
    loc, file, _ = hits[0]
    return [r for r in records.values() if r["id"] == loc or (
        r.get("file") == file and any(s.get("label") == key for s in r.get("citation_statements", [])))]


def _local_finding(q, metadata, labels):
    if q.candidate_index is None and q.field in {"paper_title", "paper_authors"}:
        field = "title" if q.field == "paper_title" else "authors"
        value = metadata.get(field)
        if value:
            return {"status": "resolved", "finding": json_text(value), "evidence_ids": []}
    if q.field == "tex_label":
        hits = labels.get(_label_key(q.query or q.subject), ())
        if len(hits) == 1:
            loc, file, line = hits[0]
            return {"status": "resolved", "finding": f"{file} L{line}", "evidence_ids": [loc]}
    return None


def _passage(row):
    return {k: row[k] for k in ("id", "section", "label", "text", "evidence_type") if row.get(k)}


def _open_in_passages(ledger, origins, labels):
    linked = {}
    for key, item in ledger.items():
        if item["status"] != "unresolved":
            continue
        target = item.get("target_id") or item["source_id"]
        query = (item.get("query") or "").strip()
        label = labels.get(_label_key(query), ())
        if target in origins or any(loc in origins for loc, _, _ in label) or (
            query and any(query.casefold() in r["text"].casefold() for r in origins.values())
        ):
            linked[key] = item
    return linked


def _owner(row):
    return row["kind"], row.get("source_number"), row["claim"], row["evidence_ids"][0]


def attach_closed_readings(candidates, ledger):
    """Carry a fact established in a later chunk back to the exact lead which requested it."""
    closed = {}
    still_open = {}
    for item in ledger.values():
        if item.get("settled_later") and item.get("owner"):
            closed.setdefault(tuple(item["owner"]), []).append(item)
        elif item["status"] == "unresolved" and item.get("owner"):
            still_open.setdefault(tuple(item["owner"]), []).append(item["finding"])
    if not closed:
        return candidates
    output = []
    for row in candidates:
        answers = closed.get(_owner(row), [])
        if answers:
            ids = [loc for item in answers for loc in item["evidence_ids"]]
            row = {**row, "evidence_ids": list(dict.fromkeys([*row["evidence_ids"], *ids]))}
            if row.get("decision") == "unresolved" and "candidate_id" not in row:
                pending = still_open.get(_owner(row), [])
                row.update(decision="unresolved" if pending else "admitted",
                           decision_reason="; ".join(pending) if pending else row["significance"])
        output.append(row)
    return output


def read_segment(models, rubric, packet, metadata, labels, ledger, readings, resolve, notify):
    origins = {r["id"]: r for r in packet["passages"]}
    source_packet = {**packet, "passages": [_passage(r) for r in origins.values()],
                     "paper_identity": {k: v for k, v in metadata.items() if v}}
    previous = _open_in_passages(ledger, origins, labels)
    previous_ids = {f"prior{i}": key for i, key in enumerate(previous, 1)}
    if previous_ids:
        source_packet["previously_open"] = [
            {"question_id": qid, "field": ledger[key]["field"], "subject": ledger[key]["subject"],
             "candidate": ledger[key].get("owner"), "question": ledger[key]["finding"]}
            for qid, key in previous_ids.items()]
    scan = models.structured(SegmentLeads, rubric + "\n" + SCAN_INSTRUCTIONS, source_packet)
    for answer in scan.closed_readings:
        ledger[previous_ids[answer.question_id]].update(answer.model_dump(exclude={"question_id"}), settled_later=True)
    candidates = list(scan.candidates)
    pending = {}
    questions = {}
    for q in scan.questions:
        lead = scan.candidates[q.candidate_index] if q.candidate_index is not None else None
        key = q.key(lead)
        questions[key] = q
        row = ledger.get(key)
        if row and (row["status"] == "resolved" or row.get("checked")):
            if lead and row["status"] == "resolved" and row.get("edit"):
                candidates[q.candidate_index] = lead.model_copy(update=row["edit"])
            continue
        answer = _local_finding(q, metadata, labels)
        header = {"source_id": q.source_id, "source_number": lead.source_number if lead else None,
                  "target_id": q.target_id, "query": q.query, "owner": _owner(lead.model_dump()) if lead else None,
                  "field": q.field, "subject": q.subject}
        ledger[key] = {**header, **(answer or {"status": "unresolved", "finding": q.question, "evidence_ids": []})}
        if answer:
            if lead and answer["evidence_ids"]:
                ids = list(dict.fromkeys([*lead.evidence_ids, *answer["evidence_ids"]]))
                candidates[q.candidate_index] = lead.model_copy(update={"evidence_ids": ids})
                ledger[key]["edit"] = {"evidence_ids": ids}
            notify("evidence", {"tool": "paper_metadata" if q.field != "tex_label" else "tex_label",
                                "question": q.question, "finding": answer["finding"], "locations": answer["evidence_ids"]})
        elif q.lookup():
            pending[key] = q
    if pending:
        checks = list(pending.values())
        evidence = resolve([q.lookup() for q in checks], {q.source_id: origins[q.source_id] for q in checks})
        readings.extend(evidence["source_images"])
        # A lookup may resolve to exactly what the first reader already saw. Keep
        # the limitation available to later chunks without paying to reread it.
        extra = set(r["id"] for r in evidence["passages"]) - origins.keys()
        with_images = {r["source_id"] for r in evidence["source_images"]}
        followup = {}
        located = []
        for (key, q), details in zip(pending.items(), evidence["source_questions"], strict=True):
            ledger[key]["checked"] = True
            if (q.target_id or q.source_id) in with_images or q.target_id in extra or extra.intersection(details.get("search_hits", [])):
                followup[key] = q
                located.append(details)
        pending = followup
        evidence["source_questions"] = located
    if pending:
        checks = list(pending.values())
        affected = sorted({q.candidate_index for q in checks if q.candidate_index is not None})
        relevant = {q.source_id for q in checks} | {q.target_id for q in checks}
        relevant.update(loc for detail in located for loc in detail.get("search_hits", []))
        evidence["passages"] = [r for r in evidence["passages"] if r["id"] in relevant]
        for i, (q, resolved) in enumerate(zip(checks, evidence["source_questions"], strict=True)):
            resolved.update(question_id=f"q{i+1}", field=q.field, subject=q.subject, candidate_index=q.candidate_index)
        evidence.update(passages=[_passage(r) for r in evidence["passages"]],
                        affected_candidates=[{"candidate_index": i, **candidates[i].model_dump()} for i in affected])
        result = models.structured(ReadingAnswers, CHECK_INSTRUCTIONS, evidence)
        keys = {f"q{i+1}": key for i, key in enumerate(pending)}
        for answer in result.answers:
            ledger[keys[answer.question_id]].update(answer.model_dump(exclude={"question_id"}))
        for edit in result.edits:
            i = edit.candidate_index
            changes = edit.model_dump(exclude_unset=True, exclude={"candidate_index", "evidence_ids"}, exclude_none=True)
            ids = [*candidates[i].evidence_ids, *edit.evidence_ids]
            changes["evidence_ids"] = list(dict.fromkeys(ids))
            candidates[i] = candidates[i].model_copy(update=changes)
            for key, q in pending.items():
                if q.candidate_index == i and ledger[key]["status"] == "resolved":
                    ledger[key]["edit"] = changes
        candidates.extend(result.additions)
    for key, q in questions.items():
        if q.candidate_index is not None:
            i = q.candidate_index
            ids = list(dict.fromkeys([*candidates[i].evidence_ids, *ledger[key].get("evidence_ids", [])]))
            candidates[i] = candidates[i].model_copy(update={"evidence_ids": ids})
    candidates = [c.candidate([ledger[key]["finding"] for key, q in questions.items()
                               if q.candidate_index == i and ledger[key]["status"] == "unresolved"])
                  for i, c in enumerate(candidates)]
    return Findings(candidates=candidates, uncertainties=open_readings(ledger))


def open_readings(ledger):
    return list(dict.fromkeys(
        f"{row['source_id']} ({row['source_number'] or row['subject']}, {row['field']}): {row['finding']}"
        for row in ledger.values() if row["status"] == "unresolved"))
