"""Project a localizable reply error onto its original questions and evidence."""
from dataclasses import dataclass
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, ValidationError

from .models import EditorialDefect
from .repair_progress import block_at
from .review_response import ResultNomination
from .segment_evidence import ReadingAnswer, ReadingAnswers


class AnswerCorrections(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answers: list[ReadingAnswer]
    needs_full_context: bool = False


class IssueCorrections(BaseModel):
    model_config = ConfigDict(extra="forbid")
    issues: list[EditorialDefect]
    needs_full_context: bool = False


class NominationCorrections(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nominations: list[ResultNomination]
    needs_full_context: bool = False


@dataclass
class PartialReply:
    schema: type[BaseModel]
    payload: dict
    merge_rows: Callable[[list], dict]

    def merge(self, document):
        patch = self.schema.model_validate(document)
        if patch.needs_full_context:
            raise ValueError("full_context_requested")
        if isinstance(patch, AnswerCorrections):
            rows = patch.answers
        elif isinstance(patch, NominationCorrections):
            rows = patch.nominations
        else:
            rows = patch.issues
        shown = {p["id"] for p in self.payload["passages"]}
        shown.update(p["source_id"] for p in self.payload["source_images"])
        for row in rows:
            ids = row.evidence_ids if isinstance(row, ReadingAnswer) else [row.evidence_id]
            if not set(ids) <= shown:
                raise ValueError("unseen_patch_source")
        return self.merge_rows(rows)


def _evidence(packet, ids, *, targeted=False, source_index=None):
    passages = {p["id"]: p for p in packet.get("passages", []) if p["id"] in ids}
    images = [p for p in packet.get("source_images", []) if p["source_id"] in ids]
    if targeted:
        for row in packet.get("checked_source_images", ()):
            if row["source_id"] in ids and row not in images:
                images.append(row)
        # Location IDs are registered in the original request. Resolve only the
        # defective rows' sources locally, including a passage shown abridged.
        for row in packet.get("source_locations", ()):
            sid = row["id"]
            if sid not in ids or (sid in passages and not passages[sid].get("abridged")):
                continue
            original = (source_index or {}).get(sid)
            if original:
                passages[sid] = {k: original[k] for k in ("id", "text", "label", "section", "page") if k in original}
    return {"passages": list(passages.values()), "source_images": images}


def _reading_plan(document, packet):
    questions = {q["question_id"]: q for q in packet["source_questions"]}
    previous = document.get("answers")
    if not questions or not isinstance(previous, list):
        return None
    groups = {qid: [] for qid in questions}
    for item in previous:
        if not isinstance(item, dict) or not isinstance(item.get("question_id"), str) or item["question_id"] not in groups:
            # An unidentified answer could belong to any question.
            return None
        groups[item["question_id"]].append(item)
    registered = {p["id"] for p in packet["passages"]} | {p["source_id"] for p in packet["source_images"]}
    kept, targets = {}, []
    for qid, rows in groups.items():
        if len(rows) == 1:
            try:
                answer = ReadingAnswer.model_validate(rows[0])
            except ValidationError:
                pass
            else:
                if set(answer.evidence_ids) <= registered and (answer.status != "resolved" or answer.evidence_ids):
                    kept[qid] = rows[0]
                    continue
        targets.append(qid)
    if not targets:
        return None
    # Validate all frozen edits and additions against the original context separately.
    # Placeholders are only used in this local check and are never returned as answers.
    try:
        ReadingAnswers.model_validate({**document, "answers": [
            {"question_id": q, "status": "unresolved", "finding": "pending"} for q in questions
        ]}, context=packet)
    except ValidationError:
        return None
    selected = [questions[q] for q in targets]
    owners = {q.get("candidate_index") for q in selected}
    candidates = [r for r in packet["affected_candidates"] if r["candidate_index"] in owners]
    ids = {sid for q in selected for sid in (q.get("source_id"), q.get("target_id"), *q.get("search_hits", [])) if sid}
    ids.update(sid for row in candidates for sid in row.get("evidence_ids", []))
    context = _evidence(packet, ids)
    if not (context["passages"] or context["source_images"]):
        return None
    edits = [e for e in document.get("edits", []) if e["candidate_index"] in owners]
    context.update(source_questions=selected, affected_candidates=candidates, existing_edits=edits,
                   previous_rows=[r for q in targets for r in groups[q]],
                   correction="Return one corrected answer for each listed question_id. Keep existing edits consistent. "
                              "If a different edit, addition or unavailable source is needed, set needs_full_context.")

    def merge(rows):
        replacements = {row.question_id: row.model_dump() for row in rows}
        if len(replacements) != len(rows) or set(replacements) != set(targets):
            raise ValueError("answer_patch_membership")
        merged = {**kept, **replacements}
        return {**document, "answers": [merged[qid] for qid in questions]}

    return PartialReply(AnswerCorrections, context, merge)


def _review_plan(document, packet, errors, *, targeted=False, source_index=None):
    if document.get("source_checks"):
        return None
    verdict = document.get("verdict", document)
    if not isinstance(verdict, dict) or verdict.get("route") not in ("draft", "extract"):
        return None
    locations = [e["loc"][1:] if e["loc"][:1] == ("verdict",) else e["loc"] for e in errors]
    family = "nominations" if targeted and locations and all(loc[:1] == ("nominations",) for loc in locations) else "issues"
    issues = verdict.get(family)
    if not isinstance(issues, list):
        return None
    indexes = set()
    for loc in locations:
        if len(loc) < 2 or loc[0] != family or not isinstance(loc[1], int) or not 0 <= loc[1] < len(issues):
            return None
        indexes.add(loc[1])
    indexes = sorted(indexes)
    if not indexes:
        return None
    draft = packet.get("draft", {})
    visible = {r["draft_field"]: r for key in ("changed_blocks", "context_blocks") for r in packet.get(key, [])}
    evidence = {r["block_id"]: r["evidence_ids"] for r in draft.get("evidence_links", [])}
    ids, items = set(), []
    for idx in indexes:
        issue = issues[idx]
        if not isinstance(issue, dict) or not isinstance(issue.get("evidence_id"), str):
            return None
        if family == "nominations":
            ids.add(issue["evidence_id"])
            items.append({"previous_issue": issue})
            continue
        if not isinstance(issue.get("draft_field"), str):
            return None
        path = issue["draft_field"]
        current = visible.get(path, {})
        text = current.get("text") or block_at(draft, path)
        absent = (targeted and verdict["route"] == "extract" and path in {"history", "groups"}
                  and issue.get("operation") in {"recover", "insert"})
        if not absent and (not isinstance(text, str) or not text):
            return None
        ids.update(evidence.get(path, current.get("evidence_ids", [])))
        ids.add(issue["evidence_id"])
        items.append({"previous_issue": issue, "draft_text": "" if absent else text})
    context = _evidence(packet, ids, targeted=targeted, source_index=source_index)
    shown = {p["id"] for p in context["passages"]} | {p["source_id"] for p in context["source_images"]}
    if any(i["previous_issue"]["evidence_id"] not in shown for i in items):
        return None
    context.update(route=verdict["route"], rows_in_order=items,
                   correction="Correct each listed editorial issue in order, preserving its draft field and original "
                              "evidence location. If the verdict, other issues or wider context must change, set needs_full_context.")
    if targeted:
        context["candidates"] = [{k: row[k] for k in ("candidate_id", "source_number", "kind", "decision") if k in row}
                                 for row in packet.get("candidates", ()) if ids.intersection(row.get("evidence_ids", ()))]
        if family == "nominations":
            context["correction"] = ("Complete each listed nomination in order using its original evidence location. "
                                     "If the result or verdict requires different evidence, set needs_full_context.")

    def merge(rows):
        if len(rows) != len(indexes):
            raise ValueError("review_patch_membership")
        result = list(issues)
        for row, idx in zip(rows, indexes, strict=True):
            old = issues[idx]
            if row.evidence_id != old["evidence_id"] or (family == "issues" and row.draft_field != old["draft_field"]):
                raise ValueError("review_patch_identity")
            result[idx] = row.model_dump()
        repaired = {**verdict, family: result}
        return {**document, "verdict": repaired} if "verdict" in document else repaired

    return PartialReply(NominationCorrections if family == "nominations" else IssueCorrections, context, merge)


def plan_reply_repair(schema, document, payload, errors, *, targeted=False, source_index=None):
    if not isinstance(document, dict):
        return None
    if schema is ReadingAnswers:
        return _reading_plan(document, payload)
    return _review_plan(document, payload, errors, targeted=targeted, source_index=source_index)


def retry_diagnostics(errors):
    categories = {
        "Return one resolved or unresolved answer": "question_coverage",
        "Only cite evidence supplied": "unknown_evidence",
        "Name the source that resolves": "missing_answer_evidence",
        "Only edit each affected candidate": "candidate_edit_membership",
        "A pass has no issues": "route_issues_conflict",
        "Target missing claims": "missing_recovery_evidence",
        "Route evidence recovery": "recovery_route",
        "Use a registered source_id": "unknown_source_check",
        "Supply the actual new wording": "missing_replacement",
        "The proposed wording is identical": "unchanged_replacement",
        "Quote the": "missing_excerpt",
        "remove or recover must": "unexpected_replacement",
        "recover requires kind": "invalid_recovery_kind",
        "A result nomination requires": "nomination_route",
        "A nomination must use": "unknown_nomination_source",
        "A required_change saying no edit": "contradictory_edit",
    }
    result = []
    for row in errors[:16]:
        msg = row["msg"].removeprefix("Value error, ")
        code = next((code for prefix, code in categories.items() if msg.startswith(prefix)), row["type"])
        result.append({"field": ".".join(map(str, row["loc"])) or "document", "code": code})
    return result
