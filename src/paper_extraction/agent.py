"""One persistent per-paper workflow, with bounded evidence-only tool loops."""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from importlib.resources import files
from pathlib import Path
from typing import Annotated, TypedDict

from filelock import FileLock
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .models import AdmissionReview, Draft, DraftPatch, Findings, Models, Review, json_text
from .editorial import DRAFT_GUIDANCE, EDIT_CONTRACT_VERSION, EXTRACTION_GUIDANCE, REVIEW_GUIDANCE, evidence_packet
from .profile import BuildFailure, bind_formal_citations, build_case, saved_build_targets
from .pdfops import page_count, render_page
from .sources import SourceIndex
from .selection import SELECTION_VERSION, exclusion_notes, select_contributions
from .scope_reconciliation import scope_failure_reason
from .store import (CONTENT_REVIEW_VERSION, legacy_confirmation_failure, legacy_content_failure,
                    repairable_citation_draft, saved_writer_targets, stale_selection_failure)
from .writer_patch import PATCH_GUIDANCE, WRITER_VERSION, apply_patch, repair_targets

Notify = Callable[[str, dict], None]
_DOCUMENTS = ("introduction.tex", "introduction.pdf", "preamble-v13.tex")
_BUDGET = 18000
_TOOL_ROUNDS = 4


class EvidenceState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    rounds: int


class PaperState(TypedDict, total=False):
    cursor: int
    candidates: list[dict]
    uncertainties: list[str]
    selection_version: str
    writer_version: str
    pending_writer_review: dict
    recheck_saved_selection: bool
    prerequisites_checked: list[str]
    attempt: int
    reviews: list[dict]
    draft: dict
    issues: list[str]
    target_ids: list[str]
    result_dir: str
    status: str
    reason: str
    hashes: dict[str, str]


def _rules() -> str:
    assets = files("paper_extraction").joinpath("profile_assets")
    return "\n".join(assets.joinpath(name).read_text(encoding="utf-8") for name in
                     ("intro-rules-v13.txt", "prompt-v13.txt", "template-v13.tex", "preamble-v13.tex"))


def _hash_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _thread_id(source: Path, source_root: Path | None, settings: dict, rules: str) -> str:
    identity = hashlib.sha256(rules.encode())
    identity.update(json_text([settings.get(k) for k in
                               ("provider", "base_url", "text_model", "vision_model")]).encode())
    identity.update(source.resolve().as_posix().encode())
    inputs = [source] if source_root is None else sorted(
        (p for p in source_root.rglob("*") if p.is_file()),
        key=lambda p: p.relative_to(source_root).as_posix(),
    )
    for path in inputs:
        identity.update(json_text([path.relative_to(source_root).as_posix()
                                   if source_root else source.name, _hash_file(path)]).encode())
    return identity.hexdigest()


def _batches(records: list[dict], *, stable_scan=False) -> list[list[dict]]:
    batches, current, size = [], [], 0
    for record in records:
        new_size = 3500 if stable_scan and record.get("transcribed_from_image") else (len(str(record["text"])) or 3500)
        if current and size + new_size > _BUDGET:
            batches.append(current)
            current, size = [], 0
        current.append(record)
        size += new_size
    if current:
        batches.append(current)
    return batches


def _missing_categories(candidates):
    kinds = {c["kind"] for c in candidates if c["decision"] == "admitted"}
    missing = []
    if "historical" not in kinds:
        missing.append("historical")
    if not kinds.intersection({"current", "application"}):
        missing.append("current")
    return missing


_PRIOR = re.compile(
    r"\b(?:prior|previously|generaliz\w*|extends?|classical|known\s+(?:that|when|for|in)|"
    r"asserted that|showed that|proved (?:that|by)|theorem of|due to|result of)\b", re.I)


def _prerequisite_sources(catalog, candidates, missing):
    relevant = {"historical"} if "historical" in missing else set()
    if "current" in missing:
        relevant.update({"current", "application"})
    locations = {loc for c in candidates if c["kind"] in relevant or
                 ("historical" in missing and _PRIOR.search(c["claim"] + " " + c["significance"]))
                 for loc in c["evidence_ids"]}
    ranked = []
    for i, r in enumerate(catalog):
        introduction = i < 2 or r.get("page") in (1, 2) or re.search(
            r"abstract|introduction|background", r.get("section", ""), re.I)
        explicit = ("historical" in missing and _PRIOR.search(r["text"])) or (
            "current" in missing and re.search(r"\bmain (?:result|theorem)|we (?:prove|show|establish)", r["text"], re.I))
        if introduction or explicit or r["id"] in locations:
            ranked.append((0 if introduction else 1 if explicit else 2, i, r))
    return [r for _, _, r in sorted(ranked)]


def _restore_scan_readings(saver, config, index):
    """Recover machine page readings captured by the previous extract subgraph."""
    missing = {r["id"]: r["page"] for r in index.catalog() if r.get("page") and not r["text"].strip()}
    if not missing:
        return
    for saved in saver.list(config):
        messages = saved.checkpoint.get("channel_values", {}).get("messages", [])
        for msg in messages:
            if not isinstance(msg, HumanMessage) or not isinstance(msg.content, str):
                continue
            try:
                payload = json.loads(msg.content)
            except ValueError:
                continue
            for row in payload.get("passages", []) if isinstance(payload, dict) else []:
                if not isinstance(row, dict):
                    continue
                loc, page = row.get("id"), row.get("page")
                if loc in missing and missing[loc] == page and row.get("transcribed_from_image") and row.get("text"):
                    index.save_transcription(loc, page, row["text"])
                    missing.pop(loc)
        if not missing:
            break



@contextmanager
def _private_model_errors(keys):
    # LangGraph persists node exceptions. Sanitize provider errors before either graph sees them.
    try:
        yield
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        if status in (401, 403):
            raise RuntimeError(f"Model access denied (HTTP {status}). Check the configured API key and permissions.") from None
        original = detail = str(exc)
        for key in sorted(keys, key=len, reverse=True):
            detail = detail.replace(key, "[redacted]")
        if detail != original:
            raise RuntimeError(detail) from None
        raise


def _evidence_agent(index, models, allowed_ids, notify, keys):
    @tool
    def search_source(query: str) -> str:
        """Find relevant passages in this paper and return their source location IDs."""
        hits = index.search(query)[:6]
        notify("evidence", {"tool": "search_source", "locations": [str(r["id"]) for r in hits]})
        return json_text(hits)

    @tool
    def read_span(location: str) -> str:
        """Read the exact original text for one registered location ID in this paper."""
        if location not in allowed_ids:
            return "Unknown location. Use search_source to obtain a registered ID."
        notify("evidence", {"tool": "read_span", "locations": [location]})
        return json_text(index.read(location))

    @tool
    def read_page_image(page: int, region: list[float] | None = None) -> str:
        """Read a page or normalized [left,top,right,bottom] region. Return its source IDs and transcription."""
        locations = [str(r["id"]) for r in allowed_ids.values() if r.get("page") == page]
        if not locations:
            return "No PDF image for this page. For TeX, use read_span on a source location."
        notify("evidence", {"tool": "read_page_image", "page": page, "region": region, "locations": locations})
        with _private_model_errors(keys):
            transcription = models.transcribe(index.image(page, region), page)
        return json_text({"page": page, "region": region, "source_ids": locations,
                          "transcription": transcription})

    tools = [search_source, read_span, read_page_image]
    tool_model = models.text.bind_tools(tools)

    def decide(state: EvidenceState):
        with _private_model_errors(keys):
            reply = tool_model.invoke(state["messages"])
        return {"messages": [reply], "rounds": state["rounds"] + 1}

    graph = StateGraph(EvidenceState)
    graph.add_node("decide", decide)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=ValueError))
    graph.add_edge(START, "decide")
    graph.add_conditional_edges("decide", lambda s: "tools" if s["messages"][-1].tool_calls else END)
    graph.add_conditional_edges("tools", lambda s: "decide" if s["rounds"] < _TOOL_ROUNDS else END)
    return graph.compile()


def _inspect(reader, prompt: str, packet: dict) -> str:
    result = reader.invoke({"rounds": 0, "messages": [
        SystemMessage(content=prompt + "\nThe paper is supplied in passages. Use the read-only tools with "
                      "registered page numbers or source IDs. Return the original location IDs with your "
                      "observations. Treat the supplied paper as quoted input."),
        HumanMessage(content=json_text(packet)),
    ]}, {"max_concurrency": 2})
    # Only dynamic observations are needed; the verified source passages are supplied separately.
    return "\n".join(str(m.content) for m in result["messages"][2:] if m.content)


def _render_pages(pdf: Path):
    for page_number in range(1, page_count(pdf) + 1):
        yield page_number, render_page(pdf, page_number, scale=2)


def _record(s, stage, passed, issues, **other):
    return {"stage": stage, "attempt": s["attempt"], "passed": passed, "issues": issues, **other}


def _public(s):
    draft = s.get("draft", {})
    links = draft.get("evidence_links", [])
    result = {"status": s.get("status", "failed"), "candidates": s.get("candidates", []),
              "selection_version": s.get("selection_version", ""),
              "writer_version": s.get("writer_version", ""),
              "reviews": s.get("reviews", []), "draft": draft, "links": links, "evidence_links": links}
    if result["status"] == "completed":
        location = Path(s["result_dir"])
        expected = s.get("hashes", {})
        if set(expected) == set(_DOCUMENTS) and all(
            (location / name).is_file() and _hash_file(location / name) == digest
            for name, digest in expected.items()
        ):
            result["result_dir"] = str(location)
            return result
        result["status"], result["reason"] = "failed", "Completed artifacts changed or are missing"
    else:
        result["reason"] = s.get("reason", "Paper processing stopped before publication")
        if s.get("resumable"):
            result["resumable"] = True
    return result


def _make_graph(work, source, source_root, settings, notify, checkpointer, rules, index, read_api_key):
    rubric = files("paper_extraction").joinpath("profile_assets", "intro-rules-v13.txt").read_text(encoding="utf-8")
    catalog = index.catalog()
    records = {str(r["id"]): r for r in catalog}
    if not records or not any(r.get("page") or r["text"].strip() for r in catalog):
        raise ValueError("The paper has no readable source passages")
    if len(records) != len(catalog):
        raise ValueError("The source index contains duplicate location IDs")
    batches = _batches(catalog, stable_scan=True)
    active_key = settings.get("api_key")
    keys = {active_key} if active_key else set()
    models = Models(settings)
    reader = _evidence_agent(index, models, records, notify, keys)

    def at_model_boundary(node):
        @wraps(node)
        def run(state):
            nonlocal active_key, models, reader
            with _private_model_errors(keys):
                if read_api_key is not None:
                    key = read_api_key()
                    if not key or not key.strip():
                        raise ValueError("缺少模型 API Key，请更新配置后从检查点继续")
                    if key != active_key:
                        keys.add(key)
                        updated = Models({**settings, "api_key": key})
                        reader = _evidence_agent(index, updated, records, notify, keys)
                        models, active_key = updated, key
                return node(state)
        return run

    def extract(s):
        cursor = s["cursor"]
        pending = s.get("target_ids", [])
        bundle = _batches([records[k] for k in pending])[0] if pending else batches[cursor]
        progress = ({"message": f"定向回查 {len(bundle)} 处原文", "remaining_locations": len(pending)}
                    if pending else {"chunk": cursor + 1, "total": len(batches)})
        notify("extract", {**progress, "locations": [r["id"] for r in bundle], "targeted": bool(pending)})
        passages = []
        for r in bundle:
            if not r["text"].strip() and r.get("page"):
                transcription = models.transcribe(index.image(r["page"]), r["page"])
                index.save_transcription(str(r["id"]), r["page"], transcription)
                passages.append(r)
                notify("evidence", {"tool": "read_page_image", "page": r["page"], "locations": [r["id"]]})
            else:
                passages.append(r)
        packet = {"passages": passages, "targeted_issue": s.get("issues", []) if pending else []}
        observations = _inspect(reader, rubric, packet)
        res = models.structured(Findings, rubric + "\n" + EXTRACTION_GUIDANCE,
            {**packet, "observations": observations})
        additions = [c.model_dump() for c in res.candidates]
        unknown = sorted({k for c in additions for k in c["evidence_ids"] if k not in records})
        if unknown:
            raise ValueError(f"Model cited unregistered evidence locations: {unknown[:5]}")
        # Keep earlier and newly observed clauses. Only paper-wide selection can decide
        # whether two results sharing a number or page are alternate formulations.
        candidates = s["candidates"] + additions
        consumed = {str(r["id"]) for r in bundle}
        return {"cursor": cursor if pending else cursor + 1, "candidates": candidates, "selection_version": "",
                "recheck_saved_selection": False, "pending_writer_review": {},
                "target_ids": [k for k in pending if k not in consumed],
                "uncertainties": s["uncertainties"] + res.uncertainties}

    def source_gap(missing):
        reasons = []
        if "historical" in missing:
            reasons.append("在所给原文中未能核实直接相关、且具备可归属的假设和结论的历史定理（v13 C1–C2）。")
        if "current" in missing:
            reasons.append("在所给原文中未能确定符合 v13 B1–B4 准入条件的当前核心结果或独立应用。")
        return "".join(reasons)

    def prerequisites(s):
        candidates = list(s["candidates"])
        uncertainties = list(s["uncertainties"])
        missing = _missing_categories(candidates)
        checked = list(dict.fromkeys(s.get("prerequisites_checked", []) + missing))
        selected = _prerequisite_sources(catalog, candidates, missing)
        # A machine transcription helps locate a comparison. Send its actual page
        # image when deciding whether that comparison has enough support for v13.
        bundles = []
        digital = []
        for r in selected:
            if r.get("page") and (r.get("transcribed_from_image") or not r["text"].strip()):
                bundles.extend(_batches(digital))
                digital = []
                bundles.append([r])
            else:
                digital.append(r)
        bundles.extend(_batches(digital))
        for bundle in bundles:
            missing = _missing_categories(candidates)
            if not missing:
                break
            kinds = {"historical"} if "historical" in missing else set()
            if "current" in missing:
                kinds.update({"current", "application"})
            locs = [r["id"] for r in bundle]
            notify("evidence", {"tool": "check_prerequisites", "missing_categories": missing, "locations": locs})
            focused = [c for c in candidates if c["kind"] in kinds or
                       any(loc in locs for loc in c["evidence_ids"])]
            packet = {"missing_categories": missing, "passages": bundle,
                      "related_existing_candidates": focused}
            prompt = (rubric + "\nFill Findings for the categories listed in missing_categories. "
                      "Use the supplied passage or physical_page and copy its registered IDs "
                      "into evidence_ids. Use uncertainties for a source limitation.")
            image = None
            first = bundle[0]
            if first.get("page") and (first.get("transcribed_from_image") or not first["text"].strip()):
                image = index.image(first["page"])
                packet["physical_page"] = first["page"]
            else:
                packet["observations"] = _inspect(reader, rubric + "\nThe requested candidate categories "
                                                   "are in missing_categories.", packet)
            found = models.structured(Findings, prompt, packet, image=image)
            for obj in found.candidates:
                c = obj.model_dump()
                if c["kind"] not in kinds:
                    continue
                if not all(k in records for k in c["evidence_ids"]):
                    raise ValueError("Prerequisite reader cited an unregistered source location")
                if image is not None and first["id"] not in c["evidence_ids"]:
                    continue  # A different physical page cannot support this image-only decision.
                previous = next((i for i, old in enumerate(candidates) if old["kind"] == c["kind"]
                                 and old["claim"] == c["claim"] and old["evidence_ids"] == c["evidence_ids"]), None)
                if previous is None:
                    candidates.append(c)
                else:
                    candidates[previous] = c
            uncertainties.extend(found.uncertainties)
        return {"candidates": candidates, "uncertainties": uncertainties, "prerequisites_checked": checked,
                "selection_version": "", "status": "", "reason": ""}

    def select(s):
        notify("evidence", {"tool": "select_contributions", "message": "根据整篇原文筛选主要贡献",
                            "candidate_count": len(s["candidates"])})
        candidates = select_contributions(index.catalog(), s["candidates"], models)
        update = {"candidates": candidates, "selection_version": SELECTION_VERSION, "recheck_saved_selection": False}
        missing = _missing_categories(candidates)
        scope_reason = scope_failure_reason(candidates)
        gap = source_gap(missing) if missing and set(missing).issubset(s.get("prerequisites_checked", [])) else ""
        if scope_reason or gap:
            return {**update, "status": "failed", "reason": "\n".join(reason for reason in (scope_reason, gap) if reason)}
        return update

    def selected_packet(s, draft=None):
        # Local estimates remain visible in the external evidence register. The writer
        # and reviewer receive the source-bound whole-paper choices and their primary passages.
        candidates = [{k: v for k, v in c.items() if not k.startswith("local_") and k != "significance"}
                      for c in s["candidates"] if c["decision"] == "admitted"]
        original = index.catalog()
        packet = evidence_packet(original, candidates, draft)
        locations = {r["id"] for r in packet["passages"]}
        for c in candidates:
            locations.update(c.get("selection_evidence_ids") or c["evidence_ids"][:1])
        packet["passages"] = [r for r in original if r["id"] in locations]
        return packet

    def write_draft(s):
        if s.get("selection_version") != SELECTION_VERSION:
            return {}
        attempt = s["attempt"] + 1
        if _missing_categories(s["candidates"]):
            # An old in-flight graph may still have a pending write node.
            return {}
        previous = s.get("draft")
        prior_check = s.get("pending_writer_review") or (s["reviews"][-1] if previous and s.get("reviews") else {})
        defects = prior_check.get("defects", [])
        targets = repair_targets(previous, prior_check, records.keys())
        notify("draft", {"attempt": attempt, "mode": "patch" if targets else "full",
                         "blocks": [item["block_id"] for item in targets]})
        packet = selected_packet(s, previous)
        passages = {r["id"]: r for r in packet["passages"]}
        for defect in defects:
            loc = defect["evidence_id"]
            if loc in records:
                passages[loc] = records[loc]
        packet.update(passages=list(passages.values()), previous_draft=packet.pop("draft", None),
                      repair_issues=s.get("issues", []), uncertainties=s.get("uncertainties", []))
        if targets:
            packet["repair_targets"] = targets
            patch = models.structured(DraftPatch, rules + "\n" + PATCH_GUIDANCE, packet)
            draft = apply_patch(previous, patch, packet, records)
        else:
            packet["identified_defects"] = defects
            draft = models.structured(Draft, rules + "\n" + DRAFT_GUIDANCE, packet).model_dump()
            draft = bind_formal_citations(draft, records)
        unknown = sorted({k for l in draft["evidence_links"] for k in l["evidence_ids"] if k not in records})
        issues = [f"Replace unregistered evidence IDs: {unknown[:5]}"] if unknown else []
        return {"attempt": attempt, "draft": draft, "issues": issues, "writer_version": WRITER_VERSION,
                "pending_writer_review": {}}

    def review(s):
        notify("review", {"attempt": s["attempt"]})
        issues = list(s["issues"])
        targets = []
        defects = []
        if not issues:
            packet = selected_packet(s, s["draft"])
            packet["previous_model_exclusion_notes"] = exclusion_notes(s["candidates"])
            observation = _inspect(reader, rubric, packet)
            instructions = rubric + "\n" + REVIEW_GUIDANCE
            verdict = models.structured(AdmissionReview, instructions, {
                **packet, "source_observations": observation,
            })
            issues = verdict.repair_issues()
            defects = [item.model_dump() for item in verdict.issues]
            if any(item.evidence_id not in records for item in verdict.issues):
                raise ValueError("The reviewer cited an unregistered original evidence location")
            if verdict.route == "extract":
                targets = list(dict.fromkeys(verdict.evidence_ids))
                if any(key not in records for key in targets):
                    raise ValueError("The reviewer returned an unknown original evidence location")
        route = "extract" if targets else ("draft" if issues else "pass")
        if issues:
            notify("review", {"attempt": s["attempt"], "route": route, "issues": issues,
                              "locations": targets, "message": issues[0]})
        return {"issues": issues, "target_ids": targets,
                "reviews": s["reviews"] + [_record(s, "review", not issues, issues, route=route,
                                                   review_version=CONTENT_REVIEW_VERSION, evidence_ids=targets,
                                                   defects=defects, edit_contract_version=EDIT_CONTRACT_VERSION)]}

    def compile_pdf(s):
        notify("compile", {"attempt": s["attempt"]})
        payload = {k: v for k, v in s["draft"].items() if k != "evidence_links"}
        repair_targets = []
        try:
            location = build_case(work, source, source_root, payload)
            issues = []
        except (ValueError, RuntimeError) as exc:
            location, issues = None, [str(exc)[:3000]]
            repair_targets = exc.repair_targets if isinstance(exc, BuildFailure) else saved_build_targets(payload, [str(exc)])
            notify("compile", {"attempt": s["attempt"], "issues": issues, "message": issues[0]})
        return {"result_dir": str(location) if location else "", "issues": issues,
                "reviews": s["reviews"] + [_record(s, "compile", not issues, issues, repair_targets=repair_targets)]}

    def visual(s):
        issues, reviews = [], []
        for page, png in _render_pages(Path(s["result_dir"]) / "introduction.pdf"):
            notify("visual", {"page": page, "attempt": s["attempt"]})
            review = models.structured(Review, "Visually inspect this rendered math Introduction "
                "page for clipped or overflowing glyphs/formulas, misplaced source citations, uneven "
                "condition left alignment, broken relations columns and orphaned headings. Normal "
                "whitespace is allowed. Report only visible defects with a location.",
                {"page": page, "title": s["draft"]["title"]}, image=png)
            issues.extend(f"Page {page}: {msg}" for msg in review.issues)
            reviews.append(_record(s, "visual", review.passed, review.issues, page=page))
        if not reviews:
            issues.append("PDF has no rendered pages")
        return {"issues": issues, "reviews": s["reviews"] + reviews}

    def publish(s):
        case = Path(s["result_dir"])
        hashes = {name: _hash_file(case / name) for name in _DOCUMENTS}
        receipt = work / "published.json"
        committed = {**s, "status": "completed", "hashes": hashes}
        temp = receipt.with_suffix(".tmp")
        temp.write_text(json_text(committed), encoding="utf-8")
        temp.replace(receipt)
        notify("publish", {"attempt": s["attempt"], "files": list(hashes)})
        return {"status": "completed", "hashes": hashes}

    def failed(s):
        return {"status": "failed", "reason": "; ".join(s["issues"]) or "Automatic review failed"}

    def advance(next_node):
        def route(s):
            if not s["issues"]:
                return next_node
            if s["attempt"] >= 3:
                return "failed"
            return "extract" if s.get("target_ids") else "draft"
        return route

    graph = StateGraph(PaperState)
    for name, node in (("extract", extract), ("prerequisites", prerequisites), ("select", select),
                       ("draft", write_draft), ("review", review),
                       ("compile", compile_pdf), ("visual", visual), ("publish", publish), ("failed", failed)):
        if name in {"extract", "prerequisites", "select", "draft", "review", "visual"}:
            node = at_model_boundary(node)
        graph.add_node(name, node)
    graph.add_edge(START, "extract")
    def after_extract(s):
        if s.get("target_ids") or s["cursor"] < len(batches):
            return "extract"
        # Locate missing source material before the first global choice when the
        # initial local register already shows a gap.
        missing = set(_missing_categories(s["candidates"]))
        return "prerequisites" if missing.difference(s.get("prerequisites_checked", [])) else "select"

    def after_select(s):
        if s.get("status") == "failed":
            return END
        return "prerequisites" if _missing_categories(s["candidates"]) else "draft"

    graph.add_conditional_edges("extract", after_extract)
    graph.add_edge("prerequisites", "select")
    graph.add_conditional_edges("select", after_select)
    graph.add_conditional_edges("draft", lambda s: "review" if s.get("selection_version") == SELECTION_VERSION else "select")
    for node, next_node in (("review", "compile"), ("compile", "visual"), ("visual", "publish")):
        graph.add_conditional_edges(node, advance(next_node))
    graph.add_edge("failed", END)
    graph.add_edge("publish", END)
    return graph.compile(checkpointer=checkpointer), len(batches)


def run_paper(work: Path, source: Path, source_root: Path | None, settings: dict, notify: Notify,
              *, retry_legacy_review: bool = False, read_api_key: Callable[[], str] | None = None) -> dict:
    work = Path(work).resolve()
    work.mkdir(parents=True, exist_ok=True)
    with FileLock(work / "paper.lock"):
        return _run_locked(work, source, source_root, settings, notify, retry_legacy_review, read_api_key)


def _run_locked(work: Path, source: Path, source_root: Path | None, settings: dict, notify: Notify,
                retry_legacy_review: bool, read_api_key: Callable[[], str] | None) -> dict:
    work, source = Path(work).resolve(), Path(source).resolve()
    source_root = Path(source_root).resolve() if source_root is not None else None
    work.mkdir(parents=True, exist_ok=True)
    state = {}
    resumable = False
    try:
        if source_root and (not source.is_relative_to(source_root) or work.is_relative_to(source_root)):
            raise ValueError("Source entry must be inside its declared root, and output outside it")
        rules = _rules()
        thread = _thread_id(source, source_root, settings, rules)
        binding = work / "paper-binding.json"
        if binding.exists() and json.loads(binding.read_text("utf-8"))["identity"] != thread:
            return _public({"status": "failed", "reason": "Source or model configuration changed. Use a new run directory."})
        if not binding.exists():
            temp = binding.with_suffix(".tmp")
            temp.write_text(json_text({"identity": thread}), encoding="utf-8")
            temp.replace(binding)
        receipt = work / "published.json"
        if receipt.is_file():
            return _public(json.loads(receipt.read_text("utf-8")))
        with SqliteSaver.from_conn_string(os.fspath(work / "checkpoint.sqlite")) as saver:
            index = SourceIndex(source, source_root, work)
            graph, batch_count = _make_graph(work, source, source_root, settings, notify, saver, rules, index, read_api_key)
            config = {"configurable": {"thread_id": thread}, "recursion_limit": 3 * batch_count + 30}
            prior = graph.get_state(config)
            state = prior.values
            finished_scan = state.get("cursor") == batch_count
            explicit_upgrade = (retry_legacy_review and not prior.next and finished_scan and
                                (legacy_content_failure(state) or stale_selection_failure(state)))
            old_terminal = retry_legacy_review and not prior.next and finished_scan
            recovered = None
            if old_terminal:
                from .saved_selection_recovery import saved_selection_recovery
                recovered = saved_selection_recovery(saver, config, state, index)
            citation_draft = repairable_citation_draft(state, index) if old_terminal else None
            repeat_review = old_terminal and (recovered is not None or legacy_confirmation_failure(state) or citation_draft is not None)
            saved_edits = saved_writer_targets(state, index) if old_terminal and not repeat_review else []
            pending_old_draft = (finished_scan
                                 and state.get("selection_version") != SELECTION_VERSION
                                 and any(node in {"draft", "review", "compile", "visual", "publish", "failed"}
                                         for node in prior.next))
            if prior.created_at and not prior.next and state.get("status") and not (explicit_upgrade or repeat_review or saved_edits):
                return _public(state)
            if prior.created_at:
                _restore_scan_readings(saver, config, index)
            if repeat_review:
                base = recovered or state
                message = ("恢复上轮数学核验通过且仅待排版修订的筛选和草稿，重新核验原文" if recovered else
                           "按已绑定原文补齐定位，复核最终草稿" if citation_draft is not None else
                           "复用最终草稿和整篇筛选，重新核查上一轮仅确认无误的意见")
                notify("review", {"message": message, "selection_version": base["selection_version"],
                                  "previous_checkpoint": base.get("recovery_checkpoint_id", prior.config["configurable"]["checkpoint_id"])})
                graph.update_state(config, {"status": "", "reason": "", "issues": [], "target_ids": [],
                                            "attempt": 1, "reviews": [], "result_dir": "", "hashes": {},
                                            "recheck_saved_selection": True, "pending_writer_review": {},
                                            "candidates": base["candidates"], "selection_version": base["selection_version"],
                                            "draft": bind_formal_citations(base["draft"], {r["id"]: r for r in index.catalog()})},
                                   as_node="draft")
            elif saved_edits:
                # Fork the terminal checkpoint immediately before the next writer call.
                # Keep the exact failed review as input, with prior history in the archive.
                last = state["reviews"][-1]
                notify("draft", {"message": "复用原文筛选和最终草稿，按上一轮反馈局部修订",
                                 "blocks": [item["block_id"] for item in saved_edits],
                                 "previous_checkpoint": prior.config["configurable"]["checkpoint_id"]})
                graph.update_state(config, {"status": "", "reason": "", "issues": last["issues"],
                                            "target_ids": [], "attempt": 0, "reviews": [], "result_dir": "",
                                            "hashes": {}, "recheck_saved_selection": True, "pending_writer_review": last},
                                   as_node="review")
            elif explicit_upgrade or pending_old_draft:
                # Reuse the source pass and fork the checkpoint. Historical drafts
                # and verdicts stay in the previous checkpoint and attempt archive.
                restart = {"status": "", "reason": "", "issues": [], "target_ids": [], "attempt": 0,
                           "draft": {}, "reviews": [], "result_dir": "", "hashes": {},
                           "selection_version": "", "recheck_saved_selection": False,
                           "pending_writer_review": {}, "prerequisites_checked": []}
                notify("draft", {"message": "复用已提取的候选，按整篇原文重新筛选并写稿",
                                 "candidate_count": len(state.get("candidates", [])),
                                 "previous_checkpoint": prior.config["configurable"]["checkpoint_id"]})
                graph.update_state(config, restart, as_node="extract")
            initial = None if prior.created_at else {"cursor": 0, "attempt": 0, "issues": [],
                "candidates": [], "uncertainties": [], "reviews": []}
            try:
                state = graph.invoke(initial, config)
            except Exception:
                interrupted = graph.get_state(config)
                state = interrupted.values
                resumable = bool(interrupted.next)
                raise
        return _public(state)
    except Exception as exc:
        detail = str(exc)
        key = settings.get("api_key")
        if key:
            detail = detail.replace(str(key), "[redacted]")
        return _public({**state, "status": "failed", "resumable": resumable,
                        "reason": f"{type(exc).__name__}: {detail[:3000]}"})

