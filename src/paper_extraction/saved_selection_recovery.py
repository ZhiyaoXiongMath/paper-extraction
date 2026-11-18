"""Locate an earlier restored editorial selection displaced solely by a legacy writer gate."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import ValidationError

from .models import Draft


def _pending(state):
    return {name.removeprefix("branch:to:") for name in state if name.startswith("branch:to:")}


def _saved_editorial_state(state, locations):
    if not state.get("recheck_saved_selection") or not state.get("selection_version"):
        return False
    reviews = state.get("reviews", [])
    if not reviews or not state.get("issues") or state.get("target_ids"):
        return False
    last = reviews[-1]
    if last.get("stage") not in {"compile", "visual"} or last.get("passed") is not False:
        return False
    editorial = next((r for r in reversed(reviews) if r.get("stage") == "review"), {})
    if (editorial.get("passed") is not True or editorial.get("route") != "pass"
            or editorial.get("attempt") != state.get("attempt")):
        return False
    try:
        draft = Draft.model_validate(state.get("draft", {}))
    except ValidationError:
        return False
    selected = [c for c in state.get("candidates", []) if c.get("decision") == "admitted"]
    if not any(c.get("kind") == "historical" for c in selected):
        return False
    current = [c for c in selected if c.get("kind") in {"current", "application"}]
    if not 1 <= len(current) <= 4:
        return False
    return (all(c.get("candidate_id") and c.get("evidence_ids") and set(c["evidence_ids"]) <= locations for c in selected)
            and all(set(link.evidence_ids) <= locations for link in draft.evidence_links))


def saved_selection_recovery(saver, config, terminal, index):
    """Return a source-bound accepted draft only on the failed head's authoritative parent chain."""
    reviews = terminal.get("reviews", [])
    if (terminal.get("status") != "failed" or terminal.get("resumable") or terminal.get("writer_version")
            or not terminal.get("draft") or not reviews or reviews[-1].get("passed") is not False
            or reviews[-1].get("stage") not in {"review", "compile", "visual"}):
        return None
    head = saver.get_tuple(config)
    if head is None:
        return None
    values = head.checkpoint["channel_values"]
    if (_pending(values) or values.get("status") != "failed" or
            any(values.get(k) != terminal.get(k) for k in ("draft", "reviews", "candidates"))):
        return None
    locations = {row["id"] for row in index.catalog()}
    child = None
    checkpoint = head
    while checkpoint:
        state = checkpoint.checkpoint["channel_values"]
        route = _pending(state)
        if state.get("target_ids") or route.intersection({"extract", "prerequisites", "publish"}) or state.get("status") == "completed":
            return None
        if route == {"select"}:
            # A valid rescan or selection request cannot be recognized as the stale writer gate.
            if (child is None or not checkpoint.parent_config or state.get("recheck_saved_selection") is not False
                    or state.get("selection_version") == child.get("selection_version")
                    or _pending(child) != {"draft"} or not child.get("selection_version")
                    or child.get("draft") != state.get("draft") or child.get("candidates") == state.get("candidates")):
                return None
            parent = saver.get_tuple(checkpoint.parent_config)
            if parent is None:
                return None
            saved = parent.checkpoint["channel_values"]
            fixed = ("cursor", "attempt", "draft", "candidates", "reviews", "issues", "selection_version")
            if (_pending(saved) == {"draft"} and all(saved.get(k) == state.get(k) for k in fixed)
                    and _saved_editorial_state(saved, locations)):
                return {**saved, "recovery_checkpoint_id": parent.config["configurable"]["checkpoint_id"]}
            return None
        child = state
        checkpoint = saver.get_tuple(checkpoint.parent_config) if checkpoint.parent_config else None
    return None


def saved_selection_for_work(work, terminal, index):
    work = Path(work)
    binding, checkpoint = work / "paper-binding.json", work / "checkpoint.sqlite"
    if not (binding.is_file() and checkpoint.is_file() and terminal.get("status") == "failed" and terminal.get("draft")):
        return None
    thread = json.loads(binding.read_text("utf-8"))["identity"]
    config = {"configurable": {"thread_id": thread, "checkpoint_ns": ""}}
    with closing(sqlite3.connect(checkpoint.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        return saved_selection_recovery(SqliteSaver(db), config, terminal, index)

