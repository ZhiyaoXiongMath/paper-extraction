"""Compare repeated findings at their own source, block or rendered page."""
import hashlib
import json
import re


POLICY = "source-bound-repair-1"


def _text(value):
    return " ".join(str(value or "").split())


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def block_at(draft, path):
    value = draft
    try:
        for part in path.split("."):
            value = value[int(part)] if isinstance(value, list) else value[part]
    except (KeyError, IndexError, TypeError, ValueError):
        return ""
    return value


def _compile_observation(field, diagnostic, text):
    height = re.search(r"rendered height (\d+(?:\.\d+)?)U, limit (\d+(?:\.\d+)?)U", diagnostic, re.I)
    if height:
        grouped = "resultgroup_measurement_failed" in diagnostic or "Result group exceeds" in diagnostic
        group = re.match(r"groups\.\d+", field) if grouped else None
        location = group[0] if group else field
        kind = "group height" if grouped else "block height"
        return (location, kind, height[2]), {"magnitude": max(0, float(height[1]) - float(height[2]))}
    words = re.search(r"\b(\w+_word_limit):(\d+)>(\d+)", diagnostic)
    if words:
        return (field, words[1], words[3]), {"magnitude": max(0, int(words[2]) - int(words[3]))}
    box = re.search(r"(Overfull \\[hv]box)\s*\((\d+(?:\.\d+)?)pt too (wide|high)\)", diagnostic)
    if box:
        return (field, box[1], box[3]), {"magnitude": float(box[2])}
    return (field, _text(diagnostic)), {"state": _digest(text)}


def _observations(stage, draft, entries, issues):
    for check in entries:
        if check.get("passed"):
            continue
        if stage == "review" and check.get("defects"):
            for defect in check["defects"]:
                if not isinstance(defect, dict):
                    yield ("legacy", _text(defect)), {"state": _digest(draft)}
                    continue
                name = defect.get("draft_field") or ""
                anchor = (name, defect.get("kind"), defect.get("evidence_id"))
                # A reviewer can describe the same underlying defect with another excerpt
                # or operation. Only an actual change to the affected block advances it.
                value = block_at(draft, name)
                yield anchor, {"state": _digest(_text(value) if isinstance(value, str) else value)}
        elif stage == "compile" and check.get("repair_targets"):
            for target in check["repair_targets"]:
                field = target["draft_field"]
                # Preserve separate affected line ranges when several boxes overflow in
                # one block. A lone box stays associated with its block across line shifts.
                box_lines = [re.search(r"at lines? (\d+(?:--\d+)?)", msg) for msg in target["diagnostics"]
                             if "Overfull" in msg]
                several_boxes = len({m[1] for m in box_lines if m}) > 1
                for issue in target["diagnostics"]:
                    identity, observed = _compile_observation(field, issue, target["current_text"])
                    if several_boxes and "Overfull" in issue:
                        line = re.search(r"at lines? (\d+(?:--\d+)?)", issue)
                        if line:
                            identity = (*identity, line[1])
                    yield identity, observed
        elif stage == "compile":
            for issue in check["issues"]:
                for diagnostic in issue.split(" | "):
                    yield _compile_observation("", diagnostic, diagnostic)
        elif stage == "visual":
            for issue in check["issues"]:
                fields = sorted({loc["draft_field"] for loc in check.get("locations", []) if loc.get("issue") == issue and loc.get("draft_field")})
                page = check.get("page")
                image = check.get("page_sha256")
                if image:
                    # A byte-identical rendering cannot improve at the same location.
                    # Preserve separate page and block keys for actual new locations.
                    for field in fields or [""]:
                        yield (page, field), {"state": image}
                else:
                    observed = _digest([block_at(draft, name) for name in fields] if fields else draft)
                    yield (page, *fields, _text(issue)), {"state": observed}
        else:
            for issue in check.get("issues", issues):
                yield ("", _text(issue)), {"state": _digest(draft)}


def record_progress(state, stage, entries, issues):
    history = dict(state.get("repair_progress") or {})
    any_progress = False
    observations = {}
    for identity, value in _observations(stage, state["draft"], entries, issues):
        key = _digest([stage, identity])
        if "magnitude" in value and key in observations:
            # Several boxes can overflow in one block. Track the largest remaining excess.
            value = {"magnitude": max(value["magnitude"], observations[key]["magnitude"])}
        observations[key] = value
    if issues and not observations:
        observations[_digest([stage, issues])] = {"state": _digest(state["draft"])}
    # A newly appearing or partially resolved combination gets its own repair.
    # Returning to a previously seen set cannot restart a cycle.
    open_key = "open:" + stage
    previous_group = history.get(open_key, {})
    current_group = sorted(observations)
    group_id = _digest(current_group)
    seen_groups = previous_group.get("seen", [])
    if current_group and group_id not in seen_groups:
        any_progress = True
    history[open_key] = {"seen": list(dict.fromkeys([*seen_groups, group_id]))}
    for key, value in observations.items():
        previous = history.get(key)
        if "magnitude" in value:
            if previous is None or value["magnitude"] < previous["magnitude"]:
                any_progress = True
                history[key] = value
        else:
            seen = (previous or {}).get("states", [])
            if value["state"] not in seen:
                any_progress = True
                history[key] = {"states": [*seen, value["state"]]}
    return {"repair_policy": POLICY, "repair_progress": history,
            "repair_stalled": bool(issues) and not any_progress}


def legacy_quota_stop(state):
    """Recognize a saved terminal verdict from the former three-draft graph."""
    if state.get("repair_policy") or state.get("status") != "failed" or not state.get("draft"):
        return False
    checks = state.get("reviews") or []
    final = checks[-1] if checks else {}
    round_no = final.get("attempt", 0)
    stage = final.get("stage")
    if round_no < 3 or stage not in {"review", "compile", "visual"}:
        return False
    issues = [str(msg) for check in checks if check.get("stage") == stage and check.get("attempt") == round_no
              and check.get("passed") is False for msg in check.get("issues", [])]
    reason = state.get("reason") or ""
    return bool(issues) and all(msg in reason for msg in issues)


def approved_legacy_layout_targets(state, index, current_version):
    """Reuse a prior paper-wide selection only for a bound layout edit of its approved same-round draft."""
    from .writer_patch import repair_targets

    version = re.fullmatch(r"v13-whole-paper-(\d+)", state.get("selection_version") or "")
    current = re.fullmatch(r"v13-whole-paper-(\d+)", current_version)
    checks = state.get("reviews") or []
    if (not version or not current or not 0 < int(version[1]) < int(current[1])
            or state.get("status") != "failed" or not state.get("draft") or len(checks) < 2):
        return []
    content, layout = checks[-2:]
    round_no = layout.get("attempt")
    if (layout.get("stage") != "compile" or layout.get("passed") is not False or not layout.get("issues")
            or content.get("stage") != "review" or content.get("passed") is not True
            or content.get("issues") or content.get("defects") or not round_no or content.get("attempt") != round_no
            or state.get("draft_round", state.get("attempt", round_no)) != round_no):
        return []
    ids = {r["id"] for r in index.catalog()}
    cores = [row for row in state.get("candidates", []) if row.get("decision") == "admitted"
             and row.get("kind") in {"current", "application", "historical"}]
    if not any(r["kind"] == "historical" for r in cores) or not any(r["kind"] != "historical" for r in cores):
        return []
    for row in cores:
        bound = row.get("selection_evidence_ids") or row.get("evidence_ids") or []
        if not row.get("candidate_id") or not bound or not set(bound) <= ids:
            return []
    return repair_targets(state["draft"], layout, ids)


def stalled_saved_repair(state):
    return (state.get("status") == "failed" and state.get("repair_policy") == POLICY
            and bool(state.get("draft")) and bool(state.get("reviews"))
            and (state.get("repair_stalled") is True
                 or str(state.get("reason", "")).startswith("同一缺陷修订后仍未改善，已保留本轮草稿。")))
