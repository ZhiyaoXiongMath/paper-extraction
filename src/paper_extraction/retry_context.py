"""Source-bound questions from a failed attempt when its graph cannot be resumed."""
import re


def _short(value, limit):
    value = " ".join(str(value or "").split())
    return value if len(value) <= limit else value[:limit - 1] + "…"


def _ids(values):
    return list(dict.fromkeys(x for x in values if isinstance(x, str) and len(x) <= 80))[:8]


def previous_failure(result, attempt):
    reason = result.get("reason") or ""
    if result.get("status") != "failed" or not reason:
        return {}
    issues = []
    scope = reason.startswith(("v13 范围例外", "v13 B4：", "在所给原文中", "Insufficient sourced history or current contributions"))
    if reason.startswith(("SelectionConflict:", "ValueError: Scope reconciliation ")):
        from .selection import candidate_register
        stage = "select"
        prior_ids = set(re.findall(r"\bc\d+\b", reason))
        try:
            register = candidate_register(result.get("candidates") or [])
        except (KeyError, TypeError):
            register = []  # Older snapshots may not contain the full source register.
        for row in register:
            if row["candidate_id"] in prior_ids:
                issues.append({"previous_candidate": row["candidate_id"], "source_number": _short(row.get("source_number"), 120),
                               "question": _short(row.get("claim"), 320), "issue": _short(reason, 700),
                               "evidence_ids": _ids(row["local_evidence_ids"])})
        if not issues:
            issues.append({"issue": _short(reason, 700), "evidence_ids": []})
    elif scope:
        stage = "select"
        for row in result.get("candidates", []):
            if not row.get("scope_evidence_id"):
                continue
            relevant = (row.get("scope_central_open") if reason.startswith("v13 B4") else
                        row.get("decision") == "admitted" and row.get("kind") in {"current", "application"})
            if relevant:
                issues.append({"source_number": _short(row.get("source_number"), 120),
                               "question": _short(row.get("independent_question"), 320),
                               "issue": _short(row.get("decision_reason"), 700),
                               "evidence_ids": _ids([row["scope_evidence_id"], *row.get("selection_evidence_ids", [])])})
        if not issues:
            issues = [{"issue": _short(line, 700), "evidence_ids": re.findall(r"\b[pt]\d+-\d+\b", line)[:8]}
                      for line in reason.splitlines()[1:] or [reason]]
    else:
        if result.get("resumable"):
            return {}
        checks = result.get("reviews") or []
        if not checks:
            return {}
        last = checks[-1]
        stage = last.get("stage")
        failed = [x for x in checks if x.get("stage") == stage and x.get("attempt") == last.get("attempt")
                  and x.get("passed") is False and x.get("issues")]
        if stage not in {"review", "compile", "visual"} or not failed:
            return {}
        # An interrupted API or compiler process can follow a real earlier verdict.
        # Carry only a verdict that actually explains this terminal failure.
        if not all(str(issue) in reason for row in failed for issue in row["issues"]):
            return {}
        draft = result.get("draft") or {}
        links = {r["block_id"]: r.get("evidence_ids", []) for r in draft.get("evidence_links", [])}
        for row in failed:
            if stage == "review" and row.get("defects"):
                for defect in row["defects"]:
                    field = defect.get("draft_field", "")
                    issues.append({"draft_field": _short(field, 100), "kind": _short(defect.get("kind"), 60),
                                   "issue": _short(defect.get("required_change"), 700),
                                   "evidence_ids": _ids([defect.get("evidence_id"), *links.get(field, [])])})
                continue
            if stage == "compile":
                targets = row.get("repair_targets") or []
                if not targets and draft:
                    from .profile import saved_build_targets
                    targets = saved_build_targets(draft, row["issues"])
                if targets:
                    for target in targets:
                        field = target["draft_field"]
                        issues.append({"draft_field": _short(field, 100),
                                       "issue": _short(" / ".join(target["diagnostics"]), 700),
                                       "evidence_ids": _ids(links.get(field, []))})
                    continue
            locations = {x["issue"]: x["draft_field"] for x in row.get("locations", [])}
            for msg in row["issues"]:
                if stage == "compile" and not re.search(
                    r"\bcitation\b|unsupported TeX (command|environment)|overfull|exceeds? |words?\b|rendered height|references_separate|(?:three|body)_page_limit",
                    msg, re.I
                ):
                    continue
                field = locations.get(msg) or ""
                if not field:
                    match = re.match(r"\[?((?:groups|history|shared_definitions|paragraph|title|authors|references)(?:\.[\w]+)*)\]?(?:\s*:| uses )", msg)
                    field = match[1] if match else ""
                detail = msg.split(" (draft:", 1)[0].split(" [replace:", 1)[0]
                issues.append({"draft_field": field, "issue": _short(detail, 700),
                               "evidence_ids": _ids(re.findall(r"\b[pt]\d+-\d+\b", msg) + links.get(field, []))})
    if not issues:
        return {}
    return {"attempt": attempt, "stage": stage, "issue_count": len(issues), "issues": issues[:12],
            "interpretation": "Questions reported for a previous attempt. Check against the original and v13 independently. "
                              "Field paths refer to the earlier draft. Current selection and wording are decided from source."}


def attach_previous_failure(packet, context, catalog, passages="passages"):
    """Include earlier questions with only source IDs present in this attempt."""
    if not context:
        return
    registered = {r["id"] for r in catalog}
    issues = [{**x, "evidence_ids": [k for k in x["evidence_ids"] if k in registered]} for x in context["issues"]]
    packet["previous_failure"] = {**context, "issues": issues}
    required = {k for issue in issues for k in issue["evidence_ids"][:1]}
    missing = required - {row["id"] for row in packet[passages]}
    packet[passages] = [*packet[passages], *(row for row in catalog if row["id"] in missing)]
