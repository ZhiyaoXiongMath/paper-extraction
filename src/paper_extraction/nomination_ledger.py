"""Source-bound review nominations. Selection and printed coverage are distinct decisions."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata

from .admission import current_statement, nomination_heading_label, source_statement_options


def _text(value):
    """Ignore TeX typography, retaining math tokens, case and word boundaries."""
    value = unicodedata.normalize("NFC", value)
    value = re.sub(r"([_^])\s*\{\s*([a-zA-Z0-9])\s*\}", r"\1\2", value)
    return re.findall(r"\\[A-Za-z@]+|[^\W\d_]+|\d+|[^\s\w]|_", value)


def _quoted(excerpt, source):
    needle, haystack = _text(excerpt), _text(source)
    return bool(needle) and any(haystack[i:i + len(needle)] == needle
                                for i, token in enumerate(haystack) if token == needle[0])


def _statement_body(record, statement):
    """Multiple theorems can share a source passage, and proofs can quote others."""
    lines = record["text"].splitlines()
    if record.get("page"):
        start = int(statement["id"].rsplit(".l", 1)[1])
        ends = [int(row["id"].rsplit(".l", 1)[1]) - 1 for row in source_statement_options(record)
                if int(row["id"].rsplit(".l", 1)[1]) > start]
        return "\n".join(lines[start - 1:min(ends, default=len(lines))])
    span = next((r for r in record.get("source_statements", []) if r["id"] == statement["id"]), None)
    if span is None:
        return ""
    start = span["line"] - record["line"]
    end = span["end_line"] - record["line"] + 1
    return "\n".join(lines[max(start, 0):min(len(lines), max(end, 0))])


def _identity(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _locate(proposal, records):
    loc, excerpt = proposal["evidence_id"], proposal["source_excerpt"]
    source = records[loc]
    if proposal["scope"] == "historical":
        return None, [loc], _quoted(excerpt, source["text"])
    locator = nomination_heading_label(proposal["source_heading"])
    ids = [loc]
    # A multi-span TeX environment can begin before the quoted record.
    # Search only formal spans in the same file overlapping this location.
    if source.get("line"):
        for other in records.values():
            if other["id"] != loc and other.get("file") == source.get("file") and any(
                    span["line"] <= source["end_line"] and span["end_line"] >= source["line"]
                    for span in other.get("source_statements", ())):
                ids.append(other["id"])
    statement = current_statement({"source_number": locator, "evidence_ids": ids}, records)
    if not statement:
        return None, [loc], False
    origin = next(key for key in ids if any(option["id"] == statement["id"]
                                            for option in source_statement_options(records[key])))
    if origin == loc:
        body = _statement_body(source, statement)
    else:
        span = next(row for row in records[origin]["source_statements"] if row["id"] == statement["id"])
        start = max(0, span["line"] - source["line"])
        end = max(0, span["end_line"] - source["line"] + 1)
        body = "\n".join(source["text"].splitlines()[start:end])
    return statement, list(dict.fromkeys((loc, origin))), _quoted(excerpt, body)


def printed_cores(draft):
    printed = {}
    bindings = draft.get("core_bindings") or {}
    for name, blocks in (("history", draft.get("history", [])), ("groups", draft.get("groups", []))):
        for index, block in enumerate(blocks or []):
            field = f"{name}.{index}"
            if name == "groups":
                field += ".theorem"
                block = block.get("theorem") if isinstance(block, dict) else None
            if field in bindings and isinstance(block, str) and block.strip():
                printed[bindings[field]] = (field, block)
    return printed


def _still_decided(entry, candidates, printed):
    row = candidates.get(entry["candidate_id"], {})
    if entry.get("decision") == "excluded":
        return row.get("decision") == "excluded" and not row.get("represented_by")
    if entry.get("decision") == "represented":
        block = printed.get(entry["represented_by"])
        return bool(block and entry["covered_excerpt"] in block[1])
    if entry.get("decision") == "admitted":
        return row.get("decision") == "admitted"
    return False


def register_nominations(snapshot, proposals, records, previous, candidates, draft):
    """Allocate IDs before any re-read; exact repeats reuse their evidence and disposition."""
    ledger, rows = dict(previous), list(candidates)
    indexed = {row["candidate_id"]: row for row in rows if row.get("candidate_id")}
    sequence = max((int(k[1:]) for k in indexed if re.fullmatch(r"c[1-9]\d*", k)), default=0)
    printed = printed_cores(draft)
    pending = []
    for proposal in proposals:
        loc, heading = proposal["evidence_id"], proposal["source_heading"]
        excerpt, loss = proposal["source_excerpt"], proposal["omission_loss"]
        source = records[loc]
        current = proposal["scope"] == "current"
        statement, evidence_ids, resolved = _locate(proposal, records)
        source_key = statement["id"] if statement else [loc, heading]
        identity = _identity(snapshot, proposal["scope"], source_key, _text(excerpt), _text(loss),
                             loc, _identity(source["text"]))
        if identity in pending:
            continue
        pending.append(identity)
        old = ledger.get(identity, {})
        if old and _still_decided(old, indexed, printed):
            ledger[identity] = {**old, "reused": True}
            continue
        # A candidate ID from the reviewer is a hint, not proof that two
        # conclusions (or two different clauses of one theorem) are the same.
        key = old.get("candidate_id")
        if not key and resolved and statement:
            # Changed evidence may reopen the same claim. Two different
            # omission scopes can quote the same complete compound theorem.
            key = next((entry["candidate_id"] for entry in ledger.values()
                        if entry["snapshot"] == snapshot and entry["source_statement_id"] == statement["id"]
                        and _text(entry["source_excerpt"]) == _text(excerpt)
                        and _text(entry["omission_loss"]) == _text(loss)
                        and indexed.get(entry["candidate_id"], {}).get("decision") != "admitted"), None)
        if not key:
            hint = indexed.get(proposal.get("candidate_id"), {})
            if (hint and hint["decision"] != "admitted" and loc in hint.get("evidence_ids", ())
                    and _text(excerpt) == _text(hint["claim"])
                    and all(_text(entry["omission_loss"]) == _text(loss) for entry in ledger.values()
                            if entry["candidate_id"] == hint["candidate_id"])
                    and (not statement or current_statement(hint, records) == statement)):
                key = hint["candidate_id"]
        if key not in indexed:
            sequence += 1
            key = f"c{sequence}"
            row = {"candidate_id": key, "kind": "current" if current else "historical", "source_number": heading,
                   "claim": excerpt, "assumptions": "", "dependencies": [], "evidence_ids": evidence_ids,
                   "decision": "unresolved", "significance": loss, "decision_reason": loss,
                   "nomination_scope": _identity(_text(loss))}
            if statement:
                row["source_statement_id"] = statement["id"]
                if resolved:
                    row["nomination_statement_id"] = statement["id"]
            rows.append(row)
            indexed[key] = row
        ledger[identity] = {**proposal, "candidate_id": key, "snapshot": snapshot,
                            "source_statement_id": statement["id"] if statement else None,
                            "source_verified": resolved, "decision": "pending", "reused": False}
    return ledger, pending, rows


def selector_nominations(ledger, keys):
    return [{"core_id": row["candidate_id"], "scope": row["scope"], "source_heading": row["source_heading"],
             "evidence_id": row["evidence_id"], "omission_loss": row["omission_loss"]}
            for key in keys if not (row := ledger[key]).get("reused")]


def settle_nominations(ledger, keys, candidates):
    rows = {r["candidate_id"]: r for r in candidates}
    result = dict(ledger)
    for key in keys:
        entry = ledger[key]
        if entry.get("reused"):
            continue
        row = rows.get(entry["candidate_id"])
        if row is None:
            result[key] = {**entry, "decision": "unresolved", "reason": "Nominated candidate missing from selection."}
            continue
        decision = row.get("decision", "unresolved")
        coverage = row.get("nomination_coverage")
        if coverage and decision == "excluded":
            result[key] = {**entry, "decision": "represented", **coverage}
        elif row.get("represented_by") and decision == "excluded":
            # A legacy variant ID by itself cannot certify printed coverage.
            result[key] = {**entry, "decision": "unresolved", "reason": row["decision_reason"]}
        else:
            result[key] = {**entry, "decision": decision, "reason": row["decision_reason"]}
    # Preserve the earlier decision for inspection, while distinguishing a
    # valid result explicitly replaced by a newly approved stronger one.
    for key, entry in tuple(result.items()):
        row = rows.get(entry["candidate_id"], {})
        if entry["decision"] == "admitted" and row.get("superseded_by") and row.get("decision") == "excluded":
            result[key] = {**entry, "decision": "superseded", "superseded_by": row["superseded_by"],
                           "reason": row["decision_reason"]}
    return result


def delivered_nominations(ledger, draft):
    printed = printed_cores(draft)
    result = {}
    for key, row in ledger.items():
        block = printed.get(row.get("represented_by", row["candidate_id"]))
        covered = bool(block and row["decision"] in {"represented", "admitted"})
        if covered and row["decision"] == "represented":
            covered = row["covered_excerpt"] in block[1]
        if covered:
            status = "printed"
        elif row["decision"] == "excluded":
            status = "excluded"
        elif row["decision"] == "superseded" and row.get("superseded_by") in printed:
            status = "superseded"
        elif row["decision"] == "superseded" and block:
            status = "printed"
        elif row["decision"] == "admitted":
            status = "unprinted"
        else:
            status = "unresolved"
        result[key] = {**row, "printed_field": block[0] if block and status == "printed" else None,
                       "publication_state": status}
    return result
