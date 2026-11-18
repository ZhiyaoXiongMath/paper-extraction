"""Lossless wire-shape repairs and safe diagnostics for a whole-source reply."""
import re


def unfence(text):
    match = re.fullmatch(r"\s*\x60{3}(?:json)?[ \t]*\r?\n(.*?)\r?\n[ \t]*\x60{3}\s*", text, re.S | re.I)
    return match[1] if match else text


def normalize_direct_shape(document):
    if not isinstance(document, dict) or not isinstance(document.get("draft"), dict):
        return document
    draft = dict(document["draft"])
    bindings = draft.get("core_bindings")
    if (isinstance(bindings, list) and bindings
            and all(isinstance(r, dict) and set(r) == {"block_id", "candidate_id"}
                    and all(isinstance(v, str) for v in r.values()) for r in bindings)
            and len({r["block_id"] for r in bindings}) == len(bindings)):
        draft["core_bindings"] = {r["block_id"]: r["candidate_id"] for r in bindings}
    links = draft.get("evidence_links")
    if (isinstance(links, dict) and links and all(isinstance(ids, list) for ids in links.values())):
        draft["evidence_links"] = [{"block_id": block, "evidence_ids": ids} for block, ids in links.items()]
    elif isinstance(links, list):
        unique = []
        for row in links:
            # Conflicting links for the same block must go back to the model.
            if not (isinstance(row, dict) and set(row) == {"block_id", "evidence_ids"} and row in unique):
                unique.append(row)
        if len(unique) != len(links):
            draft["evidence_links"] = unique
    return {**document, "draft": draft} if draft != document["draft"] else document


_DOCUMENT_ERRORS = {
    "Number retained cores": "nonconsecutive_core_ids", "Unknown source ID": "unknown_source_id",
    "A qualifying current result": "inconsistent_delivery_type", "Request source evidence": "inconsistent_source_request",
    "Write one formal statement": "core_membership", "Use only registered source IDs": "unregistered_evidence",
    "Provide one evidence link": "evidence_link_coverage", "Bind each printed": "core_binding_coverage",
    "Provide heading attributions": "formal_source_coverage", "Duplicate bibliography keys": "duplicate_reference",
    "Copy evidence_ids": "unregistered_evidence", "The Introduction has more than six": "definition_limit",
    "Cite the examined source": "missing_screening_evidence", "Identify the source with a": "missing_citation",
    "The formal statement needs its own source": "missing_citation",
}

_VALIDATOR_FIELDS = {
    "nonconsecutive_core_ids": "cores[].candidate_id", "unknown_source_id": "evidence_ids",
    "evidence_link_coverage": "evidence_links", "unregistered_evidence": "evidence_links[].evidence_ids",
    "core_binding_coverage": "core_bindings", "core_membership": "draft.core_bindings",
    "formal_source_coverage": "formal_sources", "duplicate_reference": "references[].key",
    "missing_screening_evidence": "summary.evidence_ids",
}


def safe_direct_errors(schema, errors):
    spec = schema.model_json_schema()
    allowed = set(spec.get("properties", {}))
    for obj in spec.get("$defs", {}).values():
        allowed.update(obj.get("properties", {}))
    diagnostics = []
    for error in errors[:16]:
        parts = []
        for part in error["loc"]:
            if isinstance(part, int) or part in allowed:
                parts.append(str(part))
            elif isinstance(part, str) and re.fullmatch(r"(?:history|shared_definitions)\.\d+|groups\.\d+\.(?:theorem|definitions\.\d+)", part):
                parts.append(part)
            else:
                parts.append("*")  # A key supplied by the model may contain arbitrary manuscript text.
        code = error["type"]
        msg = error["msg"].removeprefix("Value error, ")
        if code == "value_error":
            code = next((v for prefix, v in _DOCUMENT_ERRORS.items() if msg.startswith(prefix)), code)
            if code in _VALIDATOR_FIELDS:
                parts.append(_VALIDATOR_FIELDS[code])
            else:
                binding = re.fullmatch(r"(history\.\d+|groups\.\d+\.theorem): "
                                       r"(role differs from approved core|cite a source supporting approved core) c\d+", msg)
                if binding:
                    parts.extend(("draft", "core_bindings", binding[1]))
                    code = "role_mismatch" if binding[2].startswith("role") else "missing_supporting_evidence"
        diagnostics.append({"field": ".".join(parts) or "document", "code": code})
    return diagnostics
