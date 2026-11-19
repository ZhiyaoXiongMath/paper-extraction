"""Lossless wire-shape repairs and safe diagnostics for a whole-source reply."""
import re
from collections import defaultdict

from pydantic import ValidationError

from .focused_evidence import projected_sources
from .repair_progress import block_at


class EvidenceCoverageError(ValueError):
    def __init__(self, blocks):
        self.blocks = blocks
        super().__init__(f"Provide one evidence link for each block: {sorted(blocks)}")


def prepare_evidence_patch(document, blocks, payload):
    """Freeze valid rows. Only a model can choose the evidence for a missing link."""
    draft = document["draft"]
    rows = draft["evidence_links"]
    by_block = defaultdict(list)
    for row in rows:
        by_block[row["block_id"]].append(row)
    kept = [by_block[path][0] for path in sorted(blocks) if len(by_block[path]) == 1]
    targets = sorted(path for path in blocks if len(by_block[path]) != 1)
    core_by_id = {row["candidate_id"]: row for row in document.get("cores", [])}
    group_owners = draft.get("core_bindings", {})
    source_ids = set()
    target_texts = []
    orphan_ids = {sid for row in rows if row["block_id"] not in blocks for sid in row["evidence_ids"]}
    for path in targets:
        group = path.split(".")[:2]
        owner_path = ".".join([*group, "theorem"]) if group[0] == "groups" else path
        core = core_by_id.get(group_owners.get(owner_path), {})
        hints = set(core.get("evidence_ids", [])) | orphan_ids
        for row in rows:
            if row["block_id"].split(".")[:2] == group:
                hints.update(row["evidence_ids"])
        if not hints:
            # Shared context can refer to several results. Give the correction
            # the material already cited by the model, including definition sources.
            hints = {sid for row in [*rows, *core_by_id.values()] for sid in row["evidence_ids"]}
        source_ids.update(hints)
        target_texts.append({"block_id": path, "text": block_at(draft, path),
                             "attribution": draft.get("formal_sources", {}).get(path, {})})
    packet = {"blocks": target_texts, "passages": projected_sources(payload["passages"], source_ids, abridge=True)}
    packet["source_images"] = [r for r in payload.get("source_images", []) if r["source_id"] in source_ids]
    return kept, targets, packet


def with_evidence_links(document, links):
    pending = {row["block_id"]: row for row in links}
    ordered = []
    for row in document["draft"]["evidence_links"]:
        if row["block_id"] in pending:
            ordered.append(pending.pop(row["block_id"]))
    return {**document, "draft": {**document["draft"], "evidence_links": [*ordered, *pending.values()]}}


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


def direct_source_action(document, payload, check_schema):
    """Read the action independently of any unvalidated manuscript fields."""
    if not isinstance(document, dict) or not document.get("source_checks"):
        return [], []
    raw_checks = document["source_checks"]
    if not isinstance(raw_checks, list):
        return [], [{"field": "source_checks", "code": "list_type"}]
    source = {row["id"]: row for row in payload.get("passages", [])}
    source.update({row["id"]: row for row in payload.get("source_locations", []) if row["id"] not in source})
    checks, invalid = [], []
    for i, raw in enumerate(raw_checks):
        field = f"source_checks.{i}"
        try:
            check = check_schema.model_validate(raw)
        except ValidationError as exc:
            loc = exc.errors(include_input=False)[0]["loc"]
            detail = loc[0] if loc and loc[0] in check_schema.model_fields else None
            invalid.append({"field": f"{field}.{detail}" if detail else field, "code": "invalid_source_check"})
            continue
        target = check.target_id or check.source_id
        if check.source_id not in source or target not in source:
            invalid.append({"field": field, "code": "unknown_source_id"})
        elif not check.question.strip() or (check.read_image and not source[target].get("page")):
            invalid.append({"field": field, "code": "invalid_source_check"})
        elif check not in checks:
            checks.append(check)
    return checks, invalid[:16]


def provisional_direct_answer(document):
    return {key: document[key] for key in ("cores", "draft", "summary") if document.get(key)}


def direct_retry_packet(payload, document, diagnostics):
    """Repair the existing response using its cited material and a location index."""
    if not isinstance(document, dict):
        return None
    previous = provisional_direct_answer(document)
    if not previous and not document.get("source_checks"):
        return None
    if document.get("source_checks"):
        previous["source_checks"] = document["source_checks"]
    locations = payload.get("source_locations") or payload.get("passages", [])
    # A finish packet can represent locations already sent with full text by ID.
    # The correction will drop unrelated passages; restore their directory entries.
    by_id = {row["id"]: row for row in payload.get("passages", [])}
    locations = [by_id.get(row["id"], row) if set(row) == {"id"} else row for row in locations]
    used = set()

    def find_ids(value):
        if isinstance(value, str):
            used.add(value)
        elif isinstance(value, list):
            for item in value:
                find_ids(item)
        elif isinstance(value, dict):
            for item in value.values():
                find_ids(item)

    find_ids(previous)
    return {
        "previous_response": previous,
        "previous_schema_failure": diagnostics,
        "source_metadata": payload.get("source_metadata", {}),
        "source_locations": [{k: row[k] for k in ("id", "label", "section", "page", "file", "line") if k in row}
                             for row in locations],
        "passages": [row for row in payload.get("passages", []) if row["id"] in used],
        "source_questions": payload.get("source_questions", []),
        "source_images": payload.get("source_images", []),
        "correction": (
            "Repair the listed fields of previous_response. Preserve verified wording and evidence. "
            "Use registered locations and a concrete question for a source request; set cores to [] "
            "and draft and summary to null in that action. If source_questions already contains answers, "
            "return the corrected decision. Do not invent a fact, location, or uncited hypothesis."
        ),
    }


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
