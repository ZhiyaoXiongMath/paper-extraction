"""Pass exceptional coverage decisions to the existing writer and reviewer."""
import re

from .repair_progress import block_at


SELECTION_SCOPE = (
    "When omitting a result as covered, identify the retained core and the clause implying the omitted conclusion "
    "throughout its stated range. Check the direction of implication, hypotheses and quantifiers. If only a "
    "special case is covered, assess the remaining scope on its own merits. For a fifth or later result explain "
    "what would still be lost if its essential contribution were stated with the closest retained result under "
    "B5/C5. Incompatible hypotheses or a substantial independent application can warrant a separate block."
)
DIRECT_SCOPE = (
    SELECTION_SCOPE + " Only when relying on coverage, give the original heading being omitted and append "
    "Printed [source_id] «exact covering clause from this core's draft» to that retained core's source_reason. "
    "Bind source_id to the omitted assertion's original passage and include it in evidence_ids."
)
PLANNED_SCOPE = (
    SELECTION_SCOPE + " Only when relying on coverage, name the omitted candidate in the retained core's "
    "source_reason and append Scope [source_id] «precise scope to print». Bind source_id to the omitted "
    "assertion's original passage and include it in evidence_ids. Use exclusions for independently declined candidates."
)
DEFINITION_SCOPE = (
    "If a later claim evaluates an object pointwise, specify the source's representative when it is otherwise "
    "defined only up to equivalence."
)

_COVERAGE = re.compile(r"\b(Printed|Scope)\s+\[([^\[\]\s]+)\]\s*«([^»]+)»")


def coverage_obligations(candidates, draft=None):
    """Bind only explicit coverage promises to their owner, source and current manuscript."""
    blocks = {owner: field for field, owner in (draft or {}).get("core_bindings", {}).items()}
    obligations = []
    for row in candidates:
        if row.get("decision") != "admitted" or row["kind"] not in {"current", "application"}:
            continue
        rationale = row.get("decision_reason", "")
        for tag, source, clause in _COVERAGE.findall(rationale):
            item = {"candidate_id": row["candidate_id"], "source_id": source, "scope": clause,
                    "source_bound": source in row["evidence_ids"]}
            if explanation := _COVERAGE.sub("", rationale).strip():
                item["selection_reason"] = explanation
            if draft is not None:
                field = blocks.get(row["candidate_id"])
                item["bound_field"] = field
                if tag == "Printed":
                    # Only this core's theorem or its own explanation can support its promise.
                    fields = [field] if field else []
                    if field and field.endswith(".theorem"):
                        fields.append(field.removesuffix(".theorem") + ".paragraph")
                    item["verbatim_field"] = next((f for f in fields if clause in (block_at(draft, f) or "")), None)
            obligations.append(item)
    return obligations
