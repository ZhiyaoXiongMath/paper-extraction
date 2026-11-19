"""Apply a reviewer's exact text edit without asking a writer to restate the block."""
import re

from .citation_layout import citation_keys, named_prerequisite_citations
from .models import BlockReplacement, DraftPatch
from .writer_patch import write_block_patch


def _exact_row(target, keys):
    defects = target.get("defects", ())
    if (not defects or target.get("diagnostics") or target.get("remove")
            or not target.get("current_text") or target["block_id"].startswith("references.")):
        return None
    text = target["current_text"]
    spans = []
    for defect in defects:
        before, after = defect["current_excerpt"], defect["replacement_text"]
        if (defect["operation"] != "replace" or defect["draft_field"] != target["block_id"]
                or not before.strip() or not after.strip() or text.count(before) != 1):
            return None
        # Record edits, citation ownership and paragraph movement need the writer's
        # structured fields. A plain text splice cannot express those operations.
        if re.search(r"bibliograph|bibitem|(?:in|to|from) the heading|paragraph.position|move (?:the|this) paragraph",
                     defect["required_change"], re.I):
            return None
        if citation_keys(after) - keys or re.search(r"\\(?:ref|eqref|pageref|autoref|label)\s*\{", after):
            return None
        start = text.index(before)
        spans.append((start, start + len(before), after))
    spans.sort()
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        return None
    for start, end, replacement in reversed(spans):
        text = text[:start] + replacement + text[end:]
    ids = list(dict.fromkeys(target["bound_evidence_ids"] + target["required_evidence_ids"]))
    return BlockReplacement(block_id=target["block_id"], text=text, evidence_ids=ids)


def write_targeted_patch(models, instructions, packet, records, *, on_exact=None, preview=None):
    keys = {ref["key"] for ref in packet["references"]}
    local, remaining = [], []
    for target in packet["repair_targets"]:
        exact = _exact_row(target, keys)
        if exact and preview:
            if formal := target.get("formal_source"):
                exact = exact.model_copy(update={"text": named_prerequisite_citations(exact.text, formal)})
            # Give a failed local splice to the *existing* scoped writer, with
            # the error from the actual publication renderer and the old evidence.
            try:
                preview(target, exact)
            except ValueError as exc:
                remaining.append({**target, "diagnostics": [str(exc)[:600]]})
                continue
        if exact:
            local.append(exact)
        else:
            remaining.append(target)
    if local and on_exact:
        on_exact([r.block_id for r in local])
    if not remaining:
        return DraftPatch.model_validate({"replacements": local}, context=packet)
    if not local:
        return write_block_patch(models, instructions, {**packet, "repair_targets": remaining},
                                 coupled_references=True, records=records)

    # The writer still sees what the neighbouring definitions and statements will
    # say. The saved draft is updated atomically after all edits are available.
    context = {**packet["context_blocks"], **{r.block_id: r.text for r in local}}
    pending = {**packet, "repair_targets": remaining, "context_blocks": context}
    patch = write_block_patch(models, instructions, pending, coupled_references=True, records=records)
    return patch.model_copy(update={"replacements": [*local, *patch.replacements]})
