"""Source-bound edits to existing v13 Introduction blocks."""
import re
from copy import deepcopy

from pydantic import ValidationError

from .admission import source_statement_options
from .citation_layout import citation_keys
from .models import MAX_DEFINITIONS, MAX_GROUP_DEFINITIONS, AdmissionReview, AuthoritativeDraft, CurrentDraft, Draft, DraftPatch
from .profile_variants import is_versioned_release
from .profile import bind_formal_citations, render_tex, renderer_context, saved_build_targets
from .reference_patch import ReferencePatch, reference_packet
from .grounded_writing import definition_uses, source_address

WRITER_VERSION = "block-patch-v1"
PATCH_GUIDANCE = (
    "For each block_id in repair_targets, return one replacement with its complete revised text "
    "and registered evidence_ids. Use TeX fragments in text. Return replacements in the DraftPatch schema."
)
_BLOCK_PATH = re.compile(r"paragraph|history\.\d+|shared_definitions\.\d+|references\.\d+\.text|"
                         r"groups\.\d+\.(?:theorem|paragraph|definitions\.\d+)")


def _address(draft, block):
    value = draft
    pieces = [int(k) if k.isdigit() else k for k in block.split(".")]
    for piece in pieces[:-1]:
        value = value[piece]
    return value, pieces[-1]


def _schema(draft):
    if draft.get("core_bindings"):
        return AuthoritativeDraft
    return CurrentDraft if is_versioned_release(draft.get("editorial_profile")) else Draft


def narrow_reviewed_citation(draft, verdict, candidates, records):
    """Apply one exact review edit that drops extra locators from an existing theorem citation."""
    if (verdict.get("stage") != "review" or verdict.get("route") != "draft"
            or verdict.get("passed") is not False or verdict.get("evidence_ids")
            or len(verdict.get("defects", ())) != 1):
        return None
    try:
        review = AdmissionReview(route="draft", issues=verdict["defects"], evidence_ids=[])
    except ValidationError:
        return None
    edit = review.issues[0]
    field = re.fullmatch(r'''formal_sources\[(["'])(groups\.\d+\.theorem)\1\]\.source''', edit.draft_field)
    if (not field or edit.operation != "replace" or edit.kind != "incorrect_attribution_or_locator"
            or review.repair_issues() != verdict.get("issues")):
        return None
    block = field[2]
    citation = draft.get("formal_sources", {}).get(block, {}).get("source")
    if not citation or citation != edit.current_excerpt:
        return None
    cite = r"\\cite\[([^\[\]\n]+)\]\{([A-Za-z0-9:._-]+)\}"
    before = re.fullmatch(cite, citation)
    after = re.fullmatch(cite, edit.replacement_text)
    if not before or not after or before[2] != after[2]:
        return None
    items = re.split(r"\s+and\s+|;\s*|,\s*", before[1])
    locator = re.compile(r"(Theorem|Corollary)[~ ]+(\d+(?:\.\d+)*)(?:\([a-z0-9]+\))?")
    kept = locator.fullmatch(after[1])
    if not kept or len(items) < 2 or items.count(after[1]) != 1 or not all(map(locator.fullmatch, items)):
        return None
    core_id = draft.get("core_bindings", {}).get(block)
    core = next((row for row in candidates if row.get("candidate_id") == core_id), {})
    evidence = next((row["evidence_ids"] for row in draft["evidence_links"] if row["block_id"] == block), ())
    location = edit.evidence_id
    if (location not in records or location not in evidence or location not in core.get("evidence_ids", ())
            or core.get("decision") != "admitted"):
        return None
    # The surviving locator must point to the actual admitted heading in the cited span.
    if not any(option["id"] == core.get("source_statement_id") and option["kind"] == kept[1].lower()
               and option["number"] == kept[2] for option in source_statement_options(records[location])):
        return None
    revised = deepcopy(draft)
    revised["formal_sources"][block]["source"] = edit.replacement_text
    try:
        _schema(revised).model_validate(revised)
    except ValidationError:
        return None
    return revised


def _unused_reference(draft, field, excerpt):
    index = int(field.split(".")[-1])
    references = draft["references"]
    if (index >= len(references) or len(references) <= 1 or references[index]["key"] == "source"
            or references[index]["text"] != excerpt):
        return False
    # The ordinary renderer checks citation keys in bodies, headings and comparisons.
    try:
        render_tex({**draft, "references": references[:index] + references[index + 1:]})
    except ValueError:
        return False
    return True


def repair_targets(draft, verdict, registered, *, allow_collection_edits=False, resolve_fields=False,
                   pair_definition_sources=False):
    """Plan a patch only when every unresolved issue maps to an existing writable block."""
    if not draft or verdict.get("passed") is not False or not verdict.get("issues"):
        return []
    if verdict.get("stage") == "combined":
        merged = {}
        for check in verdict["checks"]:
            rows = repair_targets(draft, check, registered, allow_collection_edits=allow_collection_edits,
                                  resolve_fields=resolve_fields, pair_definition_sources=pair_definition_sources)
            if not rows:
                return []
            for row in rows:
                key = row["block_id"]
                if key not in merged:
                    merged[key] = row
                else:
                    merged[key]["defects"].extend(row["defects"])
                    merged[key]["diagnostics"].extend(row["diagnostics"])
                    merged[key]["required_evidence_ids"] = list(dict.fromkeys(
                        [*merged[key]["required_evidence_ids"], *row["required_evidence_ids"]]))
                    if row.get("missing_reference_keys"):
                        merged[key]["missing_reference_keys"] = sorted(set(
                            merged[key].get("missing_reference_keys", ())) | set(row["missing_reference_keys"]))
                    if row.get("attribution_fields"):
                        merged[key].setdefault("attribution_fields", {}).update(row["attribution_fields"])
        return list(merged.values())
    try:
        schema = _schema(draft)
        parsed = schema.model_validate(draft)
    except ValidationError:
        return []
    links = {link.block_id: link.evidence_ids for link in parsed.evidence_links}
    registered = set(registered)
    if any(not set(locations) <= registered for locations in links.values()):
        return []
    targets = {}

    additions = []

    def target(block, *, allow_append=False):
        if not isinstance(block, str) or not _BLOCK_PATH.fullmatch(block):
            return None
        if block not in targets:
            try:
                parent, key = _address(draft, block)
                append = (allow_append and isinstance(parent, list) and key == len(parent) and
                          bool(re.fullmatch(r"(?:shared_definitions|groups\.\d+\.definitions)\.\d+", block)))
                current = "" if append else parent.get(key, "") if key == "paragraph" else parent[key]
            except (IndexError, KeyError, TypeError):
                return None
            is_empty_paragraph = key == "paragraph" and block.startswith("groups.") and not current
            if not isinstance(current, str) or (block not in links and not append and not is_empty_paragraph and not block.startswith("references.")):
                return None
            if append:
                additions.append(block)
            targets[block] = dict(block_id=block, current_text=current, bound_evidence_ids=links.get(block, []),
                                  required_evidence_ids=[], defects=[], diagnostics=[])
            if block in draft.get("formal_sources", {}):
                targets[block]["formal_source"] = draft["formal_sources"][block]
            if is_empty_paragraph or (key == "paragraph" and block.startswith("groups.")):
                targets[block]["paragraph_position"] = parent.get("paragraph_position", "after")
        return targets[block]

    if verdict.get("stage") == "review" and verdict.get("route") == "draft":
        try:
            review = AdmissionReview(route="draft", issues=verdict.get("defects"), evidence_ids=[])
        except ValidationError:
            return []
        if review.repair_issues() != verdict["issues"] or verdict.get("evidence_ids"):
            return []
        lean = bool(draft.get("core_bindings"))
        # A new definition and its heading may arrive as two JSON edits in either
        # order. The heading alone never authorizes creating a definition.
        inserting = set()
        if pair_definition_sources and lean and allow_collection_edits:
            for edit in review.issues:
                if edit.operation == "insert" and not edit.current_excerpt.strip():
                    path, _ = source_address(draft, edit.draft_field)
                    if re.fullmatch(r"(?:shared_definitions|groups\.\d+\.definitions)\.\d+", path):
                        inserting.add(path)
        for defect in review.issues:
            field = defect.draft_field
            attr = None
            new_heading = False
            if resolve_fields:
                field, attr = source_address(draft, field)
                if inserting and not attr and defect.operation == "insert" and not defect.current_excerpt.strip():
                    for owner in inserting:
                        if field in (f"formal_sources.{owner}.source", f'formal_sources["{owner}"].source',
                                     f"formal_sources['{owner}'].source"):
                            field, new_heading = owner, True
                            break
                if attr:
                    value = draft["formal_sources"][field].get(attr, "as reported in" if attr == "reporting" else "")
                    if (defect.operation not in {"replace", "insert", "remove"}
                            or not isinstance(value, str) or defect.current_excerpt not in value
                            or (not defect.current_excerpt and (value or defect.operation != "insert"))
                            or (defect.current_excerpt and value.count(defect.current_excerpt) != 1)):
                        return []
            insert = defect.operation == "insert" and not defect.current_excerpt.strip()
            extended = allow_collection_edits and lean
            collection = re.fullmatch(r"groups\.(\d+)\.definitions", field)
            if extended and collection and insert:
                index = int(collection[1])
                groups = draft["groups"]
                if index >= len(groups) or (not resolve_fields and len(groups[index]["definitions"]) >= MAX_GROUP_DEFINITIONS):
                    return []
                # An unindexed insert identifies the containing group, not an arbitrary
                # position between existing definitions. Append in the named group.
                field += f".{len(groups[index]['definitions'])}"
                if field in targets:
                    return []
            if extended and defect.operation == "remove" and re.fullmatch(r"references\.\d+", field):
                if (field in targets or defect.evidence_id not in registered
                        or not _unused_reference(draft, field, defect.current_excerpt)):
                    return []
                targets[field] = dict(block_id=field, current_text=defect.current_excerpt,
                                      bound_evidence_ids=[], required_evidence_ids=[defect.evidence_id],
                                      defects=[defect.model_dump()], diagnostics=[], remove=True, unused_reference=True)
                continue
            row = target(field, allow_append=insert)
            if (row is None or defect.operation not in ({"replace", "insert", "remove"} if lean else {"replace", "insert"})
                    or (not lean and defect.kind == "improper_selection" and defect.operation != "replace") or defect.evidence_id not in registered
                    or (not attr and defect.current_excerpt.strip() and defect.current_excerpt not in row["current_text"])):
                return []
            if extended and field in additions:
                row["requires_formal_source"] = True
            edit = defect.model_dump()
            if new_heading:
                edit["draft_field"] = f"formal_sources.{field}.source"
            elif attr:
                row.setdefault("attribution_fields", {})[attr] = value
            elif resolve_fields:
                edit["draft_field"] = field
            row["defects"].append(edit)
            if not attr and lean and defect.operation == "remove" and defect.current_excerpt.strip() == row["current_text"].strip():
                if not re.fullmatch(r"(?:groups\.\d+\.paragraph|(?:shared_definitions|groups\.\d+\.definitions)\.\d+)", field):
                    return []
                row["remove"] = True
            if defect.evidence_id not in row["required_evidence_ids"]:
                row["required_evidence_ids"].append(defect.evidence_id)
    elif verdict.get("stage") in {"compile", "visual"}:
        errors = verdict.get("repair_targets")
        if errors is None and verdict.get("stage") == "compile":
            errors = saved_build_targets(draft, verdict["issues"])
        if not errors:
            return []
        for error in errors:
            if not isinstance(error, dict):
                return []
            row = target(error.get("draft_field"))
            diagnostics = error.get("diagnostics")
            if (row is None or error.get("current_text") != row["current_text"] or not isinstance(diagnostics, list)
                    or not diagnostics or any(not isinstance(msg, str) or not msg.strip() for msg in diagnostics)):
                return []
            row["diagnostics"].extend(diagnostics)
            row["required_evidence_ids"] = list(row["bound_evidence_ids"])
            if error.get("missing_reference_keys"):
                row["missing_reference_keys"] = error["missing_reference_keys"]
    if additions and allow_collection_edits and draft.get("core_bindings"):
        count = len(parsed.shared_definitions) + sum(len(group.definitions) for group in parsed.groups)
        removed_defs = [field for field, item in targets.items() if resolve_fields and item.get("remove")
                        and re.fullmatch(r"(?:shared_definitions|groups\.\d+\.definitions)\.\d+", field)]
        if removed_defs:
            users = definition_uses(draft, [targets[key] for key in removed_defs], renderer_context(draft)["visible_headings"])
            if any(set(use["explicit_users"]) - targets.keys() for use in users.values()):
                # A slot cannot be released locally by leaving a reference to a different numbered definition.
                return []
        delta = {}
        for field, slots in [(field, 1) for field in additions] + [(field, -1) for field in removed_defs]:
            owner = field.rsplit(".", 1)[0]
            delta[owner] = delta.get(owner, 0) + slots
        if (count + len(additions) - len(removed_defs) > MAX_DEFINITIONS or any(
                len(_address(draft, block)[0]) + delta[block.rsplit(".", 1)[0]] > MAX_GROUP_DEFINITIONS
                for block in additions)):
            # Release a slot only when its exact prior owner was explicitly removed in this same review.
            return []
    if additions and not draft.get("core_bindings"):
        prospective = deepcopy(draft)
        for block in additions:
            parent, _ = _address(prospective, block)
            parent.append("Definition")
            prospective["evidence_links"].append({"block_id": block, "evidence_ids": targets[block]["required_evidence_ids"]})
        try:
            schema.model_validate(prospective)
        except ValidationError:
            return []
    return list(targets.values())


def write_block_patch(models, instructions, packet, *, coupled_references=False, records=None):
    schema = DraftPatch
    if coupled_references:
        schema = ReferencePatch
        packet = reference_packet(packet, records or {})
        instructions += (" Every new citation key in a replacement needs its original entry in additional_references "
                         "with its source_id. Reuse an existing key for the same work. If the bibliography entry "
                         "is unavailable, cite the exact reporting passage in the supplied paper.")
    targets = packet["repair_targets"]
    local = [row for row in targets if row.get("unused_reference")]
    if not local:
        return models.structured(schema, instructions, packet)
    remaining = [row for row in targets if not row.get("unused_reference")]
    drops = {int(row["block_id"].split(".")[1]) for row in local}
    model_packet = {**packet, "repair_targets": remaining,
                    "references": (packet["references"] if coupled_references else
                                   [ref for i, ref in enumerate(packet["references"]) if i not in drops])}
    answer = models.structured(schema, instructions, model_packet) if remaining else None
    edits = answer.replacements if answer else []
    removals = [{"block_id": row["block_id"], "evidence_ids": row["required_evidence_ids"], "text": ""}
                for row in local]
    extras = {"additional_references": answer.additional_references} if answer and coupled_references else {}
    return schema(replacements=[*edits, *removals], **extras)


def apply_patch(draft, patch, packet, records):
    """Merge only requested text and its evidence links, preserving all other serialized fields."""
    if isinstance(patch, ReferencePatch):
        packet = reference_packet(packet, records)
    checked = type(patch).model_validate(patch.model_dump(), context=packet)
    result = deepcopy(draft)
    additional = {r.key: r for r in getattr(checked, "additional_references", ())}
    result["references"].extend({"key": key, "text": row.text} for key, row in additional.items())
    links = {link["block_id"]: link for link in result["evidence_links"]}
    removed = []
    targets = {t["block_id"]: t for t in packet["repair_targets"]}
    for row in checked.replacements:
        if targets[row.block_id].get("remove"):
            removed.append(row.block_id)
            continue
        parent, key = _address(result, row.block_id)
        if isinstance(parent, list) and key == len(parent):
            parent.append(row.text)
        else:
            parent[key] = row.text
        if row.paragraph_position is not None:
            parent["paragraph_position"] = row.paragraph_position
        if row.formal_source is not None and "formal_sources" in result:
            result["formal_sources"][row.block_id] = row.formal_source.model_dump()
        evidence_ids = list(row.evidence_ids)
        if additional:
            used = citation_keys(row.text)
            for value in result.get("formal_sources", {}).get(row.block_id, {}).values():
                if isinstance(value, str):
                    used.update(citation_keys(value))
            evidence_ids = list(dict.fromkeys([*evidence_ids, *(r.source_id for key, r in additional.items() if key in used)]))
        if row.block_id in links:
            links[row.block_id]["evidence_ids"] = evidence_ids
        elif not row.block_id.startswith("references."):
            result["evidence_links"].append({"block_id": row.block_id, "evidence_ids": evidence_ids})
    bound = bind_formal_citations(result, records)
    for row in checked.replacements:
        if row.block_id in removed:
            continue
        parent, key = _address(result, row.block_id)
        normalized, _ = _address(bound, row.block_id)
        previous, _ = _address(draft, row.block_id)
        before = ("" if isinstance(previous, list) and key == len(previous)
                  else previous.get(key, "") if key == "paragraph" else previous[key])
        moved = row.paragraph_position is not None and row.paragraph_position != previous.get("paragraph_position", "after")
        citation_changed = row.formal_source is not None and result.get("formal_sources", {}).get(row.block_id) != draft.get("formal_sources", {}).get(row.block_id)
        reference_added = any(k in additional for k in targets[row.block_id].get("missing_reference_keys", []))
        if before == normalized[key] and not moved and not citation_changed and not reference_added:
            raise ValueError(f"{row.block_id}: the requested change is missing after binding the original citation")
        parent[key] = normalized[key]
    # Apply all edits to old coordinates first; deleting list entries earlier
    # would redirect subsequent edits and evidence links to another definition.
    dropped = {}
    for field in sorted(removed, key=lambda v: int(v.split(".")[-1]) if v.split(".")[-1].isdigit() else 0,
                        reverse=True):
        parent, key = _address(result, field)
        if targets[field].get("unused_reference") and not _unused_reference(result, field, targets[field]["current_text"]):
            # A simultaneous change may now cite the entry. Preserve the reference
            # rather than export a dangling citation; review the merged revision.
            continue
        if isinstance(parent, list):
            parent.pop(key)
            dropped.setdefault(field.rsplit(".", 1)[0], []).append(key)
        else:
            parent[key] = ""
    def remap(field):
        prefix, _, number = field.rpartition(".")
        if prefix in dropped and number.isdigit():
            return f"{prefix}.{int(number) - sum(i < int(number) for i in dropped[prefix])}"
        return field
    if removed:
        result["evidence_links"] = [{**row, "block_id": remap(row["block_id"])} for row in result["evidence_links"]
                                    if row["block_id"] not in removed]
        result["formal_sources"] = {remap(key): item for key, item in result["formal_sources"].items() if key not in removed}
    _schema(result).model_validate(result)
    if any(key not in records for link in result["evidence_links"] for key in link["evidence_ids"]):
        raise ValueError("The repaired draft contains unregistered original evidence")
    return result


def preview_exact_patch(draft, target, replacement, packet, records):
    """Preview a local edit on a private copy, using the same renderer as publication."""
    single = {**packet, "repair_targets": [target]}
    preview = apply_patch(draft, DraftPatch(replacements=[replacement]), single, records)
    render_tex(preview)
