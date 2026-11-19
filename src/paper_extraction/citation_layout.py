"""Project one saved draft into source-labelled reading blocks for both TeX and HTML."""
import re

from .citation_types import FormalSource
from .profile_assets.tools.v13_check import ValidationError, balanced
from .profile_inputs import uses_grounded_revision


_CITE = re.compile(r"\\cite\b\s*")
_UNWRAP = re.compile(r"\\begin\s*\{(theorem|historical|definition|introparagraph)\}\s*(.*?)\s*\\end\s*\{\1\}", re.S)
_NUMBER = r"\d+(?:\.\d+)*[a-z]?"
_LABEL = re.compile(rf"(?:(Theorems?|Lemmas?|Corollar(?:y|ies)|Propositions?|Definitions?)\s+)?({_NUMBER})", re.I)
_JOIN = re.compile(r"(?:,\s*(?:and\s+)?|\s+and\s+)")
_AUTHOR = r"[A-Z][A-Za-zÀ-ž'’\\{}~.\-]*(?:\s+(?:and|van|von|de|der|[A-Z][A-Za-zÀ-ž'’\\{}~.\-]*)){0,10}"
SOURCE_ROLE_GUIDANCE = (
    "Put a formal statement's provenance in its formal_source. Use its comparison only for an actual "
    "comparison with earlier work. Keep a connecting paragraph's own locator on that paragraph.")


def tex_citations(text):
    """Read real TeX citations. Brackets in math are left untouched."""
    cursor = 0
    while match := _CITE.search(text, cursor):
        start = match.end()
        locator = ""
        try:
            if text[start:start+1] == "[":
                locator, start = balanced(text, start, "[", "]")
            keys, end = balanced(text, start)
        except ValidationError:
            cursor = match.end()
            continue
        yield match.start(), end, text[match.start():end], keys, locator
        cursor = end


def _identity(text):
    return " ".join(text.replace("~", " ").split())


def citation_keys(text):
    return {key.strip() for _, _, _, keys, _ in tex_citations(text) for key in keys.split(",")}


def bibliography_targets(draft, keys):
    """Locate missing keys before a paragraph's comparison is moved to a printed heading."""
    targets = []
    def inspect(field, text):
        if not isinstance(text, str):
            return
        missing = citation_keys(text) - keys
        attribution = draft.get("formal_sources", {}).get(field, {})
        for value in attribution.values():
            if isinstance(value, str):
                missing.update(citation_keys(value) - keys)
        if missing:
            targets.append({"draft_field": field, "current_text": text, "missing_reference_keys": sorted(missing),
                            "diagnostics": [f"{field} cites an unlisted bibliography key: {key}." for key in sorted(missing)]})
    inspect("paragraph", draft.get("paragraph"))
    for group in ("history", "shared_definitions"):
        for i, text in enumerate(draft.get(group, [])):
            inspect(f"{group}.{i}", text)
    for i, row in enumerate(draft.get("groups", [])):
        for field in ("theorem", "paragraph"):
            inspect(f"groups.{i}.{field}", row.get(field))
        for j, text in enumerate(row.get("definitions", [])):
            inspect(f"groups.{i}.definitions.{j}", text)
    return targets


def _unwrapped(text):
    match = _UNWRAP.fullmatch(text.strip())
    if not match:
        return "", text.strip()
    body = match[2].strip()
    if not body.startswith("["):
        return "", body
    try:
        note, end = balanced(body, 0, "[", "]")
        note = note.strip()
        if note.startswith("{"):
            contents, last = balanced(note, 0)
            if last == len(note):
                note = contents.strip()
        return note, body[end:].lstrip()
    except ValidationError:
        return "", body


def _separate_provenance(text):
    """Recognize explicit attribution/edge locators in older string-only drafts.

    For example, the historical attribution may contain two independent citations.
    Preserve a citation with an unknown grammatical role in its original wording.
    """
    text = text.strip()
    cites = list(tex_citations(text))
    if not cites:
        return text, ""
    # Mask actual citation commands before matching ordinary prose. This prevents
    # parentheses or square brackets inside source locators from becoming delimiters.
    chunks, cursor, commands = [], 0, []
    for i, (start, end, command, _, _) in enumerate(cites):
        chunks.extend((text[cursor:start], f"\x00{i}\x00"))
        commands.append(command)
        cursor = end
    masked = "".join(chunks) + text[cursor:]
    token = r"\x00\d+\x00"
    sequence = token + r"(?:\s*(?:[,;]\s*)?" + token + r")*"
    author_note = (rf"(?:{_AUTHOR}\s*[,~]?\s*)?{sequence}"
                   rf"(?:,\s*(?:as (?:reported|stated|restated|cited) in|cited in|reported in)\s+{sequence})?")

    def unmask(value):
        return re.sub(r"\x00(\d+)\x00", lambda m: commands[int(m[1])], value).strip()

    # A historical report can be an entire parenthesis before or after the statement.
    note = rf"\((?P<note>{author_note})\.?\)"
    prefix = re.match(rf"\\textup\{{{note}\}}", masked) or re.match(note, masked)
    if prefix:
        return unmask(masked[prefix.end():]), unmask(prefix["note"])
    tail = re.search(rf"\s+{note}\.?$", masked)
    if tail:
        return unmask(masked[:tail.start()]), unmask(tail["note"])
    # An explicit reporting prefix has a removable grammatical introducer.
    lead = re.match(rf"^(?:According to\s+(?P<cite>{sequence})|(?P<author>{_AUTHOR}\s+{sequence})"
                    rf"|In the case reported for\s+(?P<reported>{_AUTHOR})\s+(?P<source>{sequence}))"
                    r"\s*[:,]\s+(?P<rest>.+)$", masked, re.S)
    if lead:
        caption = lead["cite"] or lead["author"] or (lead["reported"] + ", as reported in " + lead["source"])
        rest = lead["rest"]
        if rest.startswith(("let ", "the ", "a ", "an ")):
            rest = rest[:1].upper() + rest[1:]
        return unmask(rest), unmask(caption)
    tail = re.search(rf"(?P<gap>\s+)(?P<note>{sequence})(?P<dot>\.?)$", masked)
    if tail:
        body = masked[:tail.start()].rstrip()
        # E.g. “in the sense of [2]” needs its object. Never sever such phrases.
        if not re.search(r"\b(?:of|by|in|from|see|and|using|with|following|to)\s*$", body, re.I):
            if tail["dot"] and body and not body.endswith((".", "!", "?", r"\]")):
                body += "."
            return unmask(body), unmask(tail["note"])
    start = re.match(rf"^(?P<note>{sequence})(?:\s*\.)?\s+(?=[A-Z$\\])", masked)
    if start:
        return unmask(masked[start.end():]), unmask(start["note"])
    return text, ""


def _named_locators(text):
    """Expand only explicit lists such as 'Theorems 5.12 and 5.15' or 'Lemma 3, Theorem 4'."""
    items = []
    kind = ""
    for part in _JOIN.split(text):
        match = _LABEL.fullmatch(part)
        if not match:
            return []
        if match[1]:
            kind = re.sub(r"ies$", "y", match[1].lower()).removesuffix("s").capitalize()
        if not kind:
            return []
        items.append(kind + "~" + match[2])
    return items


def _comparison(additional, primary):
    """Remove only duplicate bibliography locators, including simple numbered lists.

    A paragraph citing Theorems 5.12 and 5.15 in the group for 5.15 contributes
    the 5.12 comparison to that heading. Never infer containment of a range.
    """
    own = {(_identity(keys), _identity(loc)) for _, _, _, keys, loc in tex_citations(primary)}
    own_labels = {(key, label) for key, loc in own for label in _named_locators(loc)}
    comparisons = []
    for _, _, command, keys, loc in tex_citations(additional):
        identity = _identity(keys), _identity(loc)
        if identity in own:
            continue
        labels = _named_locators(identity[1])
        if labels:
            remaining = [label for label in labels if (identity[0], label) not in own_labels]
            if not remaining:
                continue
            if len(remaining) != len(labels):
                command = r"\cite[" + ", ".join(remaining) + "]{" + keys + "}"
            own_labels.update((identity[0], label) for label in remaining)
        if command not in comparisons:
            comparisons.append(command)
        own.add(identity)
    return ", ".join(comparisons)


def reading_blocks(draft, *, strict=False):
    """Return body/caption pairs without modifying the saved source or its evidence links."""
    blocks = []
    formal = draft.get("formal_sources") or {}
    grounded = uses_grounded_revision({"extraction_profile": draft.get("editorial_profile"),
                                       "workflow_revision": draft.get("workflow_revision")})

    def add(block_id, heading, text, *, is_formal=False, comparison=""):
        if not isinstance(text, str) or not text.strip():
            return
        wrapped_caption, body = _unwrapped(text)
        caption = wrapped_caption
        if is_formal:
            statement, extracted = _separate_provenance(body)
            saved = formal.get(block_id)
            if saved:
                entry = FormalSource.model_validate(saved)
                caption = (f"{entry.prior}, {entry.reporting} " if entry.prior else "") + entry.source
                comparison = " ".join((entry.comparison, comparison))
                conflicting = _comparison(" ".join((wrapped_caption, extracted)), caption + " " + comparison)
                if conflicting:
                    if strict:
                        raise ValueError(f"{block_id}: the written attribution {conflicting} differs from formal_sources; reconcile it with the source.")
                    # Saved evidence must remain visible even in a malformed historical draft.
                    if wrapped_caption:
                        caption += "; " + wrapped_caption
                else:
                    body = statement
            elif extracted and (not caption or _identity(extracted) not in _identity(caption)):
                caption = ", ".join(v for v in (caption, extracted) if v)
                body = statement
            else:
                body = statement
            extra = _comparison(comparison, caption)
            if extra:
                caption = "; ".join(v for v in (caption, "compare " + extra) if v)
        blocks.append({"id": block_id, "heading": heading, "text": body, "caption": caption})

    add("paragraph", "Setting", draft.get("paragraph"))
    definition_number = 0

    def definition(block_id, value):
        nonlocal definition_number
        definition_number += 1
        add(block_id, f"Definition {definition_number}", value, is_formal=True)

    for i, value in enumerate(draft.get("shared_definitions", [])):
        definition(f"shared_definitions.{i}", value)
    for i, value in enumerate(draft.get("history", [])):
        add(f"history.{i}", "Theorem " + chr(65 + i), value, is_formal=True)
    for i, group in enumerate(draft.get("groups", [])):
        base = f"groups.{i}."
        _, paragraph = _unwrapped(group.get("paragraph") or "")
        explicit = formal.get(base + "theorem") or {}
        body, edge = _separate_provenance(paragraph)
        if grounded:
            # Only the separate comparison field identifies a comparison with earlier work.
            # An ordinary paragraph may cite its own source even when printed before a theorem.
            comparison = explicit.get("comparison", "")
            # Keep grammatically meaningful words, such as a named author's attribution.
            plain_edge = edge
            for _, _, command, _, _ in tex_citations(edge):
                plain_edge = plain_edge.replace(command, "")
            if comparison and edge and not plain_edge.strip(" ,;.~") and not _comparison(edge, comparison):
                paragraph = body
        else:
            paragraph, comparison = body, edge
        before = group.get("paragraph_position") == "before" and not (comparison or explicit.get("comparison"))
        if before:
            add(base + "paragraph", "", paragraph)
        for j, value in enumerate(group.get("definitions", [])):
            definition(base + f"definitions.{j}", value)
        add(base + "theorem", f"Theorem {i + 1}", group.get("theorem"), is_formal=True, comparison=comparison)
        if not before:
            add(base + "paragraph", "", paragraph)
    return blocks


def visual_page_context(blocks, page_text):
    """Describe actual printed order; include full entries only for headings found on this page."""
    field_index, draft_blocks, displays, positions = {}, {}, {}, {}
    printed_theorems = set()
    found = [i for i, block in enumerate(blocks) if block["heading"]
             and re.search(r"(?im)^\s*" + re.escape(block["heading"]) + r"\b", page_text)]
    # Include the previous block if it continued onto this page and any trailing prose
    # before the next formal heading. A standalone bibliography contains no body blocks.
    first = max(0, found[0] - 1) if found else 0
    last = next((i for i in range(found[-1] + 1, len(blocks)) if blocks[i]["heading"]),
                len(blocks)) if found else len(blocks)
    if re.search(r"(?im)^\s*References\s*$", page_text) and not found:
        last = 0
    for i, block in enumerate(blocks):
        field, heading, text = block["id"], block["heading"], block["text"]
        group = field.rsplit(".", 1)[0]
        if field.endswith(".theorem"):
            printed_theorems.add(group)
        if not first <= i < last:
            continue
        field_index[field] = heading or text[:90]
        if field.endswith(".paragraph"):
            positions[field] = "after" if group in printed_theorems else "before"
        if i in found:
            draft_blocks[field] = text
        # A display can continue onto a page without the block heading. The short environment
        # list identifies it without resending the full text of off-page statements.
        environments = re.findall(r"\\begin\s*\{(conditions|relations)\}", text)
        if environments:
            displays[field] = environments
    return {"field_index": field_index, "draft_blocks": draft_blocks,
            "paragraph_positions": positions, "display_environments": displays}


def reader_draft_view(draft):
    """Expose an ephemeral projection for old and new papers; original downloads remain bound."""
    if not draft or not isinstance(draft, dict):
        return draft
    return {**draft, "reader_blocks": reading_blocks(draft)}
