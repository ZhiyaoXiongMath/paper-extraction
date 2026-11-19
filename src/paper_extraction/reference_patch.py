"""Keep a newly cited bibliography entry tied to the supplied original record."""
import re
from copy import deepcopy

from pydantic import Field, ValidationInfo, model_validator

from .citation_layout import citation_keys, tex_citations
from .models import DraftPatch, Reference
from .admission import source_statement_options
from .tex_citations import tex_syntax


class SourceReference(Reference):
    text: str = Field(min_length=1, description="Copy the complete original bibliography entry, omitting its list number or bibitem key.")
    source_id: str = Field(description="ID of the supplied original passage containing this entry.")


class ReferencePatch(DraftPatch):
    additional_references: list[SourceReference] = Field(default_factory=list)

    @model_validator(mode="after")
    def bind_references(self, info: ValidationInfo):
        if info.context is None:
            return self
        old = {r["key"] for r in info.context["references"]}
        additions = {r.key for r in self.additional_references}
        if len(additions) != len(self.additional_references) or additions & old:
            raise ValueError("Only supply a new and unique bibliography key for each new cited work")
        used = set()
        by_id = {t["block_id"]: t for t in info.context["repair_targets"]}
        for row in self.replacements:
            used.update(citation_keys(row.text))
            if row.formal_source:
                attribution = row.formal_source.model_dump()
            else:
                attribution = by_id[row.block_id].get("formal_source") or {}
            for value in attribution.values():
                if isinstance(value, str):
                    used.update(citation_keys(value))
        if used - old != additions:
            raise ValueError(f"Include the original bibliography entry for exactly these new keys: {sorted(used - old)}")
        evidence = {p["id"]: p.get("text", "") for p in
                    [*info.context["passages"], *info.context.get("reference_passages", [])]}
        for item in self.additional_references:
            # Normalise line folding only. Bibliographic facts must be copied from
            # this edition, not reconstructed from the chosen key or an author name.
            original = " ".join(evidence.get(item.source_id, "").split())
            if " ".join(item.text.split()) not in original:
                raise ValueError(f"{item.key}: copy the entry verbatim from its supplied bibliography passage")
        return self


def reference_packet(packet, records):
    """A reviewer can name the bibliography page separately from the theorem page."""
    present = {row["id"] for row in packet["passages"]}
    requested = set()
    for target in packet["repair_targets"]:
        for defect in target.get("defects", []):
            advice = defect.get("required_change", "")
            if not re.search(r"bibliograph|reference|bibitem|\\cite", advice, re.I):
                continue
            requested.update(re.findall(r"(?<![\w-])(?:p\d+|t(?:bib)?\d+)-\d+(?![\w-])", advice))
    extra = [{"id": key, "text": records[key]["text"]} for key in sorted(requested - present) if key in records]
    return {**packet, "reference_passages": extra} if extra else packet


_SECONDARY_EQUATION = re.compile(
    r"(\\cite\[)(?P<primary>(?P<kind>Theorem|Corollary)[~ ]+(?P<number>\d+(?:\.\d+)*))"
    r"\s+and\s+(?:Eq\.|Equation)[~ ]*\(\\ref\{(?P<label>[\w:.-]+)\}\)(\]\{source\})"
)


def _has_label(value, label):
    if isinstance(value, str):
        return bool(label.search(value))
    if isinstance(value, (dict, list)):
        return any(_has_label(part, label) for part in (value.values() if isinstance(value, dict) else value))
    return False


_LOCATOR_ATOM = re.compile(
    r"(?P<kind>Theorem|Corollary)[~\s]+(?P<number>\d+(?:\.\d+)*[a-z]?)\b"
    r"|(?:Eq\.|Equation)[~\s]*\(\s*\\ref\s*\{(?P<label>[\w:.-]+)\}\s*\)"
    r"|(?:(?:Eq\.|Equation)[~\s]*)?\\eqref\s*\{(?P<eq_label>[\w:.-]+)\}", re.I)
_SEPARATOR = re.compile(r"(?:\s*[,;]\s*(?:and\s+)?|\s+and\s+)", re.I)


def _source_locators(text, headings, label_exists):
    """Only edit fully understood locator lists with an independently verified main source."""
    pieces, cursor = [], 0
    for start, end, _, keys, locator in tex_citations(text):
        # 'source' is the workbench's reserved key for this snapshot. The same number in a
        # citation to a different work cannot be verified against this paper's source index.
        if keys.strip() != "source" or not (r"\ref" in locator or r"\eqref" in locator):
            continue
        pos, delimiter, kept = 0, "", []
        verified = removed = False
        while atom := _LOCATOR_ATOM.match(locator, pos):
            if atom["kind"]:
                if (atom["kind"].casefold(), atom["number"]) not in headings:
                    break
                verified = True
                kept.append((delimiter if kept else "") + atom[0])
            elif not verified:
                break  # A leading equation may be the primary source.
            elif label_exists(atom["label"] or atom["eq_label"]):
                kept.append((delimiter if kept else "") + atom[0])
            else:
                removed = True
            pos = atom.end()
            if pos == len(locator):
                if verified and removed:
                    pieces.extend((text[cursor:start], r"\cite[" + "".join(kept) + "]{" + keys + "}"))
                    cursor = end
                break
            separator = _SEPARATOR.match(locator, pos)
            if not separator:
                break
            delimiter, pos = separator[0], separator.end()
    if not pieces:
        return text
    return "".join([*pieces, text[cursor:]])


def _clean_grounded_locators(draft, records):
    linked = {row["block_id"]: row["evidence_ids"] for row in draft["evidence_links"]}
    result, labels = draft, None

    def label_exists(key):
        nonlocal labels
        if labels is None:
            files = {}
            for rec in records.values():
                if rec.get("line"):
                    files.setdefault(rec.get("file", ""), []).append((rec["line"], rec.get("text", "")))
            labels = set()
            for chunks in files.values():
                source = tex_syntax("\n".join(text for _, text in sorted(chunks)))
                labels.update(re.findall(r"\\label\s*\{\s*([\w:.-]+)\s*\}", source))
        return key in labels or _has_label(draft, re.compile(r"\\label\s*\{\s*" + re.escape(key) + r"\s*\}"))

    for i, group in enumerate(draft["groups"]):
        theorem, paragraph = f"groups.{i}.theorem", f"groups.{i}.paragraph"
        explicit = (draft.get("formal_sources") or {}).get(theorem, {})
        for field, text in (("paragraph", group.get("paragraph")), ("comparison", explicit.get("comparison"))):
            if not text or r"\cite" not in text:
                continue
            locs = list(linked.get(paragraph, ()))
            if field == "comparison":
                locs.extend(linked.get(theorem, ()))
            if not locs or any(not records.get(k, {}).get("line") for k in locs):
                continue
            headings = {(h["kind"], h["number"]) for k in locs for h in source_statement_options(records[k])}
            value = _source_locators(text, headings, label_exists)
            if value != text:
                if result is draft:
                    result = deepcopy(draft)
                if field == "paragraph":
                    result["groups"][i][field] = value
                else:
                    result["formal_sources"][theorem][field] = value
    return result


def clean_comparison_locators(draft, records, *, structured=False):
    """A source theorem remains citable when an additional equation label does not exist."""
    if structured:
        return _clean_grounded_locators(draft, records)
    # Only comparison prose is eligible. In particular an equation reference in
    # the mathematical statement, or the sole provenance, needs a real decision.
    linked = {row["block_id"]: row["evidence_ids"] for row in draft["evidence_links"]}
    result = draft
    for i, group in enumerate(draft["groups"]):
        text = group.get("paragraph", "")
        if not text or not _SECONDARY_EQUATION.search(text):
            continue
        source = [records[k] for k in linked.get(f"groups.{i}.paragraph", ()) if k in records]
        if not source or any(not r.get("line") for r in source):
            continue
        headings = {(h["kind"], h["number"]) for row in source for h in source_statement_options(row)}
        def retained(match):
            if (match["kind"].lower(), match["number"]) not in headings:
                return match[0]
            label = re.compile(r"\\label\s*\{\s*" + re.escape(match["label"]) + r"\s*\}")
            if any(label.search(r.get("text", "")) for r in records.values()) or _has_label(draft, label):
                return match[0]
            return match[1] + match["primary"] + "]{source}"
        cleaned = _SECONDARY_EQUATION.sub(retained, text)
        if cleaned != text:
            if result is draft:
                result = deepcopy(draft)
            result["groups"][i]["paragraph"] = cleaned
    return result
