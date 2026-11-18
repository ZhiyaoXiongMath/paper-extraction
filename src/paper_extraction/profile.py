"""Generate an English v13 introduction from bounded, structured model fields."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from urllib.parse import quote

from filelock import FileLock

from .pdfops import page_count, render_page
from .profile_assets.tools import v13_check
from .profile_variants import companion_text, profile_bundle, release_for_draft

ASSETS = Path(__file__).with_name("profile_assets")
PROFILE_FILES = v13_check.FILES
_VALID_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9:._-]*")
_CONTROL = re.compile(r"\\([A-Za-z]+|.)", re.DOTALL)
_ENVIRONMENT = re.compile(r"\s*\{([a-zA-Z]+)\}")
_ENVIRONMENTS = {"conditions", "relations", "aligned", "cases", "matrix", "pmatrix", "bmatrix", "enumerate"}
_COMMANDS = frozenset("""
cite text mbox emph textit textbf textrm textnormal mathbb mathbf mathcal mathfrak mathrm mathsf
mathit mathscr mathop operatorname boldsymbol pmod bmod mod frac dfrac tfrac binom sqrt
overline underline widehat widetilde hat tilde bar dot ddot vec acute grave breve check
partial nabla lim liminf limsup inf sup max min arg det dim ker hom Hom exp log ln sin cos tan
arcsin arccos arctan gcd deg rank tr Re Im ell hbar imath jmath wp
alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa varkappa lambda mu nu
xi pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi psi omega
Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega
le leq leqslant ge geq geqslant ne neq lt gt ll gg lesssim gtrsim equiv sim simeq cong approx
propto asymp prec succ preceq succeq in ni notin subset subseteq subsetneq supset supseteq
varnothing emptyset forall exists nexists infty pm mp times div cdot ast star circ bullet
cap cup setminus smallsetminus wedge vee land lor neg perp parallel mid nmid lvert rvert
lVert rVert vert Vert lceil rceil lfloor rfloor langle rangle lbrace rbrace
bigcup bigcap bigvee bigwedge bigoplus bigotimes oplus otimes odot ominus sum prod coprod
int iint iiint iiiint idotsint oint bigsqcup sqcup sqcap amalg sqsubseteq
to mapsto gets longmapsto rightarrow leftarrow leftrightarrow longrightarrow longleftarrow
longleftrightarrow Rightarrow Leftarrow Leftrightarrow Longrightarrow Longleftarrow Longleftrightarrow
uparrow downarrow hookrightarrow hookleftarrow twoheadrightarrow rightharpoonup leadsto
longhookrightarrow iff implies ldots cdots vdots ddots dots dotsb dotsc
left right middle big Big bigg Bigg bigl bigr Bigl Bigr biggl biggr Biggl Biggr
quad qquad thinspace substack overset underset stackrel limits nolimits overrightarrow
S P item label ref eqref tag sec csc cot tanh sinh cosh
rm texttt textsc textsl textsf textup textsuperscript textsubscript textbackslash textasciitilde textasciicircum symbol mathbin mathrel mathopen mathclose
mathord mathpunct mathinner not thickapprox ensuremath mathnormal
displaystyle textstyle scriptstyle scriptscriptstyle lessgtr gtrless
longleftrightarrow varinjlim varprojlim injlim projlim
genfrac dbinom tbinom cfrac xrightarrow xleftarrow overleftrightarrow overleftarrow
underrightarrow underleftarrow underleftrightarrow dotsm dotsi dotso mathstrut
underbrace overbrace phantom mathring mathchoice complement wr
aleph beth hslash eth imath jmath wp emptyset diagup diagdown square
bigtriangleup bigtriangledown top bot lozenge angle sphericalangle
aa AA ae AE oe OE o O l L ss i j c H v u r b d t
""".split())
# Primitive glyph names declared by the standard LaTeX, AMS and newtx fonts.
# These are data, separate from the executable template and the four v13 companions.
_COMMANDS |= frozenset((ASSETS / "math-symbols.txt").read_text(encoding="ascii").split())
_CONTROL_SYMBOLS = frozenset(r"""!,:; {}#$%&_()/[]\"'`\^~=.|\->""")
_ALIGN = {"conditions", "relations", "aligned", "cases", "matrix", "pmatrix", "bmatrix"}
_TEXT_REPLACE = {"–": "--", "—": "---", "’": "'", "‘": "'", "“": "\x60\x60", "”": "''", "\u00a0": "~"}


def _list(value, field: str, low=0, high=None):
    if not isinstance(value, list) or len(value) < low or (high is not None and len(value) > high):
        bound = f"{low}–{high}" if high is not None else f"at least {low}"
        raise ValueError(f"{field} must contain {bound} items.")
    return value


def _line_ranges(ranges):
    merged = []
    for first, last in sorted(ranges):
        if merged and first <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], last)
        else:
            merged.append([first, last])
    return ", ".join(str(first) if first == last else f"{first}--{last}" for first, last in merged)


def _file_locator(name):
    # The English v13 preamble cannot typeset arbitrary Unicode paths. A URI
    # path retains every original UTF-8 byte when such a filename is needed.
    prefix = ""
    if not name.isascii():
        prefix, name = "URI path ", quote(name, safe="/")
    escapes = {char: "\\" + char for char in r"#$%&_{}"}
    escapes.update({"[": r"\symbol{91}", "]": r"\symbol{93}", "\\": r"\textbackslash{}",
                    "^": r"\textasciicircum{}", "~": r"\textasciitilde{}"})
    path = "".join(escapes.get(char, char) for char in name)
    return prefix + r"\texttt{" + path + "}"


def _bound_locator(ids, records, environment=""):
    by_file = {}
    locations = []
    for loc in ids:
        record = records.get(loc, {})
        page = record.get("page")
        if type(page) is int and page >= 1:
            by_file.setdefault(None, []).append((page, page))
            continue
        filename, start, end = record.get("file"), record.get("line"), record.get("end_line")
        if not filename or type(start) is not int or type(end) is not int or not (1 <= start <= end):
            return ""
        locations.append(record)
        by_file.setdefault(filename, []).append((start, end))
    if locations and None not in by_file:
        expected = {"definition"} if environment == "definition" else {"theorem", "proposition", "corollary", "lemma"}
        exact = {(ref["kind"], ref["number"]) for record in locations for ref in record.get("citation_statements", [])
                 if ref.get("contains_span") and ref["kind"].casefold() in expected}
        if environment in {"definition", "theorem"} and len(exact) == 1:
            kind, number = exact.pop()
            return kind + "~" + number
        sections = []
        for record in locations:
            section = record.get("citation_section") or {}
            title = section.get("title", "")
            if number := section.get("number"):
                sections.append(r"\S~" + number)
            else:
                sections.append(title if re.fullmatch(r"[A-Za-z][A-Za-z0-9 .,:()-]*", title) else "")
        if all(sections):
            return "; ".join(dict.fromkeys(sections))
    parts = []
    for filename, ranges in by_file.items():
        if filename is None:
            label = "PDF physical p." if len(set(ranges)) == 1 else "PDF physical pp."
        else:
            label = _file_locator(filename) + " L"
        parts.append(label + "~" + _line_ranges(ranges))
    return "; ".join(parts)


def _rewrite_working_locators(text, locator, ids, records):
    if not locator:
        return text
    label_locators = {statement["label"]: statement["kind"] + "~" + statement["number"]
                      for key in ids for statement in records.get(key, {}).get("citation_statements", [])
                      if statement.get("label")}
    for key in ids:
        section = records.get(key, {}).get("citation_section", {})
        if section.get("label") and section.get("number"):
            label_locators[section["label"]] = r"\S~" + section["number"]
    pieces = []
    start = 0
    for match in re.finditer(r"\\cite\b\s*(?=\[)", text):
        try:
            old, cursor = v13_check.balanced(text, match.end(), "[", "]")
            keys, end = v13_check.balanced(text, cursor)
        except v13_check.ValidationError:
            continue
        if keys.strip() != "source":
            continue
        hint = re.fullmatch(r"(?:\\(?:auto)?ref\s*\{([^{}]+)\}|([\w.:-]+))", old.strip())
        resolved = label_locators.get(hint[1] or hint[2]) if hint else None
        line_locator = re.search(r"\.tex\b|\bL\s*~?\s*\d|\blines?\s*~?\s*\d", old, re.I)
        if resolved or (line_locator and not re.search(r"\.tex\b|\bL\s*~?\s*\d", locator, re.I)):
            pieces.append(text[start:match.start()])
            pieces.append(r"\cite[" + (resolved or locator) + "]{source}")
            start = end
    pieces.append(text[start:])
    return "".join(pieces)


def bind_formal_citations(draft: dict, records: dict[str, dict]) -> dict:
    """Fill formal citations and replace TeX navigation details when a printed locator is available."""
    result = deepcopy(draft)
    if not any(ref.get("key") == "source" for ref in result.get("references", [])):
        return result
    links = {row["block_id"]: row["evidence_ids"] for row in result.get("evidence_links", [])}

    def bind(block, text, environment):
        ids = links.get(block, [])
        locator = _bound_locator(ids, records, environment)
        if not text or re.search(r"\\cite\b", text):
            return _rewrite_working_locators(text, locator, ids, records) if text else text
        if not locator or environment == "introparagraph":
            return text
        citation = r" \cite[" + locator + "]{source}"
        wrapped = re.fullmatch(r"(\s*\\begin\s*\{" + environment + r"\}.*?)(\s*\\end\s*\{"
                               + environment + r"\}\s*)", text, re.S)
        if wrapped:
            return wrapped[1] + citation + wrapped[2]
        return text + citation

    for field, environment in (("history", "historical"), ("shared_definitions", "definition")):
        for i, text in enumerate(result.get(field, [])):
            result[field][i] = bind(f"{field}.{i}", text, environment)
    for i, group in enumerate(result.get("groups", [])):
        for j, text in enumerate(group.get("definitions", [])):
            group["definitions"][j] = bind(f"groups.{i}.definitions.{j}", text, "definition")
        group["theorem"] = bind(f"groups.{i}.theorem", group["theorem"], "theorem")
        if group.get("paragraph"):
            group["paragraph"] = bind(f"groups.{i}.paragraph", group["paragraph"], "introparagraph")
    if result.get("paragraph"):
        result["paragraph"] = bind("paragraph", result["paragraph"], "introparagraph")
    return result


def bind_source_reference(draft, metadata):
    """Render the supplied manuscript's bibliography entry independently of its editorial notes."""
    result = deepcopy(draft)
    source_entry = next((item for item in result["references"] if item["key"] == "source"), None)
    if source_entry is None:
        return result
    has_front_matter = bool(metadata.get("title") and metadata.get("authors"))
    circular = re.search(r"\\cite\s*(?:\[.*?\])?\s*\{source\}", source_entry["text"], re.S)
    if not has_front_matter and not circular:
        # A bare fragment or PDF may not expose structured front matter. Keep the
        # source-supported bibliography the writer assembled, including dates.
        return result
    title = metadata.get("title") or draft["title"]
    authors = metadata.get("authors") or draft["authors"]
    author_text = authors[0]
    if len(authors) > 1:
        conjunction = " and " if len(authors) == 2 else ", and "
        author_text = ", ".join(authors[:-1]) + conjunction + authors[-1]
    citation = author_text + r", \emph{" + title + "}"
    if metadata.get("arxiv"):
        citation += ", arXiv:" + metadata["arxiv"]
    if metadata.get("date"):
        citation += ", " + metadata["date"].rstrip(".")
    citation += "."
    if result["history"]:
        # C2: the manuscript supplies the prior attributions in this single-paper workflow.
        citation += " Historical attributions follow this manuscript."
    try:
        _fragment(citation, "source bibliography", {"source"}, allow_blocks=False)
    except ValueError:
        if metadata.get("title") or metadata.get("authors") or metadata.get("date"):
            # Manuscript-specific macros belong to the input, so use the writer's equivalent front matter.
            return bind_source_reference(draft, {"arxiv": metadata.get("arxiv")})
        raise
    source_entry["text"] = citation
    return result


def _citation(text: str, after_command: int, field: str, keys: set[str]) -> tuple[str, int]:
    cursor = after_command
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    command = r"\cite"
    if text[cursor:cursor+1] == "[":
        locator, cursor = v13_check.balanced(text, cursor, "[", "]")
        clean = _fragment(locator, field, keys, allow_blocks=False) if locator.strip() else ""
        # TeX delimited optional arguments stop at the first ungrouped ']'.
        # A locator may legitimately mention an original bibliography number.
        if "[" in clean or "]" in clean:
            clean = re.sub(r"(\\(?:left|right|middle|bigl|bigr|Bigl|Bigr))\s*([\[\]])",
                           lambda m: m[1] + (r"\lbrack " if m[2] == "[" else r"\rbrack "), clean)
            # Group each literal pair. Grouping the entire locator alone loses
            # protection when the LaTeX/hyperref citation macros forward it.
            chars = list(clean)
            openings = []
            for pos, char in enumerate(clean):
                if char == "[":
                    openings.append(pos)
                elif char == "]" and openings:
                    start = openings.pop()
                    if clean[max(0, start - 1):start] != "{" or clean[pos + 1:pos + 2] != "}":
                        chars[start], chars[pos] = "{[", "]}"
            clean = "".join(chars)
        command += "[" + clean + "]"
    labels, end = v13_check.balanced(text, cursor)
    cited = [key.strip() for key in labels.split(",")]
    if not all(key in keys for key in cited):
        raise ValueError(f"{field} cites an unlisted bibliography key.")
    return command + "{" + ",".join(cited) + "}", end


def _fragment(value, field: str, keys: set[str], *, formal=False, allow_blocks=True) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text.")
    if "^^" in value or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u202a-\u202e\u2066-\u2069]", value):
        raise ValueError(f"{field} contains an unsupported control character or TeX escape.")
    result = []
    environments = []
    depth = 0
    has_citation = False
    i = 0
    while i < len(value):
        char = value[i]
        if char == "\\":
            command = _CONTROL.match(value, i)
            if not command:
                raise ValueError(f"{field} ends with an incomplete TeX command.")
            name = command[1]
            i = command.end()
            if name == "cite":
                cite, i = _citation(value, i, field, keys)
                result.append(cite)
                has_citation = True
                continue
            if name in ("begin", "end"):
                env = _ENVIRONMENT.match(value, i)
                if not allow_blocks or not env or env[1] not in _ENVIRONMENTS:
                    raise ValueError(f"{field} uses an unsupported TeX environment.")
                if name == "begin":
                    environments.append((env[1], depth))
                elif not environments or environments.pop() != (env[1], depth):
                    raise ValueError(f"{field} contains mismatched environments.")
                result.append(command[0] + env[0])
                i = env.end()
                continue
            if name not in _COMMANDS and name not in _CONTROL_SYMBOLS:
                raise ValueError(f"{field} uses unsupported TeX command {command[0]}. Use standard notation.")
            result.append(command[0])
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                raise ValueError(f"{field} has an extra closing brace.")
        if char in "%#" or (char == "&" and not any(e in _ALIGN for e, _ in environments)):
            result.append("\\" + char)
        else:
            result.append(_TEXT_REPLACE.get(char, char))
        i += 1
    if depth or environments:
        raise ValueError(f"{field} has unbalanced braces or environments.")
    if formal and not has_citation:
        raise ValueError(f"{field} needs a citation identifying its source location, e.g. \\cite[p. 1]{{source}}.")
    return "".join(result).strip()


def renderer_context(draft: dict | None, candidates=()) -> dict:
    """Describe the printed numbering and the actual definition scopes of a v13 draft."""
    draft = draft or {}
    labels = {}
    shared = []
    groups = []
    count = 0
    for index, _ in enumerate(draft.get("shared_definitions", [])):
        block = f"shared_definitions.{index}"
        count += 1
        labels[block] = f"Definition {count}"
        shared.append(block)
    for index, _ in enumerate(draft.get("history", [])):
        labels[f"history.{index}"] = f"Theorem {chr(ord('A') + index)}"
    for index, group in enumerate(draft.get("groups", [])):
        formal = f"groups.{index}.theorem"
        local = []
        for j, _ in enumerate(group.get("definitions", [])):
            block = f"groups.{index}.definitions.{j}"
            count += 1
            labels[block] = f"Definition {count}"
            local.append(block)
        labels[formal] = f"Theorem {index + 1}"
        groups.append({"theorem_block_id": formal, "definitions_printed_in_this_group": local})
    by_id = {c["candidate_id"]: c for c in candidates if c.get("candidate_id")}
    dependencies = [{
        "definition_candidate_id": key,
        "definition_source_ids": row.get("evidence_ids", []),
        "needed_by_candidates": [{"candidate_id": owner, "source_number": by_id[owner].get("source_number")}
                                 for owner in row["needed_by"] if owner in by_id],
    } for key, row in by_id.items() if row.get("needed_by")]
    return {
        "numbering": ("The renderer prints shared definitions first, then group definitions, numbered "
                      "continuously from Definition 1. Historical headings are Theorem A, B, C. Current "
                      "headings are Theorem 1, 2, ... . Draft text can identify a needed definition by "
                      "its precise mathematical name or its printed number. The template prints these "
                      "numbers automatically. The Draft schema has no required label/ref field. "
                      "The theorem text determines which defined concepts it actually identifies."),
        "visible_headings": labels, "shared_definition_blocks": shared, "result_groups": groups,
        "source_candidate_dependencies": dependencies,
    }


def render_tex(draft: dict, *, block_map: list | None = None, _legacy_citations: bool = False,
               allow_empty_history: bool = False) -> str:
    if not isinstance(draft, dict):
        raise ValueError("The model draft must be an object.")
    refs = _list(draft.get("references"), "references", 1)
    keys = set()
    for ref in refs:
        if not isinstance(ref, dict) or not isinstance(ref.get("key"), str):
            raise ValueError("Each bibliography item needs a string key.")
        key = ref["key"]
        if not _VALID_KEY.fullmatch(key) or key in keys:
            raise ValueError(f"Invalid or duplicate bibliography key {key!r}.")
        keys.add(key)

    title = _fragment(draft.get("title"), "title", keys, allow_blocks=False)
    authors = [_fragment(a, "authors", keys, allow_blocks=False)
               for a in _list(draft.get("authors"), "authors", 1)]
    header = companion_text(ASSETS, "template-v13.tex", release_for_draft(draft)).split(r"\begin{document}", 1)[0]
    parts = [header, r"\begin{document}", rf"\paperheader{{{title}}}{{{', '.join(authors)}}}",
             r"\section{Introduction}"]

    def block(env, val, field, draft_field):
        original = val
        caption = ""
        # A provider sometimes returns the exact public environment shown in the
        # template. Unwrap only the single outer environment for this field. Its
        # caption and its body still pass the same independent fragment checks.
        wrapped = re.fullmatch(r"\s*\\begin\s*\{" + env + r"\}\s*(.*?)\s*\\end\s*\{"
                               + env + r"\}\s*", val, re.S) if isinstance(val, str) else None
        if wrapped:
            val = wrapped[1]
            if env != "introparagraph" and val.startswith("["):
                raw_caption, end = v13_check.balanced(val, 0, "[", "]")
                if raw_caption.strip():
                    caption = _fragment(raw_caption, field + " caption", keys, allow_blocks=False)
                val = val[end:].lstrip()
        caption_has_source = bool(re.search(r"\\cite\b", caption))
        body = _fragment(val, field, keys, formal=env != "introparagraph" and not caption_has_source)
        opening = rf"\begin{{{env}}}"
        if env != "introparagraph" and not caption_has_source:
            locator = list(re.finditer(r"\\cite\b", body))[-1]
            citation, end = _citation(body, locator.end(), field, keys)
            prefix = re.match(r"(?:\s*\.)?[ \t\r\n~]+(?=[A-Z])", body[end:])
            leading = _legacy_citations is not True and not body[:locator.start()].strip() and prefix
            trailing = not body[end:].strip(" .\t\r\n")
            if leading or trailing or _legacy_citations:
                caption = (caption + " " if caption else "") + citation
            if leading:
                # A standalone prefixed source locator becomes the printed caption.
                # Embedded citations retain their grammatical role in the prose.
                body = body[end + prefix.end():]
            elif trailing:
                tail = body[end:].strip()
                body = body[:locator.start()].rstrip(" ~\r\n\t")
                if tail == "." and not body.endswith("."):
                    body += "."
        elif caption_has_source and not _legacy_citations:
            # Keep a grammatical attribution in the body if the same locator is
            # already present there. An independent caption still stays visible.
            citation_match = re.search(r"\\cite\b", caption)
            citation, end = _citation(caption, citation_match.end(), field, keys)
            if citation in body and not caption[end:].strip() and len(list(re.finditer(r"\\cite\b", caption))) == 1:
                caption = caption[:citation_match.start()].rstrip(" ,;~")
        if caption:
            opening += "[{" + caption + "}]"
        if block_map is not None:
            first = 1 + sum(part.count("\n") + 1 for part in parts)
            block_map.append({"environment": env, "draft_field": draft_field, "current_text": original,
                              "first": first, "last": first + opening.count("\n") + body.count("\n") + 2})
        parts.extend([opening, body, rf"\end{{{env}}}"])

    block("introparagraph", draft.get("paragraph"), "paragraph", "paragraph")
    definitions = _list(draft.get("shared_definitions", []), "shared_definitions", 0, 2)
    count = len(definitions)
    for i, value in enumerate(definitions):
        block("definition", value, "shared_definitions", f"shared_definitions.{i}")
    for i, value in enumerate(_list(draft.get("history", []), "history", 0 if allow_empty_history else 1, 3)):
        block("historical", value, "history", f"history.{i}")
    for number, group in enumerate(_list(draft.get("groups"), "groups", 1), 1):
        if not isinstance(group, dict):
            raise ValueError(f"groups[{number}] must be an object.")
        parts.append(r"\begin{resultgroup}")
        group_id = f"groups.{number-1}"
        paragraph = group.get("paragraph")
        before = group.get("paragraph_position") == "before"
        if paragraph and before:
            block("introparagraph", paragraph, f"group {number} paragraph", f"{group_id}.paragraph")
        for j, value in enumerate(_list(group.get("definitions", []), "group definitions", 0, 2)):
            count += 1
            block("definition", value, f"group {number} definitions", f"{group_id}.definitions.{j}")
        block("theorem", group.get("theorem"), f"group {number} theorem", f"{group_id}.theorem")
        if paragraph and not before:
            block("introparagraph", paragraph, f"group {number} paragraph", f"{group_id}.paragraph")
        parts.append(r"\end{resultgroup}")
    if count > 6:
        raise ValueError("The v13 profile allows a maximum of six definitions.")
    parts.extend([r"\clearpage", rf"\begin{{thebibliography}}{{{len(refs)}}}"])
    for ref in refs:
        content = _fragment(ref.get("text"), "bibliography", keys, allow_blocks=False)
        parts.append(rf"\bibitem{{{ref['key']}}} {content}")
    parts.extend([r"\end{thebibliography}", r"\end{document}"])
    return "\n".join(parts) + "\n"


class BuildFailure(ValueError):
    def __init__(self, message, repair_targets, *, case_dir=None, technical=False):
        super().__init__(message)
        self.repair_targets = repair_targets
        self.case_dir = case_dir
        self.technical = technical


def _build_targets(blocks, issues, log="", *, release=v13_check.RELEASE):
    expanded = release == v13_check.EXPANDED_RELEASE
    limits = v13_check.EXPANDED_BLOCK_LIMITS if expanded else v13_check.BLOCK_LIMITS
    group_limit = v13_check.EXPANDED_GROUP_HEIGHT if expanded else 22
    targets = {}
    for detail in (part.strip() for issue in issues for part in issue.split(" | ")):
        if detail.startswith("Engineering checks failed:"):
            failed = {check.strip() for check in detail.split(":", 1)[1].split(",")}
            if failed <= {"no_blocking_diagnostics", "v13_reader_format_contract"}:
                continue
        if re.fullmatch(r"pdfLaTeX failed on pass \d+, exit \d+; see log", detail):
            continue
        if detail.endswith("Fatal error occurred, no output PDF file produced!"):
            continue
        matched = []
        indexed = re.fullmatch(r"(introparagraph|historical|theorem|definition)_(\d+)_"
                               r"(word_limit:\d+>\d+|display_budget|missing_source_locator|manual_spacing)", detail)
        box = re.fullmatch(r"Overfull \\[hv]box .*?at lines? (\d+)(?:--(\d+))?\.?", detail)
        measured = re.search(r"Package intro-v13 Error: Block (\d+) \(([PDHT])\) exceeds (\d+)U", detail)
        aggregate = re.search(r"(Total connecting prose|Shared definitions) exceeds (\d+)U", detail)
        if indexed:
            same_kind = [row for row in blocks if row["environment"] == indexed[1]]
            i = int(indexed[2]) - 1
            if 0 <= i < len(same_kind):
                matched = [same_kind[i]]
                if indexed[3].startswith("word_limit:"):
                    actual, limit = indexed[3].split(":")[1].split(">")
                    detail += f". Shorten this block from {actual} prose words to at most {limit}, retaining necessary hypotheses and conclusions."
                elif indexed[3] == "display_budget":
                    cap = limits[indexed[1]]
                    detail += f". Use at most {cap['displays']} display groups and {cap['rows']} display source rows."
                elif indexed[3] == "manual_spacing":
                    detail += ". Replace stacked spacing or explicit horizontal fills with normal math spacing or aligned rows."
        elif box:
            first, last = int(box[1]), int(box[2] or box[1])
            matched = [row for row in blocks if row["first"] <= first <= last <= row["last"]]
            if r"\hbox" in detail:
                detail += ". Fit the affected prose or formula within the text width using the permitted display rows."
        elif measured:
            i = int(measured[1]) - 1
            if 0 <= i < len(blocks) and limits[blocks[i]["environment"]]["kind"] == measured[2]:
                matched = [blocks[i]]
                reading = re.search(rf"(?m)^INTRO-BLOCK\|{i+1}\|{measured[2]}\|(\d+)\|(\d+)\|", log)
                if reading and int(reading[2]):
                    actual = int(reading[1]) * int(measured[3]) / int(reading[2])
                    detail += f" (rendered height {actual:.3f}U, limit {measured[3]}U)"
        elif aggregate or detail in {"rendered_total_prose_height_exceeded", "rendered_shared_definitions_height_exceeded"}:
            prose = (aggregate[1] == "Total connecting prose" if aggregate
                     else detail == "rendered_total_prose_height_exceeded")
            matched = [row for row in blocks if (row["environment"] == "introparagraph" if prose
                       else row["draft_field"].startswith("shared_definitions."))]
            unit = re.search(r"(?m)^INTRO-UNIT-SP:\s*([1-9]\d*)\s*$", log)
            measured_blocks = re.findall(r"(?m)^INTRO-BLOCK\|\d+\|([DP])\|(\d+)\|\d+\|(\d+)\s*$", log)
            heights = [int(height) for kind, height, group in measured_blocks
                       if (kind == "P" if prose else kind == "D" and group == "0")]
            if unit and heights:
                limit = (int(aggregate[2]) if aggregate else
                         v13_check.EXPANDED_PROSE_HEIGHT if prose else v13_check.EXPANDED_SHARED_HEIGHT)
                detail += f" (rendered height {sum(heights) / int(unit[1]):.3f}U, limit {limit}U)"
        elif (detail in {"rendered_block_height_exceeded", "resultgroup_measurement_failed"}
              or re.search(r"Package intro-v13 Error: Result group exceeds \d+U", detail)):
            group = detail != "rendered_block_height_exceeded"
            prefix = "INTRO-GROUP" if group else "INTRO-BLOCK"
            pattern = (rf"(?m)^{prefix}\|(\d+)\|" + ("" if group else "[PDHT]\\|") + r"(\d+)\|(\d+)\|")
            for number, height, limit in re.findall(pattern, log):
                i = int(number) - 1
                if int(height) <= int(limit) or int(limit) <= 0:
                    continue
                scoped = ([row for row in blocks if row["draft_field"].startswith(f"groups.{i}.")]
                          if group else blocks[i:i+1])
                for row in scoped:
                    units = group_limit if group else limits[row["environment"]]["units"]
                    measured_height = int(height) * units / int(limit)
                    note = f"{detail} (rendered height {measured_height:.3f}U, limit {units}U)"
                    targets.setdefault(row["draft_field"], []).append(note)
                    matched.append(row)
            if matched:
                continue
        if not matched:
            # A global or ambiguous problem still goes through the ordinary full-draft repair.
            return []
        for row in matched:
            targets.setdefault(row["draft_field"], []).append(detail)
    return [{"draft_field": row["draft_field"], "current_text": row["current_text"],
             "diagnostics": list(dict.fromkeys(targets[row["draft_field"]]))}
            for row in blocks if row["draft_field"] in targets]


def saved_build_targets(draft, issues):
    """Recover exact fields from this saved draft and its v13 diagnostics."""
    blocks = []
    if not issues or any(not isinstance(issue, str) or len(issue) >= 3000 for issue in issues):
        return []  # Old review traces truncate each diagnostic at this length.
    try:
        render_tex(draft, block_map=blocks)
    except (ValueError, TypeError, KeyError):
        return []
    return _build_targets(blocks, issues, release=release_for_draft(draft))


def _diagnosis(case: Path, status: dict) -> str:
    details = [status.get("error", "v13 verification failed")]
    if status.get("body_page_limit") is False and status.get("main_text_pages"):
        details.append(f"Introduction body has {status['main_text_pages']} pages, limit {status['max_body_pages']}.")
    details.extend(status.get("format_contract", {}).get("errors", []))
    details.extend(status.get("blocking_diagnostics", []))
    log = case / "introduction.log"
    if log.is_file() and not status.get("compiled"):
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("!") or re.search(r"\.tex:\d+:|Package intro-v13 Error", line):
                details.append(line)
    return " | ".join(map(str, details))


def _compile_case(case: Path, bundle: Path = ASSETS):
    # PDFium is process-global. A compiler process per case isolates parallel workers.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "openin_any": "p", "openout_any": "p"}
    completed = subprocess.run(
        [sys.executable, str(bundle / v13_check.TOOL), "build",
         "--rules-dir", str(bundle), "--case-dir", str(case)],
        capture_output=True, encoding="utf-8", errors="replace", timeout=240, env=env,
    )
    state_path = case / "build-status.json"
    if not state_path.is_file():
        raise BuildFailure(f"v13 checker could not finish: {completed.stderr[-1800:]}", [], case_dir=case, technical=True)
    try:
        return completed.returncode, json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BuildFailure("v13 checker did not write a complete status report.", [], case_dir=case, technical=True) from exc


def build_case(work: Path, source: Path, source_root: Path | None, draft: dict) -> Path:
    """Return a source-bound case with engineering-accepted PDF, TeX and preamble."""
    blocks = []
    tex = render_tex(draft, block_map=blocks)
    case = None
    try:
        release = release_for_draft(draft)
        bundle = profile_bundle(ASSETS, Path(work), release)
        case = v13_check.allocate(bundle, Path(source), Path(work), source_root=source_root)
        (case / "introduction.tex").write_text(tex, encoding="utf-8")
        returncode, status = _compile_case(case, bundle)
        if returncode or not status.get("engineering_accepted"):
            message = _diagnosis(case, status)
            log_file = case / "introduction.log"
            log = log_file.read_text(encoding="utf-8", errors="replace") if log_file.is_file() else ""
            technical = returncode == 130 or (returncode == 2 and (
                not status.get("command") or status.get("error_type") in {
                    "OSError", "FileNotFoundError", "PermissionError", "TimeoutExpired"
                } or bool(re.search(r"LaTeX Error: File .+ not found\.|I can't find the format file", log))
            ))
            raise BuildFailure(message, _build_targets(blocks, [message], log, release=release),
                               case_dir=case, technical=technical)
        return case
    except (OSError, subprocess.TimeoutExpired, v13_check.ValidationError) as exc:
        raise BuildFailure(f"v13 build unavailable: {exc}", [], case_dir=case, technical=True) from exc


def recover_draft_case(work: Path, source: Path, source_root: Path | None, draft: dict,
                       *, destination: Path | None = None) -> Path | None:
    """Recompile a uniquely matching original TeX/preamble pair without altering its verdict or inputs."""
    tex = render_tex(draft)
    legacy_tex = None
    work, source = Path(work).resolve(), Path(source).resolve()
    snapshot = v13_check.source_snapshot(source, source_root)
    source_hash = v13_check.digest(source)
    bundle = profile_bundle(ASSETS, work, release_for_draft(draft))
    preamble_hash = v13_check.digest(bundle / "preamble-v13.tex")
    matches = []
    for manifest in work.glob("*/input-manifest.json"):
        case = manifest.parent
        paths = [manifest, case / "build-status.json", case / "introduction.tex", case / "preamble-v13.tex"]
        if case.is_symlink() or any(path.is_symlink() or not path.is_file() for path in paths):
            continue
        try:
            binding, old = (json.loads(path.read_text("utf8")) for path in paths[:2])
        except (ValueError, OSError):
            continue
        if (binding.get("source_sha256") != source_hash or binding.get("snapshot", {}).get("sha256") != snapshot["sha256"]
                or old.get("input_binding", {}).get("manifest_sha256") != v13_check.digest(manifest)
                or old.get("tex_sha256") != v13_check.digest(paths[2])
                or v13_check.digest(paths[3]) != preamble_hash):
            continue
        historical_text = paths[2].read_text("utf8")
        if historical_text != tex:
            if legacy_tex is None:
                legacy_tex = [render_tex(draft, _legacy_citations=mode) for mode in (True, "copy-inline")]
            if historical_text not in legacy_tex:
                continue
        matches.append((case, old))
    if len(matches) != 1:
        return None
    original, old = matches[0]
    provenance = {"kind": "local_recompile", "original_case": original.name, "original_run_id": old["run_id"],
                  "original_tex_sha256": old["tex_sha256"], "original_preamble_sha256": preamble_hash,
                  "original_pdf_sha256": old.get("compiled_pdf_sha256"), "source_sha256": source_hash,
                  "original_checker_sha256": old.get("checker_sha256"),
                  "draft_sha256": hashlib.sha256(json.dumps(draft, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf8")).hexdigest()}

    def has_pdf(case, status):
        art = status.get("diagnostic_artifacts") or {}
        rid = str(status.get("run_id", ""))
        if not re.fullmatch(r"[a-f0-9]{32}", rid):
            return False
        loc = case
        files = {"introduction.tex": old["tex_sha256"], "preamble-v13.tex": preamble_hash,
                 "introduction.pdf": status.get("compiled_pdf_sha256") or status.get("pdf_sha256")}
        expected_pages = status.get("pdf_pages")
        if art:
            if art.get("directory") != f"draft-artifacts/{rid}" or art.get("run_id") != rid:
                return False
            loc, files, expected_pages = case / art["directory"], art.get("files", {}), art.get("pages")
            receipt = loc / "receipt.json"
            if not loc.resolve().is_relative_to(case.resolve()) or receipt.is_symlink():
                return False
            try:
                archived = json.loads(receipt.read_text("utf8"))
            except (ValueError, OSError):
                return False
            if archived != art:
                return False
        if (files.get("introduction.tex") != old["tex_sha256"] or files.get("preamble-v13.tex") != preamble_hash
                or not files.get("introduction.pdf")):
            return False
        for name in ("introduction.tex", "preamble-v13.tex", "introduction.pdf"):
            if (loc / name).is_symlink() or not (loc / name).is_file() or v13_check.digest(loc / name) != files[name]:
                return False
        pdf = loc / "introduction.pdf"
        try:
            pages = page_count(pdf)
            if not pages or (expected_pages is not None and pages != expected_pages):
                return False
            for page in range(1, pages + 1):
                render_page(pdf, page, scale=0.25)
            return True
        except (OSError, RuntimeError, ValueError):
            return False

    if has_pdf(original, old):
        return original
    target = Path(destination or work).resolve() / "recovered-drafts"
    target.mkdir(parents=True, exist_ok=True)
    with FileLock(target / ".recovery.lock"):
        for record in target.glob("*/draft-recovery.json"):
            status_path = record.parent / "build-status.json"
            if not status_path.is_file() or json.loads(record.read_text("utf8")) != provenance:
                continue
            if has_pdf(record.parent, json.loads(status_path.read_text("utf8"))):
                return record.parent
        case = v13_check.allocate(bundle, source, target, source_root=source_root)
        for name in ("introduction.tex", "preamble-v13.tex"):
            shutil.copyfile(original / name, case / name)
        new_binding = json.loads((case / "input-manifest.json").read_text("utf8"))
        if (v13_check.digest(case / "introduction.tex") != old["tex_sha256"]
                or v13_check.digest(case / "preamble-v13.tex") != preamble_hash
                or new_binding["snapshot"]["sha256"] != snapshot["sha256"]):
            return None
        v13_check.atomic_json(case / "draft-recovery.json", provenance)
        _, status = _compile_case(case, bundle)
        return case if has_pdf(case, status) else None

