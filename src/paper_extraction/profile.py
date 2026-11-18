"""Generate an English v13 introduction from bounded, structured model fields."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import os
import re
import subprocess
import sys
from urllib.parse import quote

from .profile_assets.tools import v13_check

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
_CONTROL_SYMBOLS = frozenset(r"""!,:; {}#$%&_()/[]\"'\^~=.|\->""")
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


def _bound_locator(ids, records):
    by_file = {}
    for loc in ids:
        record = records.get(loc, {})
        page = record.get("page")
        if type(page) is int and page >= 1:
            by_file.setdefault(None, []).append((page, page))
            continue
        filename, start, end = record.get("file"), record.get("line"), record.get("end_line")
        if not filename or type(start) is not int or type(end) is not int or not (1 <= start <= end):
            return ""
        by_file.setdefault(filename, []).append((start, end))
    parts = []
    for filename, ranges in by_file.items():
        if filename is None:
            label = "PDF physical p." if len(set(ranges)) == 1 else "PDF physical pp."
        else:
            label = _file_locator(filename) + " L"
        parts.append(label + "~" + _line_ranges(ranges))
    return "; ".join(parts)


def bind_formal_citations(draft: dict, records: dict[str, dict]) -> dict:
    """Fill absent formal citations from this paper's registered source spans."""
    result = deepcopy(draft)
    if not any(ref.get("key") == "source" for ref in result.get("references", [])):
        return result
    links = {row["block_id"]: row["evidence_ids"] for row in result.get("evidence_links", [])}

    def bind(block, text, environment):
        if not text or re.search(r"\\cite\b", text):
            return text
        locator = _bound_locator(links.get(block, []), records)
        if not locator:
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


def render_tex(draft: dict, *, block_map: list | None = None) -> str:
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
    header = (ASSETS / "template-v13.tex").read_text(encoding="utf-8").split(r"\begin{document}", 1)[0]
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
            # The matching template prints the locator in the optional theorem heading.
            locator = list(re.finditer(r"\\cite\b", body))[-1]
            citation, end = _citation(body, locator.end(), field, keys)
            caption = (caption + " " if caption else "") + citation
            if not body[end:].strip(" .\t\r\n"):
                tail = body[end:].strip()
                body = body[:locator.start()].rstrip(" ~\r\n\t")
                if tail == "." and not body.endswith("."):
                    body += "."
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
    for i, value in enumerate(_list(draft.get("history"), "history", 1, 3)):
        block("historical", value, "history", f"history.{i}")
    for number, group in enumerate(_list(draft.get("groups"), "groups", 1, 4), 1):
        if not isinstance(group, dict):
            raise ValueError(f"groups[{number}] must be an object.")
        parts.append(r"\begin{resultgroup}")
        group_id = f"groups.{number-1}"
        for j, value in enumerate(_list(group.get("definitions", []), "group definitions", 0, 2)):
            count += 1
            block("definition", value, f"group {number} definitions", f"{group_id}.definitions.{j}")
        block("theorem", group.get("theorem"), f"group {number} theorem", f"{group_id}.theorem")
        if group.get("paragraph"):
            block("introparagraph", group["paragraph"], f"group {number} paragraph", f"{group_id}.paragraph")
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
    def __init__(self, message, repair_targets):
        super().__init__(message)
        self.repair_targets = repair_targets


def _build_targets(blocks, issues, log=""):
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
                               r"(word_limit:\d+>\d+|display_budget|missing_source_locator)", detail)
        box = re.fullmatch(r"Overfull \\[hv]box .*?at lines? (\d+)(?:--(\d+))?\.?", detail)
        measured = re.search(r"Package intro-v13 Error: Block (\d+) \(([PDHT])\) exceeds (\d+)U", detail)
        if indexed:
            same_kind = [row for row in blocks if row["environment"] == indexed[1]]
            i = int(indexed[2]) - 1
            if 0 <= i < len(same_kind):
                matched = [same_kind[i]]
                if indexed[3].startswith("word_limit:"):
                    actual, limit = indexed[3].split(":")[1].split(">")
                    detail += f". Shorten this block from {actual} prose words to at most {limit}, retaining necessary hypotheses and conclusions."
                elif indexed[3] == "display_budget":
                    cap = v13_check.BLOCK_LIMITS[indexed[1]]
                    detail += f". Use at most {cap['displays']} display groups and {cap['rows']} display source rows."
        elif box:
            first, last = int(box[1]), int(box[2] or box[1])
            matched = [row for row in blocks if row["first"] <= first <= last <= row["last"]]
            if r"\hbox" in detail:
                detail += ". Fit the affected prose or formula within the text width using the v13 display budget."
        elif measured:
            i = int(measured[1]) - 1
            if 0 <= i < len(blocks) and v13_check.BLOCK_LIMITS[blocks[i]["environment"]]["kind"] == measured[2]:
                matched = [blocks[i]]
                reading = re.search(rf"(?m)^INTRO-BLOCK\|{i+1}\|{measured[2]}\|(\d+)\|(\d+)\|", log)
                if reading and int(reading[2]):
                    actual = int(reading[1]) * int(measured[3]) / int(reading[2])
                    detail += f" (rendered height {actual:.3f}U, limit {measured[3]}U)"
        elif detail in {"rendered_block_height_exceeded", "resultgroup_measurement_failed"} or "Package intro-v13 Error: Result group exceeds 22U" in detail:
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
                    units = 22 if group else v13_check.BLOCK_LIMITS[row["environment"]]["units"]
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
    return _build_targets(blocks, issues)


def _diagnosis(case: Path, status: dict) -> str:
    details = [status.get("error", "v13 verification failed")]
    details.extend(status.get("format_contract", {}).get("errors", []))
    details.extend(status.get("blocking_diagnostics", []))
    log = case / "introduction.log"
    if log.is_file() and not status.get("compiled"):
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("!") or re.search(r"\.tex:\d+:|Package intro-v13 Error", line):
                details.append(line)
    return " | ".join(map(str, details))


def build_case(work: Path, source: Path, source_root: Path | None, draft: dict) -> Path:
    """Return a source-bound case with engineering-accepted PDF, TeX and preamble."""
    blocks = []
    tex = render_tex(draft, block_map=blocks)
    try:
        case = v13_check.allocate(ASSETS, Path(source), Path(work), source_root=source_root)
        (case / "introduction.tex").write_text(tex, encoding="utf-8")
        # PDFium is process-global. A compiler process per case isolates parallel workers.
        env = os.environ.copy()
        env.update(PYTHONIOENCODING="utf-8", openin_any="p", openout_any="p")
        completed = subprocess.run(
            [sys.executable, str(ASSETS / "tools" / "v13_check.py"), "build",
             "--rules-dir", str(ASSETS), "--case-dir", str(case)],
            capture_output=True, encoding="utf-8", errors="replace", timeout=240, env=env,
        )
        state_path = case / "build-status.json"
        if not state_path.is_file():
            raise ValueError(f"v13 checker could not finish: {completed.stderr[-1800:]}")
        status = json.loads(state_path.read_text(encoding="utf-8"))
        if completed.returncode or not status.get("engineering_accepted"):
            message = _diagnosis(case, status)
            log_file = case / "introduction.log"
            log = log_file.read_text(encoding="utf-8", errors="replace") if log_file.is_file() else ""
            raise BuildFailure(message, _build_targets(blocks, [message], log))
        return case
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"v13 build unavailable: {exc}") from exc

