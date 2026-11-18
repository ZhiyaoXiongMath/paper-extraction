"""Generate an English v13 introduction from bounded, structured model fields."""
from __future__ import annotations

from pathlib import Path
import json
import os
import re
import subprocess
import sys

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
rm texttt textsc textsl textsf textup textsuperscript textsubscript mathbin mathrel mathopen mathclose
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


def _citation(text: str, after_command: int, field: str, keys: set[str]) -> tuple[str, int]:
    cursor = after_command
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    command = r"\cite"
    if text[cursor:cursor+1] == "[":
        locator, cursor = v13_check.balanced(text, cursor, "[", "]")
        clean = _fragment(locator, field, keys, allow_blocks=False) if locator.strip() else ""
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


def render_tex(draft: dict) -> str:
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

    def block(env, val, field):
        body = _fragment(val, field, keys, formal=env != "introparagraph")
        opening = rf"\begin{{{env}}}"
        if env != "introparagraph":
            # The matching template prints the locator in the optional theorem heading.
            locator = list(re.finditer(r"\\cite\b", body))[-1]
            citation, end = _citation(body, locator.end(), field, keys)
            opening += "[{" + citation + "}]"
            if not body[end:].strip(" .\t\r\n"):
                tail = body[end:].strip()
                body = body[:locator.start()].rstrip(" ~\r\n\t")
                if tail == "." and not body.endswith("."):
                    body += "."
        parts.extend([opening, body, rf"\end{{{env}}}"])

    block("introparagraph", draft.get("paragraph"), "paragraph")
    definitions = _list(draft.get("shared_definitions", []), "shared_definitions", 0, 2)
    count = len(definitions)
    for value in definitions:
        block("definition", value, "shared_definitions")
    for value in _list(draft.get("history"), "history", 1, 3):
        block("historical", value, "history")
    for number, group in enumerate(_list(draft.get("groups"), "groups", 1, 4), 1):
        if not isinstance(group, dict):
            raise ValueError(f"groups[{number}] must be an object.")
        parts.append(r"\begin{resultgroup}")
        for value in _list(group.get("definitions", []), "group definitions", 0, 2):
            count += 1
            block("definition", value, f"group {number} definitions")
        block("theorem", group.get("theorem"), f"group {number} theorem")
        if group.get("paragraph"):
            block("introparagraph", group["paragraph"], f"group {number} paragraph")
        parts.append(r"\end{resultgroup}")
    if count > 6:
        raise ValueError("The v13 profile allows a maximum of six definitions.")
    parts.extend([r"\clearpage", rf"\begin{{thebibliography}}{{{len(refs)}}}"])
    for ref in refs:
        content = _fragment(ref.get("text"), "bibliography", keys, allow_blocks=False)
        parts.append(rf"\bibitem{{{ref['key']}}} {content}")
    parts.extend([r"\end{thebibliography}", r"\end{document}"])
    return "\n".join(parts) + "\n"


def _diagnosis(case: Path, status: dict) -> str:
    details = [status.get("error", "v13 verification failed")]
    details.extend(status.get("format_contract", {}).get("errors", []))
    details.extend(status.get("blocking_diagnostics", [])[:4])
    log = case / "introduction.log"
    if log.is_file() and not status.get("compiled"):
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("!") or re.search(r"\.tex:\d+:|Package intro-v13 Error", line):
                details.append(line)
                if len(details) >= 5:
                    break
    return " | ".join(map(str, details))


def build_case(work: Path, source: Path, source_root: Path | None, draft: dict) -> Path:
    """Return a source-bound case with engineering-accepted PDF, TeX and preamble."""
    tex = render_tex(draft)
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
            raise ValueError(_diagnosis(case, status))
        return case
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"v13 build unavailable: {exc}") from exc

