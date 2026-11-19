"""Publish a readable source-bound TeX/PDF pair without another model call."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import textwrap

from PIL import Image, ImageDraw, ImageFont

from .citation_layout import reading_blocks
from .pdfops import page_count, render_page
from .profile import ASSETS, BuildFailure, build_case, draft_layout_targets, profile_bundle, release_for_draft, render_tex
from .profile_assets.tools import v13_check
from .profile_variants import checker_for_bundle, checker_for_release
from .sources import SourceIndex
from .writer_patch import _address

_FILES = ("introduction.tex", "introduction.pdf", "preamble-v13.tex")
_OPEN_DOCUMENT = r"\begin{document}"
_COMPILATION_ONLY = (
    r"\renewcommand{\IntroBudgetError}[1]{\PackageWarning{intro-v13}{#1}}" "\n"
)
_LITERAL_PREAMBLE = r"""\usepackage[T1]{fontenc}
\setlength{\textwidth}{150mm}
\setlength{\oddsidemargin}{5mm}
\setlength{\parskip}{4pt}
\setlength{\parindent}{0pt}
\emergencystretch=2em
"""
_ESCAPES = dict(zip(
    "\\{}$%#&_~^<>|",
    (r"\textbackslash{}", r"\{", r"\}", r"\$", r"\%", r"\#", r"\&", r"\_",
     r"\textasciitilde{}", r"\textasciicircum{}", r"\textless{}", r"\textgreater{}", r"\textbar{}"),
    strict=True,
))
_UNICODE = str.maketrans({"–": "--", "—": "---", "‘": "'", "’": "'", "“": '"', "”": '"',
                          "≤": "<=", "≥": ">=", "−": "-", "\u00a0": " ", "\u00ad": ""})
_LOCATORS = re.compile(r"\\cite\s*\[[^\]]+\]\s*\{[A-Za-z0-9:.,_ -]+\}")
_DELIVERY_ERRORS = (ValueError, OSError, RuntimeError, subprocess.TimeoutExpired)
_ALIGNED_AT = re.compile(
    r"\\\[\s*\\begin\{alignedat\}\s*\{[12]\}(?P<body>.*?)\\end\{alignedat\}\s*\\\]", re.S,
)
_INDEPENDENT_GATHERED = re.compile(
    r"(?P<lead>\b(?:following|independent)\s+(?:data|conditions|requirements)\s*:\s*)"
    r"\\\[\s*\\begin\{gathered\}(?P<body>.*?)\\end\{gathered\}\s*\\\]", re.S | re.I,
)
_ALIGNED_DISPLAY = re.compile(
    r"\\\[\s*\\begin\{aligned\}(?P<body>.*?)\\end\{aligned\}\s*\\\]", re.S,
)
_LAYOUT_SENSITIVE = re.compile(r"%|\\(?:begin|end|intertext|tag|notag|label|nonumber|displaybreak)\b|\\\\\s*\[")
_ALIGNMENT_ANCHOR = re.compile(r"(?<!\\)&")
_RELATION_START = re.compile(r"(?:\s*\{\})?\s*(?:[=<>]|\\(?:to|longrightarrow|rightarrow|leq?|geq?|equiv|sim)\b)")
_RELATION_ROW = re.compile(
    r"\s*&?(?P<left>[^&\r\n]+?)\s*&{1,2}\s*(?P<right>(?:\{\})?(?:[=<>]|\\(?:to|longrightarrow|rightarrow|leq?|geq?|equiv|sim)\b)[^&\r\n]+)\s*",
)
_MEASURED_BLOCK = re.compile(r"(?m)^INTRO-BLOCK\|(\d+)\|([PDHT])\|(\d+)\|(\d+)\|\d+\s*$")
_INTERRUPTED_MEASUREMENT = re.compile(
    r"Package intro-v13 Error: (?:Block \d+ \([PDHT]\)|Result group|Total connecting prose|Shared definitions) exceeds \d+U"
)


@dataclass(frozen=True)
class DeliveryCase:
    path: Path
    kind: str
    diagnostic: str = ""
    compiler_verified: bool = True
    draft: dict | None = None
    layout_check: str = "not_checked"
    layout_issues: tuple[str, ...] = ()


class _CompilerFailure(BuildFailure):
    def __init__(self, message, case, line):
        super().__init__(message, [], case_dir=case)
        self.line = line


def normalize_alignment(draft: dict, *, improve_layout=False) -> dict:
    """Normalize only displays whose mathematical rows and expressions can be retained."""
    def replace(match):
        body = match["body"].strip()
        if re.search(r"%|\\(?:begin|end|intertext|tag|notag|label)\b|\\\\\s*\[", body):
            return match[0]
        rows = [row for row in body.split(r"\\") if row.strip()]
        relations = [_RELATION_ROW.fullmatch(row) for row in rows]
        if not 2 <= len(rows) <= 4 or not all(relations):
            return match[0]
        lines = [f'{row["left"].strip()} &{row["right"].strip()}' for row in relations]
        return r"\begin{relations}" + "\n" + (" " + r"\\" + "\n").join(lines) + "\n" + r"\end{relations}"

    def independent_rows(match):
        body = match["body"].strip()
        if _LAYOUT_SENSITIVE.search(body) or _ALIGNMENT_ANCHOR.search(body):
            return match[0]
        rows = [row.strip() for row in body.split(r"\\")]
        if not 2 <= len(rows) <= 4 or not all(rows):
            return match[0]
        return (match["lead"] + r"\begin{conditions}" + "\n" +
                (" " + r"\\" + "\n").join("& " + row for row in rows) + "\n" + r"\end{conditions}")

    def single_row(match):
        body = match["body"].strip()
        if not body or r"\\" in body or _LAYOUT_SENSITIVE.search(body):
            return match[0]
        anchors = list(_ALIGNMENT_ANCHOR.finditer(body))
        if anchors:
            if len(anchors) != 1:
                return match[0]
            offset = anchors[0].start()
            left, right = body[:offset], body[offset + 1:]
            braces = re.findall(r"(?<!\\)[{}]", left)
            if (braces.count("{") != braces.count("}") or
                    (left.strip() and not _RELATION_START.match(right)) or not right.strip()):
                return match[0]
            body = (left.rstrip() + " " + right.lstrip()).strip()
        return r"\[" + "\n" + body + "\n" + r"\]"

    def rewrite(value):
        if isinstance(value, str):
            if improve_layout:
                value = _INDEPENDENT_GATHERED.sub(independent_rows, value)
                value = _ALIGNED_DISPLAY.sub(single_row, value)
            return _ALIGNED_AT.sub(replace, value)
        if isinstance(value, list):
            return [rewrite(row) for row in value]
        return value

    result = {**draft}
    for key in ("paragraph", "history", "shared_definitions"):
        if key in draft:
            result[key] = rewrite(draft[key])
    if "groups" in draft:
        result["groups"] = [
            {key: rewrite(value) if key in {"theorem", "paragraph", "definitions"} else value
             for key, value in group.items()}
            for group in draft["groups"]
        ]
    return result


def _compile(case: Path, *, passes: int = 2) -> list[int]:
    executable = shutil.which("pdflatex")
    if executable is None:
        raise BuildFailure("未找到 pdfLaTeX。", [], case_dir=case, technical=True)
    command = [executable, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
               "-no-shell-escape", "introduction.tex"]
    environment = {**os.environ, "openin_any": "p", "openout_any": "p"}
    inputs = (case / "introduction.tex", case / "preamble-v13.tex")
    before = [v13_check.digest(path) for path in inputs]
    returncodes = []
    for i in range(1, passes + 1):
        completed = subprocess.run(command, cwd=case, env=environment, capture_output=True,
                                   text=True, encoding="utf-8", errors="replace", timeout=120)
        output = completed.stdout + completed.stderr
        (case / f"build-pass-{i}.txt").write_text(output, encoding="utf-8")
        returncodes.append(completed.returncode)
        if completed.returncode:
            match = re.search(r"(?m)^(?:! |[^\n]*introduction\.tex:(\d+): )([^\n]+)", output)
            line = int(match[1]) if match and match[1] else None
            detail = f" {match[2].strip()[:300]}" if match else ""
            raise _CompilerFailure(f"pdfLaTeX 第 {i} 遍退出码 {completed.returncode}。{detail}", case, line)
    if [v13_check.digest(path) for path in inputs] != before:
        raise BuildFailure("编译期间 TeX 或导言区发生变化。", [], case_dir=case, technical=True)
    return returncodes


def _readable(case: Path) -> int:
    pdf = case / "introduction.pdf"
    pages = page_count(pdf)
    if not pages:
        raise ValueError("PDF 没有可打开的页面。")
    for page in range(1, pages + 1):
        render_page(pdf, page, scale=0.25)
    return pages


def _prior(work: Path, case: Path | str | None, source: Path, snapshot_hash: str) -> Path | None:
    if not case:
        return None
    case = Path(case)
    try:
        if case.is_symlink() or case.resolve().parent != work:
            return None
        manifest = case / "input-manifest.json"
        if manifest.is_symlink():
            return None
        binding = json.loads(manifest.read_text("utf-8"))
        if (binding["snapshot"]["sha256"] != snapshot_hash or Path(binding["source"]).resolve() != source
                or binding["source_sha256"] != v13_check.digest(source)):
            return None
        receipt = case / "build-status.json"
        if receipt.is_symlink():
            return None
        status = json.loads(receipt.read_text("utf-8"))
        if (not status.get("compiled") or status.get("returncodes") not in ([0, 0], [0, 0, 0])
                or status["input_binding"]["manifest_sha256"] != v13_check.digest(manifest)):
            return None
        expected = {"introduction.tex": status["tex_sha256"],
                    "preamble-v13.tex": binding["rules_sha256"]["preamble-v13.tex"],
                    "introduction.pdf": status["compiled_pdf_sha256"]}
        location = case
        if not (location / "introduction.pdf").is_file():
            diagnostic = status.get("diagnostic_artifacts") or {}
            run_id = status.get("run_id", "")
            rel = f"draft-artifacts/{run_id}"
            if (not re.fullmatch(r"[a-f0-9]{32}", run_id) or diagnostic.get("directory") != rel
                    or diagnostic.get("run_id") != run_id or diagnostic.get("files") != expected
                    or diagnostic.get("input_manifest_sha256") != v13_check.digest(manifest)):
                return None
            location = case / rel
            receipt = location / "receipt.json"
            if (location.is_symlink() or not location.resolve().is_relative_to(case.resolve())
                    or receipt.is_symlink() or json.loads(receipt.read_text("utf-8")) != diagnostic):
                return None
        if not all((path := location / name).is_file() and not path.is_symlink()
                   and v13_check.digest(path) == expected[name] for name in _FILES):
            return None
        _readable(location)
        return location
    except (OSError, ValueError, KeyError, RuntimeError):
        return None


def _patch_from_prior(draft, blocks, previous_draft, prior, compiler_line):
    """Restore one demonstrably identical source-bound block after a localized TeX error."""
    # The final renderer inserts one line in the preamble before every mapped block.
    targets = [row for row in blocks if row["environment"] != "introparagraph"
               and row["first"] <= compiler_line - 1 <= row["last"]]
    if len(targets) != 1 or not previous_draft or previous_draft.get("references") != draft.get("references"):
        return None
    old_blocks = []
    if render_tex(previous_draft, block_map=old_blocks) != (prior / "introduction.tex").read_text("utf-8"):
        return None
    fields = [row["draft_field"] for row in blocks]
    if fields != [row["draft_field"] for row in old_blocks]:
        return None
    target = targets[0]
    field = target["draft_field"]
    found = old_blocks[fields.index(field)]

    def origin(payload, row):
        path = row["draft_field"]
        keys = set(next((link["evidence_ids"] for link in payload.get("evidence_links", [])
                         if link["block_id"] == path), ()))
        if "formal_sources" in payload:
            attribution = payload["formal_sources"].get(path)
            if not attribution:
                return keys, None
            owners = payload.get("core_bindings", {})
            group = re.match(r"(groups\.\d+)\.definitions\.\d+$", path)
            owner = owners.get(group[1] + ".theorem" if group else path)
            # A core ID, its evidence and its heading attribution travel together
            # when local TeX repair restores the earlier approved wording.
            return keys, (owner, attribution) if attribution.get("source") else None
        return keys, _LOCATORS.findall(row["current_text"])

    identity = origin(draft, target)
    if (not all(identity) or identity != origin(previous_draft, found)
            or sum(origin(draft, row) == identity for row in blocks) != 1
            or sum(origin(previous_draft, row) == identity for row in old_blocks) != 1
            or target["current_text"] == found["current_text"]):
        return None
    repaired = deepcopy(draft)
    parent, key = _address(repaired, field)
    parent[key] = found["current_text"]
    return repaired


def _write_formal(case: Path, tex: str) -> None:
    # The companion stays frozen. Budget diagnostics are warnings, while actual
    # TeX failures still abort compilation and trigger deterministic recovery.
    tex = tex.replace(_OPEN_DOCUMENT, _COMPILATION_ONLY + _OPEN_DOCUMENT, 1)
    (case / "introduction.tex").write_text(tex, encoding="utf-8")


def _measure_layout(work: Path, draft: dict, case: Path | None) -> list[dict]:
    if case is None or not (case / "introduction.log").is_file():
        return []
    blocks = []
    tex = render_tex(draft, block_map=blocks)
    log = (case / "introduction.log").read_text(encoding="utf-8", errors="replace")
    measured = list(_MEASURED_BLOCK.finditer(log))
    if len(measured) < len(blocks) and _INTERRUPTED_MEASUREMENT.search(log):
        # A hard budget error may stop TeX before it measures later blocks.
        # The temporary PDF is diagnostic only and never becomes an artifact.
        with tempfile.TemporaryDirectory(prefix="layout-", dir=work) as scratch:
            probe = Path(scratch)
            shutil.copyfile(case / "preamble-v13.tex", probe / "preamble-v13.tex")
            _write_formal(probe, tex)
            try:
                _compile(probe, passes=1)
            except (BuildFailure, subprocess.TimeoutExpired):
                pass
            diagnostic_log = probe / "introduction.log"
            if diagnostic_log.is_file():
                extended = list(_MEASURED_BLOCK.finditer(
                    diagnostic_log.read_text(encoding="utf-8", errors="replace")
                ))
                if len(extended) > len(measured):
                    measured = extended
    limits = checker_for_release(release_for_draft(draft)).BLOCK_LIMITS
    rows = []
    for number, kind, height, limit in (row.groups() for row in measured):
        offset = int(number) - 1
        if not 0 <= offset < len(blocks) or not int(limit):
            continue
        block = blocks[offset]
        cap = limits[block["environment"]]
        if cap["kind"] == kind:
            rows.append({"draft_field": block["draft_field"], "current_text": block["current_text"],
                         "height_units": round(int(height) * cap["units"] / int(limit), 3),
                         "limit_units": cap["units"], "over_limit": int(height) > int(limit)})
    return rows


def local_layout_diagnostics(work: Path, draft: dict, case: Path | None, strict_targets: list[dict],
                             *, rendered: bool) -> tuple[list[dict], list[dict], list[str]]:
    """Collect independent formatting and height issues from the same saved manuscript."""
    static = draft_layout_targets(draft)
    measured = _measure_layout(work, draft, case)
    extra = list(static)
    for row in measured:
        if row["over_limit"]:
            height, limit = row["height_units"], row["limit_units"]
            extra.append({"draft_field": row["draft_field"], "current_text": row["current_text"],
                          "diagnostics": [f"Shorten this block (rendered height {height:.3f}U, limit {limit}U)."]})
    if not strict_targets and rendered:
        # A document-level failure still needs the original full-draft path.
        # Add the known local facts without claiming that they explain it all.
        return [], measured, [f'{row["draft_field"]}: {detail}' for row in extra for detail in row["diagnostics"]]
    targets = {}
    for row in [*strict_targets, *extra]:
        field = row["draft_field"]
        target = targets.setdefault(field, {**row, "diagnostics": []})
        for detail in row["diagnostics"]:
            if any(re.search(r"rendered height [\d.]+U, limit \d+U", text) for text in target["diagnostics"]) \
                    and detail.startswith("Shorten this block (rendered height"):
                continue
            if detail not in target["diagnostics"]:
                target["diagnostics"].append(detail)
    return list(targets.values()), measured, []


def _receipt(case: Path, bundle: Path, kind: str, returncodes: list[int], diagnostic: str,
             *, draft: dict | None = None, layout_check: str = "not_checked",
             layout_issues: tuple[str, ...] = ()) -> DeliveryCase:
    pages = _readable(case)
    checker = checker_for_bundle(bundle)
    binding = checker.check_binding(checker.verify(bundle), case)
    compiler_verified = returncodes in ([0, 0], [0, 0, 0])
    v13_check.atomic_json(case / "delivery.json", {
        "kind": kind,
        "diagnostic": diagnostic,
        "compiler_verified": compiler_verified,
        "compiler_returncodes": returncodes,
        "pdf_pages": pages,
        "files": {name: v13_check.digest(case / name) for name in _FILES},
        "input_binding": binding,
        "layout_check": layout_check,
        "layout_issues": list(layout_issues),
    })
    return DeliveryCase(case, kind, diagnostic, compiler_verified, draft, layout_check, layout_issues)


def _layout_findings(contract: dict, other: list[str]) -> tuple[str, ...]:
    messages = []
    labels = {"introparagraph": "引言段", "historical": "历史定理", "definition": "定义", "theorem": "定理"}
    for block in contract.get("blocks", []):
        height, limit = block.get("height_units"), block.get("height_limit_units")
        if height is not None and limit is not None and height > limit:
            name = labels.get(block["environment"], block["environment"])
            messages.append(f'{name} {block["number"]} 实测 {height:g}U，上限 {limit:g}U。')
    generic = {"rendered_block_height_exceeded"} if messages else set()
    for group in contract.get("groups", []):
        height, limit = group.get("height_units"), group.get("height_limit_units")
        if height is not None and limit is not None and height > limit:
            messages.append(f'第 {group["group"]} 组实测 {height:g}U，上限 {limit:g}U。')
            generic.add("resultgroup_measurement_failed")
    errors = contract.get("errors", [])
    return tuple(dict.fromkeys([*messages, *(error for error in errors if error not in generic), *other]))[:12]


def _measure_delivery(case: Path, checker) -> tuple[str, ...]:
    """Measure the exact fallback files, even when a strict run stopped at its first warning."""
    tex = (case / "introduction.tex").read_text(encoding="utf-8")
    log = (case / "introduction.log").read_text(encoding="utf-8", errors="replace")
    aux = (case / "introduction.aux").read_text(encoding="utf-8", errors="replace")
    pdf = case / "introduction.pdf"
    blocking, _ = checker.classify_diagnostics(log)
    findings = list(blocking)
    pages = checker.inspect_pdf(pdf, aux, tex)
    if pages["body_page_limit"] is False and (count := pages["main_text_pages"]) is not None:
        findings.append(f'正文 {count} 页，上限 {pages["max_body_pages"]} 页。')
    if not pages["references_separate"]:
        findings.append("参考文献未独立起页或缺少可识别的参考文献页。")
    return _layout_findings(checker.format_contract(tex, log, pdf), findings)


def _saved_layout(case: Path, manifest_case: Path) -> tuple[str, tuple[str, ...]]:
    try:
        status = json.loads((manifest_case / "build-status.json").read_text(encoding="utf-8"))
        if (status.get("tex_sha256") != v13_check.digest(case / "introduction.tex")
                or status.get("compiled_pdf_sha256") != v13_check.digest(case / "introduction.pdf")):
            return "not_checked", ()
        if status.get("engineering_accepted"):
            return "passed", ()
        issues = _layout_findings(status.get("format_contract", {}), status.get("blocking_diagnostics", []))
        return "failed", issues or (status.get("error") or "未通过严格排版检查。",)
    except (OSError, ValueError, TypeError):
        return "not_checked", ()


def _strict_formal(work, source, source_root, draft, bundle, snapshot_hash) -> DeliveryCase:
    checker = checker_for_bundle(bundle)
    try:
        case = build_case(work, source, source_root, draft)
    except BuildFailure as exc:
        if saved := _prior(work, exc.case_dir, source, snapshot_hash):
            layout_check, issues = _saved_layout(saved, exc.case_dir)
            case = checker.allocate(bundle, source, work, "delivery-checked", source_root)
            for name in ("introduction.tex", "introduction.pdf"):
                shutil.copyfile(saved / name, case / name)
            status = json.loads((exc.case_dir / "build-status.json").read_text(encoding="utf-8"))
            return _receipt(case, bundle, "final", status["returncodes"], "", draft=draft, layout_check=layout_check,
                            layout_issues=issues)
        # The strict preamble can stop during an over-height block before a PDF
        # exists. Build a fresh, readable pair with the same content and then
        # measure that exact output. A failed strict run never earns a pass.
        case = checker.allocate(bundle, source, work, "delivery", source_root)
        _write_formal(case, render_tex(draft, allow_empty_history=True))
        passes = _compile(case)
        try:
            issues = _measure_delivery(case, checker)
        except (*_DELIVERY_ERRORS, checker.ValidationError):
            issues = ()
        layout_check = "failed" if issues or not exc.technical else "not_checked"
        return _receipt(case, bundle, "final", passes, "", draft=draft, layout_check=layout_check,
                        layout_issues=issues or (str(exc)[:600],))
    status = json.loads((case / "build-status.json").read_text(encoding="utf-8"))
    return _receipt(case, bundle, "final", status["returncodes"], "", draft=draft, layout_check="passed")


def _literal(value: str, reference_numbers: dict | None = None) -> str:
    # Keep uncertain symbols identifiable instead of replacing a mathematical expression.
    def citation(match):
        keys = match[2]
        if reference_numbers is not None:
            keys = ", ".join(reference_numbers.get(key.strip(), key.strip()) for key in keys.split(","))
        locator = (match[1] or "").replace("~", " ") if reference_numbers is not None else match[1]
        return "[" + ", ".join(part for part in (keys, locator) if part) + "]"

    value = re.sub(r"\\cite\s*(?:\[([^\]]*)\])?\{([A-Za-z0-9:.,_ -]+)\}",
                   citation, value)
    return "".join(
        char if char.isascii() and (char.isprintable() or char.isspace()) else f"[U+{ord(char):04X}]"
        for char in value.translate(_UNICODE)
    )


def _sections(work: Path, source: Path, source_root: Path | None, draft: dict,
              candidates: list[dict] | tuple) -> list[tuple[str, str]]:
    excerpt = draft.get("source_excerpt")
    numbered = draft.get("formal_sources")
    reference_numbers = None
    if numbered:
        reference_numbers = {row["key"]: str(i) for i, row in enumerate(draft["references"], 1)}
        blocks = []
        for block in reading_blocks(draft):
            heading = block["heading"] or "Context"
            if block["caption"]:
                heading += " (" + block["caption"] + ")"
            blocks.append((heading, block["text"]))
    else:
        blocks = [] if excerpt else [("Introduction", draft.get("paragraph"))]
        blocks.extend((f"Historical result {chr(65 + i)}", row) for i, row in enumerate(draft.get("history") or []))
        blocks.extend(("Definition", row) for row in draft.get("shared_definitions") or [])
        for i, group in enumerate(draft.get("groups") or [], 1):
            if isinstance(group, dict):
                blocks.extend(("Definition", row) for row in group.get("definitions") or [])
                blocks.append((f"Theorem {i}", group.get("theorem")))
                blocks.append(("Context", group.get("paragraph")))
    blocks = [(title, value) for title, value in blocks if isinstance(value, str) and value.strip()]
    if not blocks:
        for candidate in candidates:
            if not isinstance(candidate.get("claim"), str) or candidate.get("decision") == "excluded":
                continue
            context = [candidate["claim"], candidate.get("assumptions"),
                       ", ".join(candidate.get("evidence_ids") or ())]
            label = candidate.get("source_number") or candidate.get("candidate_id") or "Extracted result"
            blocks.append((label, "\n".join(item for item in context if isinstance(item, str) and item.strip())))
    if not blocks and isinstance(excerpt, str) and excerpt.strip():
        blocks.append(("Source excerpt", excerpt))
    if not blocks:
        try:
            index = SourceIndex(source, source_root, work)
            catalog = index.catalog()
            item = next((row for row in catalog if re.search(r"(?i)\babstract\b", row["text"])), None)
            item = item or next((row for row in catalog if row["text"].strip()), None)
            if item:
                abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", item["text"], re.S)
                blocks.append((f"Source excerpt ({item['label']})", abstract[1].strip() if abstract else item["text"]))
        except (OSError, ValueError, RuntimeError):
            pass
    references = [f"[{(reference_numbers or {}).get(row['key'], row['key'])}] {row['text']}" for row in draft.get("references") or []
                  if isinstance(row, dict) and isinstance(row.get("key"), str) and isinstance(row.get("text"), str)]
    if not references:
        references.append(f"Supplied source: {source.name}.")
    references.append(f"Source file SHA-256: {v13_check.digest(source)}.")
    blocks.append(("References", "\n".join(references)))
    return [(_literal(heading, reference_numbers), _literal(body, reference_numbers)) for heading, body in blocks]


def _escape(value: str) -> str:
    return "".join(_ESCAPES.get(char, char) for char in value)


def _literal_document(case: Path, title: str, authors: str, sections: list[tuple[str, str]]) -> None:
    lines = [r"\documentclass[11pt,a4paper]{article}", r"\input{preamble-v13.tex}", _OPEN_DOCUMENT,
             r"{\Large\bfseries " + _escape(title) + r"\par}", _escape(authors), ""]
    for heading, body in sections:
        if heading == "References":
            lines.append(r"\clearpage")
        lines.append(r"\section*{" + _escape(heading) + "}")
        for paragraph in body.splitlines():
            lines += [_escape(line) for line in textwrap.wrap(paragraph, width=90)] + [""]
    lines.append(r"\end{document}")
    (case / "preamble-v13.tex").write_text(_LITERAL_PREAMBLE, encoding="utf-8")
    (case / "introduction.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _local_pdf(path: Path, title: str, authors: str, sections: list[tuple[str, str]]) -> None:
    # Only used when the bare TeX document cannot run in the local compiler environment.
    font = ImageFont.load_default(size=20)
    heading = ImageFont.load_default(size=26)
    pages = []
    offset = 0

    def write(value, face, width):
        nonlocal offset
        for line in textwrap.wrap(value, width=width) or [""]:
            if offset > 1320 or not pages:
                pages.append(Image.new("RGB", (1000, 1414), "white"))
                offset = 70
            ImageDraw.Draw(pages[-1]).text((72, offset), line, font=face, fill="black")
            offset += 30

    write(title, heading, 60)
    write(authors, font, 82)
    for label, body in sections:
        if label == "References":
            offset = 1414
        write(label, heading, 60)
        for paragraph in body.splitlines():
            write(paragraph, font, 82)
        offset += 18
    try:
        pages[0].save(path, format="PDF", resolution=120, save_all=True, append_images=pages[1:])
    finally:
        for page in pages:
            page.close()


def deliver_case(work: Path, source: Path, source_root: Path | None, draft: dict, *,
                 previous_case: Path | str | None = None, previous_draft: dict | None = None,
                 candidates: list[dict] | tuple = (), strict: bool = False) -> DeliveryCase:
    """Compile the latest structured draft or a verified, reproducible alternative."""
    work, source = Path(work).resolve(), Path(source).resolve()
    case = None
    try:
        snapshot = v13_check.source_snapshot(source, source_root)
        bundle = profile_bundle(ASSETS, work, release_for_draft(draft))
        checker = checker_for_bundle(bundle)
        mapped_blocks = []
        failed_line = None
        try:
            if strict:
                return _strict_formal(work, source, source_root, normalize_alignment(draft), bundle,
                                      snapshot["sha256"])
            tex = render_tex(draft, block_map=mapped_blocks, allow_empty_history=True)
            if previous_case and previous_draft == draft:
                prior = _prior(work, previous_case, source, snapshot["sha256"])
                if prior == Path(previous_case):
                    status = json.loads((prior / "build-status.json").read_text("utf-8"))
                    pinned = checker.verify(bundle)
                    if (status.get("engineering_accepted") and status.get("release") == pinned["release"]
                            and status.get("checker_sha256") == pinned["checker_sha256"]
                            and status.get("rules_sha256") == pinned["sha256"]
                            and (prior / "introduction.tex").read_text("utf-8") == tex):
                        # The exact current manuscript already passed the engineering
                        # checks. Preserve its verified artifact bytes for delivery.
                        return DeliveryCase(prior, "final")
            case = checker.allocate(bundle, source, work, "delivery", source_root)
            _write_formal(case, tex)
            return _receipt(case, bundle, "final", _compile(case), "")
        except _DELIVERY_ERRORS as exc:
            diagnostic = str(exc)
            failed_line = exc.line if isinstance(exc, _CompilerFailure) else None
        if prior := _prior(work, previous_case, source, snapshot["sha256"]):
            if failed_line is not None:
                try:
                    patched = _patch_from_prior(draft, mapped_blocks, previous_draft, prior, failed_line)
                    if patched:
                        tex = render_tex(patched, allow_empty_history=True)
                        case = checker.allocate(bundle, source, work, "delivery-repaired", source_root)
                        _write_formal(case, tex)
                        return _receipt(case, bundle, "repaired", _compile(case), diagnostic, draft=patched)
                except _DELIVERY_ERRORS:
                    pass
            layout_check, issues = _saved_layout(prior, Path(previous_case)) if strict else ("not_checked", ())
            return DeliveryCase(prior, "prior", diagnostic, layout_check=layout_check, layout_issues=issues)
        case = checker.allocate(bundle, source, work, "delivery-readable", source_root)
        title = _literal(draft.get("title") or source.stem)
        authors = _literal(", ".join(draft.get("authors") or []))
        sections = _sections(work, source, source_root, draft, candidates)
        _literal_document(case, title, authors, sections)
        try:
            passes = _compile(case)
        except (BuildFailure, OSError, subprocess.TimeoutExpired) as exc:
            passes = []
            diagnostic += f" 当前环境未完成安全 TeX 的双遍编译。{exc} 已由本机生成同内容的可读 PDF。"
            _literal_document(case, title, authors, sections)
            _local_pdf(case / "introduction.pdf", title, authors, sections)
        return _receipt(case, bundle, "readable", passes, diagnostic)
    except _DELIVERY_ERRORS as exc:
        raise BuildFailure(f"交付文件未能生成。{exc}", [], case_dir=case, technical=True) from exc
