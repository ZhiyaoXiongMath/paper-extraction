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
import textwrap

from PIL import Image, ImageDraw, ImageFont

from .pdfops import page_count, render_page
from .profile import ASSETS, BuildFailure, profile_bundle, release_for_draft, render_tex
from .profile_assets.tools import v13_check
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


@dataclass(frozen=True)
class DeliveryCase:
    path: Path
    kind: str
    diagnostic: str = ""
    compiler_verified: bool = True
    draft: dict | None = None


class _CompilerFailure(BuildFailure):
    def __init__(self, message, case, line):
        super().__init__(message, [], case_dir=case)
        self.line = line


def _compile(case: Path) -> list[int]:
    executable = shutil.which("pdflatex")
    if executable is None:
        raise BuildFailure("未找到 pdfLaTeX。", [], case_dir=case, technical=True)
    command = [executable, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error",
               "-no-shell-escape", "introduction.tex"]
    environment = {**os.environ, "openin_any": "p", "openout_any": "p"}
    inputs = (case / "introduction.tex", case / "preamble-v13.tex")
    before = [v13_check.digest(path) for path in inputs]
    returncodes = []
    for i in (1, 2):
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
        keys = set(next((link["evidence_ids"] for link in payload.get("evidence_links", [])
                         if link["block_id"] == row["draft_field"]), ()))
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


def _receipt(case: Path, bundle: Path, kind: str, returncodes: list[int], diagnostic: str,
             *, draft: dict | None = None) -> DeliveryCase:
    pages = _readable(case)
    binding = v13_check.check_binding(v13_check.verify(bundle), case)
    v13_check.atomic_json(case / "delivery.json", {
        "kind": kind,
        "diagnostic": diagnostic,
        "compiler_verified": returncodes == [0, 0],
        "compiler_returncodes": returncodes,
        "pdf_pages": pages,
        "files": {name: v13_check.digest(case / name) for name in _FILES},
        "input_binding": binding,
    })
    return DeliveryCase(case, kind, diagnostic, compiler_verified=returncodes == [0, 0], draft=draft)


def _literal(value: str) -> str:
    # Keep uncertain symbols identifiable instead of replacing a mathematical expression.
    value = re.sub(r"\\cite\s*(?:\[([^\]]*)\])?\{([A-Za-z0-9:.,_ -]+)\}",
                   lambda m: "[" + ", ".join(part for part in (m[2], m[1]) if part) + "]", value)
    return "".join(
        char if char.isascii() and (char.isprintable() or char.isspace()) else f"[U+{ord(char):04X}]"
        for char in value.translate(_UNICODE)
    )


def _sections(work: Path, source: Path, source_root: Path | None, draft: dict,
              candidates: list[dict] | tuple) -> list[tuple[str, str]]:
    excerpt = draft.get("source_excerpt")
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
    references = [f"[{row['key']}] {row['text']}" for row in draft.get("references") or []
                  if isinstance(row, dict) and isinstance(row.get("key"), str) and isinstance(row.get("text"), str)]
    if not references:
        references.append(f"Supplied source: {source.name}.")
    references.append(f"Source file SHA-256: {v13_check.digest(source)}.")
    blocks.append(("References", "\n".join(references)))
    return [(_literal(heading), _literal(body)) for heading, body in blocks]


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
                 candidates: list[dict] | tuple = ()) -> DeliveryCase:
    """Compile the latest structured draft or a verified, reproducible alternative."""
    work, source = Path(work).resolve(), Path(source).resolve()
    case = None
    try:
        snapshot = v13_check.source_snapshot(source, source_root)
        bundle = profile_bundle(ASSETS, work, release_for_draft(draft))
        mapped_blocks = []
        failed_line = None
        try:
            tex = render_tex(draft, block_map=mapped_blocks, allow_empty_history=True)
            case = v13_check.allocate(bundle, source, work, "delivery", source_root)
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
                        case = v13_check.allocate(bundle, source, work, "delivery-repaired", source_root)
                        _write_formal(case, tex)
                        return _receipt(case, bundle, "repaired", _compile(case), diagnostic, draft=patched)
                except _DELIVERY_ERRORS:
                    pass
            return DeliveryCase(prior, "prior", diagnostic)
        case = v13_check.allocate(bundle, source, work, "delivery-readable", source_root)
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
