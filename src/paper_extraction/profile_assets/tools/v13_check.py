#!/usr/bin/env python3
"""Optional deterministic checks, not mathematical or visual acceptance.

build defaults to bound regression; --mode compile-only explicitly skips source
identity. Exit 0 = engineering accepted in that mode, 1 = engineering rejection,
2 = preflight/runtime failure, 130 = interrupted after obtaining the build lock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RELEASE = 'intro-v13.0.0-rc1-20261023'
FILES = ('intro-rules-v13.txt', 'preamble-v13.tex', 'template-v13.tex', 'prompt-v13.txt')
TOOL = 'tools/v13_check.py'
HASH = re.compile(r'^[0-9a-f]{64}$')
SKIP_DIRS = {'.git', '.svn', '__pycache__', '.pytest_cache'}
BUILD_SUFFIXES = ('.aux', '.log', '.fls', '.out', '.toc', '.lof', '.lot', '.bcf',
                  '.run.xml', '.synctex.gz', '.fdb_latexmk', '.nav', '.snm', '.vrb')
PLACEHOLDERS = ('Exact title of the original paper', 'All original authors in source order',
                'All source authors', 'Exact source title', 'supplied version and date',
                'Replace this scaffold', 'Replace this historical statement', 'Replace this current statement')


class ValidationError(ValueError):
    """A source, manifest, environment or supported-mode requirement is unmet."""


class EngineeringRejected(Exception):
    """Compilation happened, but one or more deterministic checks failed."""


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Replace state atomically on the same filesystem; never truncate the old file."""
    tmp: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name + '-', delete=False) as f:
            tmp = f.name
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp and Path(tmp).exists():
            Path(tmp).unlink()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def string(value: Any, name: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), f'{name}: nonempty string required')
    return value


def hash_value(value: Any, name: str) -> str:
    require(isinstance(value, str) and HASH.fullmatch(value) is not None,
            f'{name}: lowercase SHA-256 required')
    return value


def relative_name(value: Any, name: str) -> str:
    value = string(value, name)
    p = Path(value)
    require(not p.is_absolute() and '..' not in p.parts and '\\' not in value,
            f'{name}: safe relative POSIX path required')
    require(value not in ('.', '') and p.as_posix() == value, f'{name}: noncanonical path')
    return value


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = root / 'SHA256SUMS.txt'
    require(manifest.is_file(), f'Missing release manifest: {manifest}')
    expected: dict[str, str] = {}
    for number, line in enumerate(manifest.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        require(len(parts) == 2, f'Invalid release manifest line {number}')
        h = hash_value(parts[0], f'manifest line {number}')
        name = relative_name(parts[1].strip().lstrip('*'), 'release file')
        require(name not in expected, f'Duplicate release file: {name}')
        path = root / name
        require(path.resolve().is_relative_to(root), f'Release file outside root: {name}')
        require(path.is_file(), f'Missing release file: {name}')
        require(digest(path) == h, f'Hash mismatch: {name}')
        expected[name] = h
    require(set(FILES + (TOOL,)).issubset(expected), 'Manifest must bind four companions and checker')
    for name in FILES:
        text = (root / name).read_text(encoding='utf-8')
        require(RELEASE in text, f'Release identity missing: {name}')
        require(re.search(r'(?:preamble|intro-rules|template|prompt)-v(?:9|10|11|12)\.', text) is None,
                f'Old companion reference: {name}')
    require(digest(Path(__file__)) == expected[TOOL], 'Executed checker differs from release checker')
    return {'release': RELEASE, 'rules_root': str(root),
            'sha256': {name: expected[name] for name in FILES},
            'checker_sha256': expected[TOOL], 'verified_release_files': len(expected)}


def slug(text: str) -> str:
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()
    return re.sub(r'[^A-Za-z0-9._-]+', '-', text).strip('.-_')[:100] or 'paper'


def snapshot_fingerprint(snapshot: dict[str, Any]) -> str:
    payload = {k: snapshot[k] for k in ('kind', 'entry', 'files')}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'))
                          .encode('utf-8')).hexdigest()


def source_snapshot(source: Path, source_root: Path | None = None) -> dict[str, Any]:
    source = source.resolve()
    require(source.is_file(), f'Source unavailable: {source}')
    require(source.suffix.lower() != '.tex' or source_root is not None,
            'TeX sources require --source-root for a declared self-contained input package')
    root = source_root.resolve() if source_root is not None else source.parent
    require(root.is_dir() and source.is_relative_to(root), 'Source must be inside source root')
    entry = source.relative_to(root).as_posix()
    files: list[dict[str, str]] = []
    if source_root is None:
        files.append({'path': entry, 'sha256': digest(source)})
    else:
        for path in sorted(root.rglob('*'), key=lambda path: path.relative_to(root).as_posix()):
            rel = path.relative_to(root)
            if any(part in SKIP_DIRS for part in rel.parts):
                continue
            require(not path.is_symlink(), f'Source-package symlink is unsupported: {rel}')
            if path.is_file() and not path.name.endswith(BUILD_SUFFIXES):
                files.append({'path': rel.as_posix(), 'sha256': digest(path)})
        require(any(f['path'] == entry for f in files), 'Root source was excluded from snapshot')
    result: dict[str, Any] = {'kind': 'declared_package' if source_root is not None else 'single_file',
                             'root': str(root), 'entry': entry, 'files': files,
                             'coverage': 'supplied files only; unprovided external/dynamic dependencies are not discovered'}
    result['sha256'] = snapshot_fingerprint(result)
    return result


def allocate(root: Path, source: Path, output_root: Path, case_id: str | None = None,
             source_root: Path | None = None) -> Path:
    info = verify(root)
    source = source.resolve()
    snap = source_snapshot(source, source_root)
    output_root = output_root.resolve()
    if source_root is not None:
        require(not output_root.is_relative_to(Path(snap['root'])),
                'Output root must not be inside the frozen source package')
    stem = f'{slug(case_id or source.stem)}--{snap["sha256"][:12]}'
    output_root.mkdir(parents=True, exist_ok=True)
    run = 1
    while True:
        out = output_root / (stem if run == 1 else f'{stem}--run-{run:03d}')
        try:
            out.mkdir()  # Atomic reservation, including parallel allocators.
            break
        except FileExistsError:
            run += 1
    try:
        shutil.copy2(root / 'preamble-v13.tex', out / 'preamble-v13.tex')
        binding = {'schema_version': 2, 'release': RELEASE, 'source': str(source),
                   'source_sha256': digest(source), 'snapshot': snap,
                   'rules_sha256': info['sha256'], 'checker_sha256': info['checker_sha256'],
                   'case_id': case_id or source.stem, 'output_directory': str(out),
                   'created_utc': utc()}
        atomic_json(out / 'input-manifest.json', binding)
    except Exception:
        # Only our newly reserved directory is affected; no other run is touched.
        shutil.rmtree(out)
        raise
    return out


def validate_binding_schema(data: Any) -> dict[str, Any]:
    require(isinstance(data, dict), 'Binding root must be a JSON object')
    for key in ('schema_version', 'release', 'source', 'source_sha256', 'snapshot',
                'rules_sha256', 'checker_sha256', 'output_directory', 'case_id', 'created_utc'):
        require(key in data, f'Binding missing field: {key}')
    require(type(data['schema_version']) is int and data['schema_version'] == 2,
            'Binding schema_version must be integer 2; reallocate rather than editing old evidence')
    for key in ('release', 'source', 'output_directory', 'case_id', 'created_utc'):
        string(data[key], key)
    for key in ('source_sha256', 'checker_sha256'):
        hash_value(data[key], key)
    rules = data['rules_sha256']
    require(isinstance(rules, dict) and set(rules) == set(FILES), 'Binding rules_sha256 has wrong keys/type')
    for key, value in rules.items():
        hash_value(value, key)
    snap = data['snapshot']
    require(isinstance(snap, dict), 'snapshot must be an object')
    for key in ('kind', 'root', 'entry', 'files', 'sha256'):
        require(key in snap, f'snapshot missing field: {key}')
    require(snap['kind'] in ('single_file', 'declared_package'), 'Invalid snapshot kind')
    require(Path(string(snap['root'], 'snapshot.root')).is_absolute(), 'snapshot.root must be absolute')
    relative_name(snap['entry'], 'snapshot.entry')
    hash_value(snap['sha256'], 'snapshot.sha256')
    require(isinstance(snap['files'], list) and bool(snap['files']), 'snapshot.files must be a nonempty list')
    seen: set[str] = set()
    for item in snap['files']:
        require(isinstance(item, dict) and set(item) == {'path', 'sha256'}, 'Invalid snapshot file entry')
        name = relative_name(item['path'], 'snapshot file path')
        require(name not in seen, f'Duplicate source entry: {name}')
        seen.add(name)
        hash_value(item['sha256'], 'snapshot file hash')
    require(snap['entry'] in seen, 'Root source not listed in snapshot')
    require(snap['files'] == sorted(snap['files'], key=lambda x: x['path']), 'Snapshot files must be sorted')
    require(snapshot_fingerprint(snap) == snap['sha256'], 'Snapshot fingerprint is inconsistent')
    return data


def check_binding(root_info: dict[str, Any], case: Path) -> dict[str, Any]:
    path = case / 'input-manifest.json'
    require(path.is_file(), 'Regression build requires input-manifest.json; allocate or explicitly use compile-only')
    binding = validate_binding_schema(json.loads(path.read_text(encoding='utf-8')))
    require(binding['release'] == RELEASE, 'Bound release mismatch')
    require(Path(binding['output_directory']).resolve() == case, 'Binding belongs to a different output directory')
    require(binding['rules_sha256'] == root_info['sha256'], 'Bound companion files changed')
    require(binding['checker_sha256'] == root_info['checker_sha256'], 'Bound checker changed')
    source = Path(binding['source'])
    require(source.is_absolute() and source.is_file(), 'Bound source unavailable or nonabsolute')
    require(digest(source) == binding['source_sha256'], 'Bound source bytes changed')
    snap = binding['snapshot']
    require((Path(snap['root']) / snap['entry']).resolve() == source.resolve(), 'Snapshot entry is not the bound source')
    current = source_snapshot(source, Path(snap['root']) if snap['kind'] == 'declared_package' else None)
    require(current['files'] == snap['files'] and current['sha256'] == snap['sha256'],
            'Source snapshot changed: dependency modified, added or removed')
    return {'checked': True, 'status': 'matched', 'manifest_sha256': digest(path),
            'snapshot_sha256': snap['sha256'], 'source_file_count': len(snap['files']),
            'scope': snap['kind'], 'coverage': current['coverage']}


def active_tex(text: str) -> str:
    """Remove ordinary TeX comments, respecting an escaped percent sign."""
    result = []
    for line in text.splitlines():
        for index, char in enumerate(line):
            if char != '%':
                continue
            count = 0
            j = index - 1
            while j >= 0 and line[j] == '\\':
                count += 1
                j -= 1
            if count % 2 == 0:
                line = line[:index]
                break
        result.append(line)
    return '\n'.join(result)


def bibliography_structure(tex: str) -> bool:
    text = active_tex(tex)
    starts = list(re.finditer(r'\\begin\s*\{thebibliography\}', text))
    ends = list(re.finditer(r'\\end\s*\{thebibliography\}', text))
    if len(starts) != 1 or len(ends) != 1 or ends[0].start() < starts[0].end():
        return False
    tail = text[ends[0].end():].strip()
    return re.fullmatch(r'\\end\s*\{document\}', tail) is not None


def classify_diagnostics(log: str) -> tuple[list[str], list[str]]:
    warnings = [line.strip() for line in log.splitlines()
                if re.search(r'Warning|Overfull|Underfull|Missing character|Undefined control', line)]
    block_patterns = (r'There were undefined references', r'(?:Citation|Reference).*undefined',
                      r'Missing character:', r'Overfull \\[hv]box', r'Undefined control sequence',
                      r'LaTeX Error', r'Package .+ Error', r'Font shape .+ undefined',
                      r'Some font shapes were not available', r'multiply defined',
                      r'Label\(s\) may have changed', r'Rerun to get cross-references right',
                      r'Rerun to get /PageLabels entry')
    blocking = [line.strip() for line in log.splitlines()
                if any(re.search(p, line, re.I) for p in block_patterns)]
    return blocking, warnings


def recorded_profiles(fls: str, case: Path) -> list[str]:
    paths: set[str] = set()
    for line in fls.splitlines():
        if line.startswith('INPUT '):
            path = Path(line[6:].strip().strip('"'))
            if re.fullmatch(r'preamble-v\d+\.tex', path.name):
                paths.add(str((path if path.is_absolute() else case / path).resolve()))
    return sorted(paths)


def aux_page(aux: str, name: str) -> int | None:
    matches = re.findall(r'\\newlabel\{' + re.escape(name) + r'\}\{\{[^{}]*\}\{(\d+)\}', aux)
    return int(matches[0]) if len(matches) == 1 else None


def inspect_pdf(pdf: Path, aux: str, tex: str) -> dict[str, Any]:
    import pypdfium2 as pdfium
    import pypdfium2.raw as raw
    start = aux_page(aux, 'intro:references-start')
    end = aux_page(aux, 'intro:references-end')
    sizes = []
    headings = []
    line_pages = []
    graphic_pages = []
    with pdfium.PdfDocument(pdf) as doc:
        for number, page in enumerate(doc, 1):
            width, height = page.get_size()
            sizes.append([round(width, 2), round(height, 2)])
            require(page.get_rotation() == 0, 'Rotated pages are outside the v13 print profile')
            left, _, _, top = page.get_bbox()
            def from_pdf(bounds):
                x0, y0, x1, y1 = bounds
                return [x0 - left, top - y1, x1 - left, top - y0]

            text_page = page.get_textpage()
            lines = []
            chars = []
            boxes = []
            def finish_line():
                text = ''.join(chars).strip()
                if text and boxes:
                    bbox = [min(b[0] for b in boxes), min(b[1] for b in boxes),
                            max(b[2] for b in boxes), max(b[3] for b in boxes)]
                    if text != str(number) or bbox[1] <= height - 60:
                        lines.append({'text': text, 'bbox': bbox})
                    if text == 'References':
                        headings.append({'page': number, 'bbox': bbox})
                chars.clear()
                boxes.clear()

            for index in range(text_page.count_chars()):
                char = chr(raw.FPDFText_GetUnicode(text_page, index))
                if char in '\r\n':
                    finish_line()
                elif char != '\x00':
                    chars.append(char)
                    if not char.isspace():
                        boxes.append(from_pdf(text_page.get_charbox(index, loose=True)))
            finish_line()
            text_page.close()
            line_pages.append(lines)
            graphics = []
            for obj in page.get_objects():
                # Use leaf bounds in the coordinate system of the containing form.
                if obj.type == raw.FPDF_PAGEOBJ_FORM and obj.level >= 14:
                    raise ValidationError('PDF graphics exceed supported nesting depth')
                if obj.type not in (raw.FPDF_PAGEOBJ_PATH, raw.FPDF_PAGEOBJ_IMAGE, raw.FPDF_PAGEOBJ_SHADING):
                    continue
                bounds = obj.get_bounds()
                container = obj.container
                while container is not None:
                    bounds = container.get_matrix().on_rect(*bounds)
                    container = container.container
                graphics.append(from_pdf(bounds))
            graphic_pages.append(graphics)
            page.close()
        actual = [h for h in headings if h['page'] == start]
        located = (bibliography_structure(tex) and start is not None and end == len(doc)
                   and 1 <= start <= len(doc) and len(actual) == 1)
        before: list[str] = []
        graphics_before: list[list[float]] = []
        if located:
            y = actual[0]['bbox'][1]
            before = [line['text'] for line in line_pages[start - 1] if line['bbox'][1] < y - 0.5]
            graphics_before = [box for box in graphic_pages[start - 1] if box[1] < y - 0.5]
        separate = bool(located and not before and not graphics_before and start > 1)
        body_pages = (start - 1 if separate else start) if located else None
        return {'pdf_pages': len(doc), 'page_sizes_pt': sizes,
                'a4': all(abs(w - 595.28) < 1 and abs(h - 841.89) < 1 for w, h in sizes),
                'reference_marker_start_page': start, 'reference_marker_end_page': end,
                'reference_headings': headings, 'bibliography_located': bool(located),
                'references_separate': separate, 'body_before_references': before,
                'graphics_before_references': graphics_before,
                'main_text_pages': body_pages,
                'three_page_limit': body_pages is not None and 1 <= body_pages <= 3}



# v13 adds a deterministic reader-format contract. This is deliberately NOT a
# mathematical importance classifier or an adversarial TeX sandbox.
BLOCK_LIMITS = {
    'historical': {'kind': 'H', 'words': 75, 'units': 8, 'displays': 1, 'rows': 2},
    'theorem': {'kind': 'T', 'words': 100, 'units': 12, 'displays': 2, 'rows': 3},
    'definition': {'kind': 'D', 'words': 65, 'units': 7, 'displays': 2, 'rows': 3},
    'introparagraph': {'kind': 'P', 'words': 45, 'units': 4, 'displays': 0, 'rows': 0},
}


def balanced(text: str, start: int, opening: str = '{', closing: str = '}') -> tuple[str, int]:
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] != opening:
        raise ValidationError('Expected a balanced ' + opening + closing + ' argument')
    depth, i = 1, start + 1
    while i < len(text):
        if text[i] == '\\':
            i += 2
            continue
        if text[i] == opening:
            depth += 1
        elif text[i] == closing:
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
        i += 1
    raise ValidationError('Unclosed argument')


def strip_named_commands(text: str, commands: tuple[str, ...]) -> str:
    pat = re.compile(r'\\(?:' + '|'.join(commands) + r')\*?\b')
    while True:
        match = pat.search(text)
        if not match:
            return text
        end = match.end()
        while end < len(text) and text[end].isspace():
            end += 1
        if end < len(text) and text[end] == '[':
            _, end = balanced(text, end, '[', ']')
        try:
            _, end = balanced(text, end)
        except ValidationError:
            end = match.end()
        text = text[:match.start()] + ' ' + text[end:]


def prose_words(text: str) -> int:
    """Supported-profile lexical count. Actual TeX height is a separate gate."""
    text = strip_named_commands(text, ('cite', 'ref', 'eqref', 'pageref', 'label'))
    extras = []
    # Words in text/mbox inside math still count; a hidden prose box is not free.
    def discard_math(match):
        body = match.group(0)
        extras.extend(re.findall(r'\\(?:text|textrm|textnormal|mbox)\s*\{([^{}]*)\}', body))
        return ' '
    text = re.sub(r'\\begin\{(?:conditions|relations)\}.*?\\end\{(?:conditions|relations)\}', discard_math, text, flags=re.S)
    text = re.sub(r'\\\[.*?\\\]|\\\(.*?\\\)|(?<!\\)\$.*?(?<!\\)\$', discard_math, text, flags=re.S)
    text += ' ' + ' '.join(extras)
    text = re.sub(r'\\(?:begin|end)\s*\{[^{}]*\}', ' ', text)
    text = re.sub(r'\\[A-Za-z@]+\*?', ' ', text)
    return len(re.findall(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", text, flags=re.UNICODE))


def environment_nodes(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    nodes, stack, errors = [], [], []
    for match in re.finditer(r'\\(begin|end)\s*\{([A-Za-z*]+)\}', text):
        direction, name = match.groups()
        if direction == 'begin':
            node = {'name': name, 'start': match.start(), 'body_start': match.end(),
                    'parent': stack[-1] if stack else None}
            stack.append(node)
        elif not stack or stack[-1]['name'] != name:
            errors.append('unbalanced_environment:' + name)
        else:
            node = stack.pop()
            node.update(body_end=match.start(), end=match.end())
            nodes.append(node)
    if stack:
        errors.append('unclosed_environments')
    return sorted(nodes, key=lambda n: n['start']), errors


def display_bodies(text: str) -> list[tuple[str, str]]:
    result = []
    # The supported source uses either \[...\] or one provided wrapper.
    for match in re.finditer(r'\\begin\{(conditions|relations)\}(.*?)\\end\{\1\}|\\\[(.*?)\\\]', text, re.S):
        result.append((match.group(1) or 'display', match.group(2) if match.group(1) else match.group(3)))
    return result


def source_rows(body: str) -> list[str]:
    rows = re.split(r'\\\\(?:\s*\[[^\]]*\])?', body)
    return [row.strip() for row in rows if row.strip()]


def format_contract(tex: str, log: str, pdf: Path) -> dict[str, Any]:
    text = active_tex(tex)
    errors: list[str] = []
    match = re.search(r'\\begin\{document\}(.*?)\\begin\{thebibliography\}', text, re.S)
    if not match:
        return {'accepted': False, 'errors': ['body_or_bibliography_missing']}
    body = match.group(1)
    nodes, parse_errors = environment_nodes(body)
    errors.extend(parse_errors)
    allowed = set(BLOCK_LIMITS) | {'resultgroup', 'conditions', 'relations', 'aligned', 'cases', 'enumerate', 'matrix', 'pmatrix', 'bmatrix'}
    for node in nodes:
        if node['name'] not in allowed:
            errors.append('unsupported_environment:' + node['name'])
    # No raw internal environments, layout reductions, pasted graphics or prose macros.
    forbidden = (r'\\(?:renewcommand|renewenvironment|def|gdef|edef|xdef)\b',
                 r'\\(?:fontsize|small|footnotesize|scriptsize|tiny|Huge|LARGE|large|linespread|geometry|newgeometry|setlength|addtolength|vspace|hspace|hfill|vfill|resizebox|scalebox|includegraphics|footnote|input|include|endinput|newpage|pagebreak)\b',
                 r'\\(?:IntroBudgetError|IntroMeasured|IntroUnit|IntroLimit|IntroGroupHeight|IntroProseHeight|IntroSharedHeight|setcounter|addtocounter|typeout|write|immediate)\b',
                 r'\$\$', r'\\begin\{(?:rawtheorem|rawdefinition|rawhistorical|proof|remark|corollary|proposition|align\*?|alignat\*?|flalign\*?|eqnarray\*?|gather\*?|multline\*?|array)\}')
    for pattern in forbidden:
        if re.search(pattern, body):
            errors.append('forbidden_body_command:' + pattern)
    # All visible prose must live in a measured block, not between groups.
    def stray(segment: str) -> bool:
        segment = strip_named_commands(segment, ('label',))
        segment = re.sub(r'\\clearpage\b', '', segment)
        return bool(segment.strip())
    top = [n for n in nodes if n['parent'] is None]
    remaining = body
    for node in sorted(top, key=lambda n: n['start'], reverse=True):
        remaining = remaining[:node['start']] + ' ' + remaining[node['end']:]
    # Header and single Introduction heading are the only unmeasured body commands.
    for command, nargs in (('paperheader', 2), ('section', 1)):
        pat = re.compile(r'\\' + command + r'\*?\b')
        occurrences = list(pat.finditer(remaining))
        if len(occurrences) != 1:
            errors.append('header_structure:' + command)
        for m in reversed(occurrences):
            end = m.end()
            try:
                for _ in range(nargs):
                    _, end = balanced(remaining, end)
                remaining = remaining[:m.start()] + ' ' + remaining[end:]
            except ValidationError:
                errors.append('malformed_header')
    if stray(remaining):
        errors.append('unmeasured_body_text')
    blocks = []
    counts = {name: 0 for name in BLOCK_LIMITS}
    first_current = min((n['start'] for n in nodes if n['name'] == 'theorem'), default=None)
    for node in nodes:
        name = node['name']
        if name == 'resultgroup':
            children = [n for n in nodes if n['parent'] is node]
            if any(n['name'] not in ('definition', 'theorem', 'introparagraph') for n in children):
                errors.append('invalid_resultgroup_child')
            inner = body[node['body_start']:node['body_end']]
            for child in sorted(children, key=lambda n: n['start'], reverse=True):
                a, b = child['start'] - node['body_start'], child['end'] - node['body_start']
                inner = inner[:a] + ' ' + inner[b:]
            if stray(inner):
                errors.append('unmeasured_resultgroup_text')
            if sum(n['name'] == 'theorem' for n in children) != 1:
                errors.append('resultgroup_requires_exactly_one_theorem')
            if sum(n['name'] == 'definition' for n in children) > 2:
                errors.append('too_many_dedicated_definitions')
        if name not in BLOCK_LIMITS:
            continue
        if node['parent'] is not None and node['parent']['name'] != 'resultgroup':
            errors.append('nested_measured_block:' + name)
        if name == 'theorem' and (node['parent'] is None or node['parent']['name'] != 'resultgroup'):
            errors.append('theorem_missing_resultgroup')
        if name == 'historical' and first_current is not None and node['start'] > first_current:
            errors.append('history_after_current_results')
        counts[name] += 1
        payload = body[node['body_start']:node['body_end']].lstrip()
        caption = ''
        if payload.startswith('['):
            try:
                caption, end = balanced(payload, 0, '[', ']')
                payload = payload[end:]
            except ValidationError:
                errors.append('malformed_caption')
        nw = prose_words(payload)
        displays = display_bodies(payload)
        nr = sum(len(source_rows(x)) for _, x in displays)
        cap = BLOCK_LIMITS[name]
        if nw > cap['words']:
            errors.append(f'{name}_{counts[name]}_word_limit:{nw}>{cap["words"]}')
        if len(displays) > cap['displays'] or nr > cap['rows']:
            errors.append(f'{name}_{counts[name]}_display_budget')
        if prose_words(caption) > 12:
            errors.append('caption_word_limit')
        # A formal source locator is required; its correctness is a manual source check.
        if name != 'introparagraph' and not re.search(r'\\cite\b', caption + payload):
            errors.append(f'{name}_{counts[name]}_missing_source_locator')
        for display_kind, formula in displays:
            if len(source_rows(formula)) > 3:
                errors.append('display_more_than_three_source_rows')
            if display_kind == 'conditions':
                if any(not row.startswith('&') or len(re.findall(r'(?<!\\)&', row)) != 1 for row in source_rows(formula)):
                    errors.append('independent_conditions_must_share_left_edge')
            elif display_kind == 'relations':
                if any(len(re.findall(r'(?<!\\)&', row)) != 1 for row in source_rows(formula)):
                    errors.append('relation_chain_requires_one_anchor_per_row')
            elif re.search(r'\\begin\{aligned\}', formula):
                errors.append('use_conditions_or_relations_for_multiline')
            if re.search(r'\\(?:hfill|hspace|quad|qquad)\b.*\\(?:quad|qquad|hfill|hspace)\b', formula, re.S):
                errors.append('manually_stretched_formula')
        blocks.append({'environment': name, 'number': counts[name], 'prose_words': nw,
                       'display_groups': len(displays), 'display_source_rows': nr})
    if not 1 <= counts['historical'] <= 3:
        errors.append('historical_count_must_be_1_to_3')
    if not 1 <= counts['theorem'] <= 4:
        errors.append('current_count_must_be_1_to_4')
    if counts['definition'] > 6:
        errors.append('definition_count_exceeds_6')
    telemetry = []
    for m in re.finditer(r'^INTRO-BLOCK\|(\d+)\|([HDTP])\|(\d+)\|(\d+)\|(\d+)\s*$', log, re.M):
        number, kind, height, limit, group = m.groups()
        telemetry.append({'index': int(number), 'kind': kind, 'height_sp': int(height),
                          'limit_sp': int(limit), 'group': int(group)})
    unit_match = re.search(r'^INTRO-UNIT-SP:\s*(\d+)\s*$', log, re.M)
    unit = int(unit_match.group(1)) if unit_match else None
    if len(telemetry) != len(blocks) or unit is None:
        errors.append('missing_or_inconsistent_measurement_telemetry')
    else:
        for block, measure in zip(blocks, telemetry):
            cap = BLOCK_LIMITS[block['environment']]
            if measure['kind'] != cap['kind'] or measure['limit_sp'] != unit * cap['units']:
                errors.append('measurement_identity_or_limit_mismatch')
            if measure['height_sp'] > measure['limit_sp']:
                errors.append('rendered_block_height_exceeded')
            block['height_units'] = round(measure['height_sp'] / unit, 3)
            block['height_limit_units'] = cap['units']
    groups = []
    for m in re.finditer(r'^INTRO-GROUP\|(\d+)\|(\d+)\|(\d+)\|(\d+)\|(\d+)\s*$', log, re.M):
        index, height, limit, theorems, definitions = map(int, m.groups())
        groups.append({'group': index, 'height_units': round(height/unit, 3) if unit else None,
                       'height_limit_units': 22, 'theorems': theorems, 'definitions': definitions})
        if not unit or limit != 22 * unit or height > limit or theorems != 1 or definitions > 2:
            errors.append('resultgroup_measurement_failed')
    if len(groups) != counts['theorem']:
        errors.append('resultgroup_telemetry_missing')
    # Verify displayed headings, not only source environment names.
    try:
        import pypdfium2 as pdfium
        with pdfium.PdfDocument(pdf) as document:
            chunks = []
            for page in document:
                text_page = page.get_textpage()
                chunks.append(text_page.get_text_range())
                text_page.close()
                page.close()
        visible = '\n'.join(chunks)
        for i in range(1, counts['theorem'] + 1):
            if not re.search(r'Theorem\s+' + str(i) + r'(?![\d.])', visible):
                errors.append('flat_current_heading_missing:' + str(i))
        for i in range(counts['historical']):
            if not re.search(r'Theorem\s+' + chr(65 + i) + r'\b', visible):
                errors.append('historical_heading_missing:' + chr(65+i))
    except Exception as exc:
        errors.append('heading_inspection_unavailable:' + type(exc).__name__)
    return {'accepted': not errors, 'errors': list(dict.fromkeys(errors)), 'counts': counts,
            'unit_tex_pt': unit/65536 if unit else None, 'blocks': blocks, 'groups': groups,
            'meaning_and_tier_judgment': 'not_automatically_certified'}


def run_compiler(command: list[str], case: Path, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=case, capture_output=True, text=True,
                          encoding='utf-8', errors='replace', timeout=timeout)


def _build(root: Path, case: Path, mode: str, timeout: float, state: dict[str, Any]) -> None:
    info = verify(root)
    src, local = case / 'introduction.tex', case / 'preamble-v13.tex'
    require(src.is_file(), f'Missing generated source: {src}')
    require(local.is_file() and digest(local) == info['sha256']['preamble-v13.tex'],
            'Local profile missing or not identical to the frozen release')
    if mode == 'regression':
        state['input_binding'] = check_binding(info, case)
        state['input_binding_checked'] = True
    else:
        state['input_binding'] = {'checked': False, 'status': 'not_checked_by_explicit_compile_only',
                                  'manifest_present': (case / 'input-manifest.json').is_file()}
    state['rules_sha256'] = info['sha256']
    state['checker_sha256'] = info['checker_sha256']
    state['tex_sha256'] = digest(src)
    tex = src.read_text(encoding='utf-8')
    require('\\input{preamble-v13.tex}' in active_tex(tex),
            'Checker supports a separate local preamble only; embedded/alternate inputs require manual verification')
    exe = shutil.which('pdflatex')
    require(exe is not None, 'pdfLaTeX is not installed')
    command = [exe, '-interaction=nonstopmode', '-halt-on-error', '-file-line-error',
               '-no-shell-escape', '-recorder', 'introduction.tex']
    state['command'], state['returncodes'] = command, []
    log = ''
    for number in range(1, 4):
        if number == 3 and not re.search(r'Label\(s\) may have changed|Rerun to get|rerunfilecheck Warning', log):
            break
        proc = run_compiler(command, case, timeout)
        (case / f'build-pass-{number}.txt').write_text(proc.stdout + proc.stderr, encoding='utf-8')
        state['returncodes'].append(proc.returncode)
        log = (case / 'introduction.log').read_text(encoding='utf-8', errors='replace') if (case / 'introduction.log').exists() else ''
        if proc.returncode:
            raise ValidationError(f'pdfLaTeX failed on pass {number}, exit {proc.returncode}; see log')
    pdf = case / 'introduction.pdf'
    require(pdf.is_file() and pdf.stat().st_size > 0, 'Compiler did not create a nonempty PDF')
    state['compiled'] = True
    state['compiled_pdf_sha256'] = digest(pdf)
    # Detect edits during the actual run, as well as stale inputs at entry.
    require(digest(src) == state['tex_sha256'], 'Generated TeX changed during build')
    require(verify(root)['sha256'] == info['sha256'] and digest(local) == info['sha256']['preamble-v13.tex'],
            'Release or local profile changed during build')
    if mode == 'regression':
        require(check_binding(info, case)['manifest_sha256'] == state['input_binding']['manifest_sha256'],
                'Binding changed during build')
    blocking, warnings = classify_diagnostics(log)
    state['blocking_diagnostics'], state['diagnostics'] = blocking, warnings
    fls_path = case / 'introduction.fls'
    fls = fls_path.read_text(errors='replace') if fls_path.exists() else ''
    profiles = recorded_profiles(fls, case)
    state['recorded_preambles'] = profiles
    state['release_assertion_match'] = bool(re.search(r'^INTRO-RELEASE-CHECK: MATCH\s*$', log, re.M))
    state['profile_identity_match'] = f'INTRO-PROFILE-RELEASE: {RELEASE}' in log
    state['local_profile_used'] = profiles == [str(local.resolve())]
    state['placeholder_hits'] = [p for p in PLACEHOLDERS if p in active_tex(tex)]
    aux = (case / 'introduction.aux').read_text(errors='replace')
    page_info = inspect_pdf(pdf, aux, tex)
    state.update(page_info)
    checks = {'release_assertion_match': state['release_assertion_match'],
              'profile_identity_match': state['profile_identity_match'],
              'local_profile_used': state['local_profile_used'],
              'no_blocking_diagnostics': not blocking,
              'no_scaffold_placeholders': not state['placeholder_hits'],
              'a4': state['a4'], 'bibliography_located': state['bibliography_located'],
              'references_separate': state['references_separate'],
              'three_page_limit': state['three_page_limit'],
              'input_binding_required_and_valid': mode == 'compile-only' or state['input_binding_checked']}
    state['format_contract'] = format_contract(tex, log, pdf)
    checks['v13_reader_format_contract'] = state['format_contract']['accepted']
    state['checks'] = checks
    failed = [name for name, ok in checks.items() if not ok]
    require(all(type(v) is bool for v in checks.values()), 'Internal error: nonboolean gate')
    if failed:
        state['failed_checks'] = failed
        raise EngineeringRejected('Engineering checks failed: ' + ', '.join(failed))
    state['engineering_accepted'] = True
    state['regression_eligible'] = mode == 'regression' and state['input_binding_checked']


def clear_pdf(case: Path) -> None:
    pdf = case / 'introduction.pdf'
    if pdf.exists() or pdf.is_symlink():
        pdf.unlink()


def build(root: Path, case: Path, mode: str = 'regression', timeout: float = 60) -> dict[str, Any]:
    """Invalidate before preflight; all owned-build failures have one cleanup path.

    A busy directory is not ours: do not delete another process's PDF/status.
    Filesystem/kill failures are reported or remain running, never reported accepted.
    """
    case = case.resolve()
    state: dict[str, Any] = {'schema_version': 2, 'release': RELEASE, 'run_id': uuid.uuid4().hex,
                            'started_utc': utc(), 'case_directory': str(case), 'mode': mode,
                            'state': 'running', 'compiled': False, 'engineering_accepted': False,
                            'regression_eligible': False, 'input_binding_checked': False,
                            'content_check': 'not_assessed_by_this_tool',
                            'visual_check': 'not_performed_by_this_tool', 'pdf_sha256': None}
    if not case.is_dir():
        return {**state, 'state': 'not_started', 'error': 'Case directory does not exist', 'exit_code': 2}
    lock = case / '.intro-build.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return {**state, 'state': 'not_started', 'error': 'Build lock exists; active run not modified', 'exit_code': 2}
    except OSError as exc:
        return {**state, 'state': 'not_started', 'error': str(exc), 'exit_code': 2}
    code = 0
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump({'run_id': state['run_id'], 'pid': os.getpid()}, f)
        clear_pdf(case)  # Before manifest parsing, environment checks or any possible old PASS.
        atomic_json(case / 'build-status.json', state)
        for suffix in ('.aux', '.out', '.toc', '.log', '.fls'):
            path = case / ('introduction' + suffix)
            if path.exists() or path.is_symlink():
                path.unlink()
        for path in case.glob('build-pass-*.txt'):
            path.unlink()
        require(mode in ('regression', 'compile-only'), 'Unknown build mode')
        require(isinstance(timeout, (int, float)) and 0 < timeout <= 3600, 'Invalid timeout')
        _build(root.resolve(), case, mode, timeout, state)
        state['state'] = 'accepted'
        state['pdf_sha256'] = digest(case / 'introduction.pdf')
    except EngineeringRejected as exc:
        code = 1
        state.update(state='rejected', error=str(exc), error_type=type(exc).__name__)
    except KeyboardInterrupt as exc:
        code = 130
        state.update(state='failed', error='Build interrupted', error_type=type(exc).__name__)
    except Exception as exc:
        code = 2
        state.update(state='failed', error=str(exc), error_type=type(exc).__name__)
    finally:
        if code:
            state['engineering_accepted'] = False
            state['regression_eligible'] = False
            state['pdf_sha256'] = None
            try:
                clear_pdf(case)
                state['deliverable_cleared'] = True
            except OSError as exc:
                state['deliverable_cleared'] = False
                state['cleanup_error'] = str(exc)
                code = 2
        state['pdf_available'] = (case / 'introduction.pdf').is_file()
        state['exit_code'], state['finished_utc'] = code, utc()
        try:
            atomic_json(case / 'build-status.json', state)
        except Exception as exc:
            # Persistence itself can fail. Never return an accepted PDF in that case.
            state = {'schema_version': 2, 'release': RELEASE, 'run_id': state['run_id'],
                     'case_directory': str(case), 'mode': mode, 'state': 'failed',
                     'compiled': bool(state.get('compiled')), 'engineering_accepted': False,
                     'regression_eligible': False, 'input_binding_checked': False,
                     'content_check': 'not_assessed_by_this_tool',
                     'visual_check': 'not_performed_by_this_tool', 'exit_code': 2,
                     'pdf_sha256': None, 'state_write_error': str(exc), 'finished_utc': utc()}
            try:
                clear_pdf(case)
            except OSError as clean_error:
                state['cleanup_error'] = str(clean_error)
            state['pdf_available'] = (case / 'introduction.pdf').is_file()
            try:
                atomic_json(case / 'build-status.json', state)
            except Exception as persist_error:
                state['failure_state_persist_error'] = str(persist_error)
        try:
            lock.unlink()
        except OSError as exc:
            # A stale lock prevents later builds, rather than admitting an unchecked run.
            state['lock_cleanup_warning'] = str(exc)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('verify', 'allocate', 'build'))
    parser.add_argument('--rules-dir', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--source', type=Path)
    parser.add_argument('--source-root', type=Path, help='Required for TeX: explicit isolated supplied source package')
    parser.add_argument('--output-root', type=Path, default=Path('introduction-v13'))
    parser.add_argument('--case-id')
    parser.add_argument('--case-dir', type=Path)
    parser.add_argument('--mode', choices=('regression', 'compile-only'), default='regression')
    parser.add_argument('--timeout', type=float, default=60)
    args = parser.parse_args()
    try:
        if args.action == 'verify':
            result, code = verify(args.rules_dir), 0
        elif args.action == 'allocate':
            if args.source is None:
                parser.error('--source is required for allocate')
            result = {'output_directory': str(allocate(args.rules_dir, args.source, args.output_root,
                                                      args.case_id, args.source_root))}
            code = 0
        else:
            if args.case_dir is None:
                parser.error('--case-dir is required for build')
            result = build(args.rules_dir, args.case_dir, args.mode, args.timeout)
            code = result['exit_code']
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return code
    except Exception as exc:
        print(json.dumps({'error': str(exc), 'error_type': type(exc).__name__, 'exit_code': 2},
                         ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
