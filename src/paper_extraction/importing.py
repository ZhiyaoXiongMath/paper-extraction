"""Stage user-selected files and identify paper roots before a batch starts."""
import json
import re
import shutil
from pathlib import Path, PurePosixPath
from uuid import uuid4

from .sources import paper_files, read_tex, uncomment

SOURCE_EXTENSIONS = {".pdf", ".tex", ".bib", ".bbl", ".sty", ".cls", ".bst", ".def",
                     ".png", ".jpg", ".jpeg", ".eps"}
_DOC = re.compile(r"\\documentclass\b|\\begin\s*\{document\}")
_RESERVED = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", re.I)


def relative_path(raw: str):
    raw = raw.replace("\\", "/")
    parts = raw.split("/")
    if any(part in ("", ".", "..") or part.startswith(".") or ":" in part
           or part.endswith((" ", ".")) or _RESERVED.match(part)
           or any(c in '<>"|?*' or ord(c) < 32 for c in part) for part in parts):
        raise ValueError("文件需要有效的相对路径")
    return PurePosixPath(raw)


def discover(root: Path, names: list[str]):
    root = root.resolve()
    tex_paths = [PurePosixPath(name) for name in names if name.lower().endswith(".tex")]
    contents = {p: uncomment(read_tex(root / p)) for p in tex_paths}
    declared = [p for p, text in contents.items() if _DOC.search(text)]

    def closure(entry):
        return {PurePosixPath(file.relative_to(root).as_posix())
                for file in paper_files(root / entry, root, strict=False)}

    # A subfiles chapter can contain its own documentclass. An explicit include edge
    # from the parent manuscript identifies it as a chapter.
    deps = {p: closure(p) for p in declared}
    mains = [p for p in declared if not any(p in dep - {other} for other, dep in deps.items())]
    if not mains and declared:
        mains = declared  # Let source indexing diagnose a cyclic input.
    loose = [p for p in tex_paths if p not in declared
             and not any(p in deps[m] or p.is_relative_to(m.parent) for m in mains)]
    deps.update({p: closure(p) for p in loose})
    mains.extend(p for p in loose if not any(p in deps[other] for other in loose if other != p))

    title_pattern = re.compile(r"\\title(?:\s*\[[^\]]*\])?\s*\{([^{}]+)\}")
    titles = {}
    for p in mains:
        match = title_pattern.search(contents[p])
        titles[p] = " ".join(match[1].split()).casefold() if match else ""

    def alternatives(a, b):
        if a.parent != b.parent:
            return False
        if titles[a] and titles[b]:
            return titles[a] == titles[b]
        # Shared style, graphics, bibliography, and notation files alone do not
        # establish that two independent articles are alternate builds of one paper.
        content_files = (deps[a] & deps[b]) - {a, b}
        return any(path.suffix.lower() == ".tex" and path.stem.casefold() not in {
            "macros", "commands", "preamble", "notation", "definitions", "defs", "packages"
        } for path in content_files)

    groups = []
    for entry in mains:
        for group in groups:
            if all(alternatives(entry, sibling) for sibling in group):
                group.append(entry)
                break
        else:
            groups.append([entry])

    papers = []

    def append(kind, name, entries, files, per_entry=None):
        paper = dict(id=uuid4().hex, kind=kind, name=name, entries=entries,
                     entry=entries[0] if len(entries) == 1 else "",
                     files=files, file_count=len(files))
        if per_entry:
            paper["entry_files"] = per_entry
        papers.append(paper)

    figures = {f for d in deps.values() for f in d if f.suffix.lower() == ".pdf"}
    corresponding_pdfs = {m.with_suffix(".pdf") for m in mains}
    for name in names:
        p = PurePosixPath(name)
        if p.suffix.lower() == ".pdf" and p not in figures and p not in corresponding_pdfs:
            append("pdf", p.stem, [name], [name])

    for group in groups:
        entries = [str(p) for p in group]
        by_entry = {str(p): sorted(str(f) for f in deps[p]) for p in group}
        files = sorted({f for listing in by_entry.values() for f in listing})
        first = group[0]
        name = first.stem if len(group) == 1 else first.parent.name or first.stem
        if name.casefold() in {"main", "paper", "article", "ms"} and first.parent.name:
            name = first.parent.name
        append("tex", name, entries, files, by_entry)
    return papers


def stage(home, files, paths):
    if len(files) != len(paths) or not files:
        raise ValueError("请选择至少一个 PDF 文件或 TeX 项目")
    token = uuid4().hex
    directory = home / "imports" / token
    source = directory / "source"
    source.mkdir(parents=True)
    seen = set()
    names = []
    try:
        for upload, raw in zip(files, paths):
            # File pickers can include editor caches, hidden VCS and builds.
            normalized = raw.replace("\\", "/")
            if any(p.startswith(".") and p not in (".", "..") for p in normalized.split("/")):
                continue
            relative = relative_path(normalized)
            if relative.suffix.lower() not in SOURCE_EXTENSIONS:
                continue
            name = str(relative)
            if name.casefold() in seen:
                raise ValueError(f"所选位置有重名文件：{name}")
            seen.add(name.casefold())
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as dest:
                shutil.copyfileobj(upload.file, dest)
            names.append(name)
        papers = discover(source, names)
        if not papers:
            raise ValueError("没有找到 PDF 或 TeX 论文")
        result = {"id":token, "papers":papers}
        (directory/"manifest.json").write_text(json.dumps(result, ensure_ascii=False), "utf-8")
        return result
    except Exception:
        shutil.rmtree(directory)
        raise


def prepare_sources(home, import_id, selected, batch_id):
    if not re.fullmatch(r"[0-9a-f]{32}", import_id):
        raise ValueError("导入编号无效")
    base = home/"imports"/import_id
    if not (base/"manifest.json").is_file():
        raise ValueError("导入预览不存在")
    manifest = json.loads((base/"manifest.json").read_text("utf-8"))
    options = {p["id"]:p for p in manifest["papers"]}
    if not selected or len({p.id for p in selected}) != len(selected):
        raise ValueError("需要选择论文，且每篇只能选择一次")
    papers = []
    directory = home/"runs"/batch_id
    try:
        for choice in selected:
            spec = options.get(choice.id)
            if spec is None or choice.entry not in spec["entries"]:
                raise ValueError("请选择导入预览中的主文件")
            pid = uuid4().hex
            target = directory/pid/"source"
            entry = PurePosixPath(choice.entry)
            selected_files = spec.get("entry_files", {}).get(choice.entry, spec["files"])
            prefix = entry.parent
            while any(not PurePosixPath(name).is_relative_to(prefix) for name in selected_files):
                prefix = prefix.parent
            for name in selected_files:
                relative = PurePosixPath(name).relative_to(prefix)
                dest = target / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(base/"source"/name, dest)
            papers.append(dict(id=pid, name=spec["name"], kind=spec["kind"],
                               source=str(target/entry.relative_to(prefix)), root=str(target) if spec["kind"] == "tex" else None))
        return papers
    except Exception:
        if directory.is_dir():
            shutil.rmtree(directory)
        raise
