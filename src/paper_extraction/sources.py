"""Read-only location index for one immutable paper snapshot."""
from codecs import BOM_UTF8
import io
import json
import re
import threading
from pathlib import Path

import pypdfium2 as pdfium

from .tex_citations import arxiv_id, bind_tex_locations, tex_syntax, top_level_tex_commands

_PDF_LOCK = threading.RLock()
_RESOURCE = re.compile(
    r"\\(?P<cmd>input|include|subfile|bibliography|addbibresource|bibliographystyle|"
    r"includegraphics|includepdf|usepackage|RequirePackage|documentclass|LoadClass)"
    r"\*?\s*(?:\[[^\]]*\]\s*)*\{(?P<names>[^{}]+)\}"
)
_GRAPHICSPATH = re.compile(r"\\graphicspath\s*\{((?:\s*\{[^{}]+\}\s*)+)\}")
_HEADING = re.compile(r"\\(?:sub)*section\*?\{([^{}]+)\}")
_EXTENSIONS = {
    "input": (".tex",), "include": (".tex",), "subfile": (".tex",),
    "bibliography": (".bib",), "addbibresource": (".bib",), "bibliographystyle": (".bst",),
    "includegraphics": (".pdf", ".png", ".jpg", ".jpeg", ".eps"),
    "includepdf": (".pdf",), "usepackage": (".sty",), "RequirePackage": (".sty",),
    "documentclass": (".cls",), "LoadClass": (".cls",),
}
_INCLUDE = {"input", "include", "subfile"}
_DEPENDENCY_FILES = {".tex", ".sty", ".cls", ".def"}
_INDEXED_FILES = {".tex", ".bib", ".bbl"}
_INPUT_ENCODING = re.compile(
    r"\\(?:usepackage|RequirePackage)\s*\[([^\[\]{}]+)\]\s*\{\s*inputenc\s*\}"
)
_DOCUMENT_START = re.compile(r"\\begin\s*\{\s*document\s*\}")
_TEX_ENCODINGS = {
    "utf8": "utf-8-sig", "utf8x": "utf-8-sig",
    "latin1": "iso8859_1", "latin2": "iso8859_2", "latin9": "iso8859_15",
    "ansinew": "cp1252", "applemac": "mac_roman", "cp1250": "cp1250",
    "cp1252": "cp1252",
}


def uncomment(text):
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def _decode_tex(path: Path, use_declared_encoding: bool,
                inherited_encoding: str | None = None) -> tuple[str, str | None]:
    data = path.read_bytes()
    declared = None
    if use_declared_encoding:
        if data.startswith(BOM_UTF8):
            declared = "utf-8-sig"
        elif b"inputenc" in data:
            # Latin-1 preserves each byte, so ASCII TeX syntax can identify the
            # encoding before any manuscript text is decoded.
            syntax = tex_syntax(data.decode("latin-1"))
            active = set(top_level_tex_commands(syntax))
            body = next((m for m in _DOCUMENT_START.finditer(syntax) if m.start() in active), None)
            limit = body.start() if body else len(syntax)
            for match in _INPUT_ENCODING.finditer(syntax, 0, limit):
                if match.start() not in active:
                    continue
                for option in match[1].split(","):
                    if encoding := _TEX_ENCODINGS.get(option.strip().lower()):
                        declared = encoding
    for encoding in dict.fromkeys((declared, inherited_encoding, "utf-8-sig", "gb18030", "cp1252")):
        if not encoding:
            continue
        try:
            return data.decode(encoding), encoding if encoding in (declared, inherited_encoding) else None
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法识别 TeX 文件编码：{path.name}")


def _paper_sources(source: Path, root: Path, *, strict: bool, use_declared_encoding: bool):
    """Read only the textual inputs of the dependency closure, in source order."""
    source, root = Path(source).resolve(), Path(root).resolve()
    ordered = {}
    graphics_dirs = []

    def visit(path, inherited_encoding=None):
        if path in ordered:
            return
        ordered[path] = None
        suffix = path.suffix.lower()
        if suffix not in _DEPENDENCY_FILES | _INDEXED_FILES:
            return
        # Preview only needs the names of external bibliographies.
        if suffix not in _DEPENDENCY_FILES and not strict:
            return
        text, inherited = _decode_tex(path, use_declared_encoding, inherited_encoding)
        ordered[path] = text
        if suffix not in _DEPENDENCY_FILES:
            return
        contents = tex_syntax(text) if use_declared_encoding else uncomment(text)
        for dirs in _GRAPHICSPATH.finditer(contents):
            for name in re.findall(r"\{([^{}]+)\}", dirs[1]):
                graphics_dirs.extend([source.parent / name, path.parent / name])
        for ref in _RESOURCE.finditer(contents):
            cmd = ref["cmd"]
            names = ref["names"].split(",") if cmd in {
                "usepackage", "RequirePackage", "bibliography", "bibliographystyle"
            } else [ref["names"]]
            for raw in names:
                name = raw.strip()
                if not name or "\\" in name or "#" in name:
                    continue
                folders = [path.parent, source.parent, root]
                if cmd in {"includegraphics", "includepdf"}:
                    folders.extend(graphics_dirs)
                extensions = ("",) if Path(name).suffix else ("", *_EXTENSIONS[cmd])
                found = None
                for directory in folders:
                    for extension in extensions:
                        candidate = (directory / (name + extension)).resolve()
                        if candidate.is_relative_to(root) and candidate.is_file():
                            found = candidate
                            break
                    if found is not None:
                        break
                if found is not None:
                    visit(found, inherited)
                elif cmd in _INCLUDE and strict:
                    raise ValueError(f"{path.relative_to(root)} 缺少引用文件 {name}")
        # Some source archives supply the generated bibliography as the only record.
        default_bbl = source.with_suffix(".bbl")
        if path == source and default_bbl.is_file():
            visit(default_bbl, inherited)

    visit(source)
    return ordered


def paper_files(source: Path, root: Path, *, strict: bool, use_declared_encoding=False):
    """Resolve declared local source dependencies without scanning neighboring papers."""
    return list(_paper_sources(source, root, strict=strict, use_declared_encoding=use_declared_encoding))


def read_tex(path: Path, *, use_declared_encoding=False, inherited_encoding=None) -> str:
    return _decode_tex(path, use_declared_encoding, inherited_encoding)[0]


def tex_documents(source: Path, root: Path, *, use_declared_encoding=False):
    """Read the main paper and its declared support files in dependency order."""
    for path, text in _paper_sources(
        source, root, strict=True, use_declared_encoding=use_declared_encoding,
    ).items():
        if path.suffix.lower() in {".tex", ".bbl"}:
            yield path, text


def page_png(path: Path, page: int, region=None) -> bytes:
    with _PDF_LOCK, pdfium.PdfDocument(path) as doc:
        if not 1 <= page <= len(doc):
            raise ValueError("页码超出论文范围")
        pdfpage = doc[page - 1]
        try:
            bitmap = pdfpage.render(scale=2)
            try:
                img = bitmap.to_pil().copy()
            finally:
                bitmap.close()
            if region is not None:
                if len(region) != 4:
                    raise ValueError("region 需要四个归一化坐标")
                left, top, right, bottom = region
                if not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
                    raise ValueError("region 超出页面范围")
                img = img.crop((int(left * img.width), int(top * img.height),
                                max(int(left * img.width) + 1, round(right * img.width)),
                                max(int(top * img.height) + 1, round(bottom * img.height))))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        finally:
            pdfpage.close()


class SourceIndex:
    def __init__(self, source: Path, source_root: Path | None, work: Path, *, admission_statements=False,
                 improved_tex=None):
        self.source = Path(source).resolve()
        self.root = Path(source_root or self.source.parent).resolve()
        improved_cache = Path(work) / "source-index-v2.json"
        self.improved_tex = self.source.suffix.lower() == ".tex" and (
            improved_cache.is_file() if improved_tex is None else improved_tex)
        self.cache = improved_cache if self.improved_tex else Path(work) / "source-index.json"
        self._items = None
        self._source_files = None
        self.admission_statements = admission_statements or self.improved_tex
        self.citation_metadata = {}

    def _pdf_entries(self):
        with _PDF_LOCK, pdfium.PdfDocument(self.source) as doc:
            for n in range(1, len(doc) + 1):
                page = doc[n - 1]
                tp = page.get_textpage()
                try:
                    page_text = tp.get_text_bounded().replace("\r\n", "\n")
                finally:
                    tp.close()
                    page.close()
                chunks = [page_text[i:i+3500] for i in range(0, len(page_text), 3500)] or [""]
                for k, chunk in enumerate(chunks, 1):
                    yield {"id": f"p{n}-{k}", "page": n, "label": f"p. {n}", "text": chunk,
                           "section": "", "file": self.source.name}

    def _tex_sources(self):
        if self._source_files is None:
            self._source_files = _paper_sources(self.source, self.root, strict=True,
                                                use_declared_encoding=self.improved_tex)
        return self._source_files

    def _tex_entries(self):
        files = self._tex_sources()
        documents = ((path, text) for path, text in files.items() if path.suffix.lower() in {".tex", ".bbl"})
        for number, (path, contents) in enumerate(documents, 1):
            rel = path.relative_to(self.root).as_posix()
            section = ""
            part, start, size, seq = [], 1, 0, 0
            for line_num, line in enumerate(contents.splitlines(), 1):
                heading = _HEADING.search(line)
                if part and (size + len(line) > 3500 or heading):
                    seq += 1
                    yield self._tex_item(number, seq, rel, start, line_num-1, part, section)
                    part, start, size = [], line_num, 0
                if heading:
                    section = heading.group(1)
                part.append(line)
                size += len(line) + 1
            if part:
                yield self._tex_item(number, seq+1, rel, start, start+len(part)-1, part, section)
        # External bibliography entries are useful evidence for historical attributions.
        bibliographies = ((path, text) for path, text in files.items() if path.suffix.lower() == ".bib")
        for i, (bib, contents) in enumerate(bibliographies, 1):
            rel = bib.relative_to(self.root).as_posix()
            lines = contents.splitlines()
            for start in range(0, len(lines), 25):
                yield self._tex_item(f"bib{i}", start+1, rel, start+1,
                                     min(start+25, len(lines)), lines[start:start+25], "Bibliography")

    @staticmethod
    def _tex_item(number, seq, file, start, end, lines, section):
        return {"id": f"t{number}-{seq}", "file": file, "line": start, "end_line": end,
                "section": section, "label": f"{file} L{start}–{end}", "text": "\n".join(lines)}

    def catalog(self) -> list[dict]:
        if self._items is None:
            if self.cache.is_file():
                self._items = json.loads(self.cache.read_text("utf-8"))
            else:
                items = list(self._pdf_entries() if self.source.suffix.lower() == ".pdf" else self._tex_entries())
                if not items:
                    raise ValueError("来源文件没有可读取的页或行")
                self.cache.parent.mkdir(parents=True, exist_ok=True)
                from tempfile import NamedTemporaryFile
                with NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.cache.parent, delete=False) as file:
                    json.dump(items, file, ensure_ascii=False)
                Path(file.name).replace(self.cache)
                self._items = items
            for item in self._items:
                if item.get("page") and not item["text"].strip():
                    sidecar = self.cache.parent / "page-readings" / (item["id"] + ".json")
                    if sidecar.is_file():
                        reading = json.loads(sidecar.read_text("utf8"))
                        if reading.get("page") == item["page"] and reading.get("transcribed_from_image"):
                            item.update(reading)
            if self.source.suffix.lower() == ".tex":
                files = self._tex_sources()
                sources = ((path, text if self.improved_tex else uncomment(text)) for path, text in files.items()
                           if path.suffix.lower() in {".tex", ".bbl"})
                declarations = (text if self.improved_tex else uncomment(text) for path, text in files.items()
                                if path.suffix.lower() in {".sty", ".cls", ".def"}) if self.admission_statements else ()
                self.citation_metadata = bind_tex_locations(self._items, self.source, self.root, sources,
                                                             admission_statements=self.admission_statements,
                                                             declaration_files=declarations,
                                                             environment_aliases=self.improved_tex)
            elif identifier := arxiv_id(self.source.name):
                self.citation_metadata = {"arxiv": identifier}
        return self._items

    def save_transcription(self, location: str, page: int, text: str):
        """Save a machine reading alongside the immutable source page and location."""
        item = next((r for r in self.catalog() if r["id"] == location and r.get("page") == page), None)
        if item is None or item["text"].strip() or not isinstance(text, str) or not text.strip():
            return
        folder = self.cache.parent / "page-readings"
        folder.mkdir(exist_ok=True)
        record = {"page": page, "text": text, "transcribed_from_image": True,
                  "evidence_type": "machine_transcription"}
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(mode="w", encoding="utf8", dir=folder, delete=False) as stream:
            json.dump(record, stream, ensure_ascii=False)
        Path(stream.name).replace(folder / (location + ".json"))
        item.update(record)


    def read(self, location: str) -> dict:
        for item in self.catalog():
            if item["id"] == location:
                return dict(item)
        raise ValueError(f"未找到证据位置 {location}")

    def search(self, query: str) -> list[dict]:
        query = query.strip().casefold()
        if not query:
            return []
        words = list(dict.fromkeys(query.split()))
        ranked = []
        for item in self.catalog():
            text = item["text"].casefold()
            score = 5 * text.count(query) + sum(text.count(w) for w in words)
            if score:
                ranked.append((score, item))
        ranked.sort(key=lambda pair: -pair[0])
        return [item for _, item in ranked[:8]]

    def image(self, page: int, region=None) -> bytes:
        if self.source.suffix.lower() != ".pdf":
            raise ValueError("该来源为 TeX；请使用 read_span 查看原始数学公式")
        return page_png(self.source, page, region)
