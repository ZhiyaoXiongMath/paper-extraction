"""Read-only location index for one immutable paper snapshot."""
import io
import json
import re
import threading
from pathlib import Path

import pypdfium2 as pdfium

from .tex_citations import arxiv_id, bind_tex_locations

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


def uncomment(text):
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def paper_files(source: Path, root: Path, *, strict: bool):
    """Resolve declared local source dependencies without scanning neighboring papers."""
    source, root = Path(source).resolve(), Path(root).resolve()
    ordered = {}
    graphics_dirs = []

    def visit(path):
        if path in ordered:
            return
        ordered[path] = None
        if path.suffix.lower() not in {".tex", ".sty", ".cls", ".def"}:
            return
        contents = uncomment(read_tex(path))
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
                    visit(found)
                elif cmd in _INCLUDE and strict:
                    raise ValueError(f"{path.relative_to(root)} 缺少引用文件 {name}")
        # Some source archives supply the generated bibliography as the only record.
        default_bbl = source.with_suffix(".bbl")
        if path == source and default_bbl.is_file():
            visit(default_bbl)

    visit(source)
    return list(ordered)


def read_tex(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法识别 TeX 文件编码：{path.name}")


def tex_documents(source: Path, root: Path):
    """Read the main paper and its declared support files in dependency order."""
    for path in paper_files(source, root, strict=True):
        if path.suffix.lower() in {".tex", ".bbl"}:
            yield path, read_tex(path)


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
    def __init__(self, source: Path, source_root: Path | None, work: Path):
        self.source = Path(source).resolve()
        self.root = Path(source_root or self.source.parent).resolve()
        self.cache = Path(work) / "source-index.json"
        self._items = None
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

    def _tex_entries(self):
        for number, (path, contents) in enumerate(tex_documents(self.source, self.root), 1):
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
        for i, bib in enumerate((p for p in paper_files(self.source, self.root, strict=True) if p.suffix.lower() == ".bib"), 1):
            rel = bib.relative_to(self.root).as_posix()
            lines = read_tex(bib).splitlines()
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
                sources = ((path, uncomment(text)) for path, text in tex_documents(self.source, self.root))
                self.citation_metadata = bind_tex_locations(self._items, self.source, self.root, sources)
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
