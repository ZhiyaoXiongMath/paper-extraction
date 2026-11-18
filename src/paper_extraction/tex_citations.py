"""Read the notation used for citations without executing an imported manuscript."""
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
import re

from .profile_assets.tools.v13_check import ValidationError, balanced


_DECLARE = re.compile(r"\\newtheorem(?P<star>\*)?\s*\{(?P<env>[\w@]+)\}\s*"
                      r"(?:\[(?P<shared>[\w@]+)\]\s*)?\{(?P<title>[A-Za-z ]+)\}"
                      r"(?:\s*\[(?P<within>[\w@]+)\])?")
_EVENT = re.compile(r"\\(?:(?P<heading>chapter|section|subsection|subsubsection)(?P<star>\*)?\s*(?=[{\[])|"
                    r"(?P<appendix>appendix)\b|(?P<env_kind>begin|end)\s*\{(?P<env>[\w*@]+)\}|"
                    r"(?P<include>input|include|subfile)\s*\{(?P<file>[^{}]+)\}|"
                    r"setcounter\s*\{(?P<counter>[\w@]+)\}\s*\{(?P<count>\d+)\})")
_HEADING_LEVELS = ("chapter", "section", "subsection", "subsubsection")
_LABEL = re.compile(r"\s*(?:\[[^\n]*?\]\s*)?\\label\s*\{([^{}]+)\}")
_FRONT_COMMAND = re.compile(r"\\(title|author|date)\*?\s*(?=[{\[])")
_AUTHOR_NOTE = re.compile(r"\\(?:thanks|footnote|thanksref|orcid|orcidlink|inst)\b")
_ARXIV_NAME = re.compile(r"(?:arxiv[._:-]?)?(\d{4}\.\d{4,5}(?:v[1-9]\d*)?)", re.I)
_RENAMED_COUNTER = re.compile(r"\\(?:renewcommand\*?|def)\s*\{?\\the([A-Za-z@]+)\b")
_COUNTER_WITHIN = re.compile(r"\\(?:numberwithin|counterwithin)\*?\s*\{([\w@]+)\}\s*\{([\w@]+)\}")


def arxiv_id(filename):
    name = Path(filename).stem
    return match[1] if (match := _ARXIV_NAME.fullmatch(name)) else ""


def _argument(text, start):
    cursor = start
    while text[cursor:cursor + 1].isspace():
        cursor += 1
    if text[cursor:cursor + 1] == "[":
        _, cursor = balanced(text, cursor, "[", "]")
    return balanced(text, cursor)


def _remove_notes(name):
    while match := _AUTHOR_NOTE.search(name):
        try:
            _, end = balanced(name, match.end())
        except ValidationError:
            return ""
        name = name[:match.start()] + name[end:]
    return " ".join(name.split()).strip()


def _front_matter(filename, text):
    title, authors, date = "", [], ""
    for match in _FRONT_COMMAND.finditer(text):
        try:
            value, _ = _argument(text, match.end())
        except ValidationError:
            continue
        if match[1] == "title" and not title:
            title = " ".join(value.split())
        if match[1] == "author":
            authors.extend(name for part in re.split(r"\\and\b", value) if (name := _remove_notes(part)))
        if match[1] == "date" and not date and r"\today" not in value:
            date = " ".join(value.split())
    return {"title": title, "authors": authors, "date": date, "arxiv": arxiv_id(filename)}


def tex_citation_catalog(source, root, documents):
    """Return front matter and section/statement ranges keyed by original relative file.

    Number common LaTeX counters in manuscript order, including literal includes.
    Counter formats redefined by the author are cited through their sections.
    """
    docs = dict(documents)
    entry = docs[source]
    front = _front_matter(source.name, entry)
    declarations = {}
    changed, within = set(), {}
    for text in docs.values():
        declarations.update((m["env"], m) for m in _DECLARE.finditer(text) if not m["star"])
        changed.update(_RENAMED_COUNTER.findall(text))
        within.update(_COUNTER_WITHIN.findall(text))
    counters = defaultdict(int)
    headings = [None] * len(_HEADING_LEVELS)
    by_file = {}
    repeated = set()
    current_section = None
    appendix = False
    file_stack = []

    def walk(path):
        nonlocal current_section, appendix
        rel = path.relative_to(root).as_posix()
        if rel in by_file:
            repeated.add(rel)
            return
        text = docs[path]
        newlines = [mark.start() for mark in re.finditer("\n", text)]
        spans, statements, opened = [], [], []
        by_file[rel] = spans, statements
        file_stack.append(path)
        spans.append((1, current_section))
        offset = text.find(r"\begin{document}")
        for match in _EVENT.finditer(text, max(0, offset)):
            line = bisect_left(newlines, match.start()) + 1
            if match["include"]:
                target = Path(match["file"].strip())
                if not target.suffix:
                    target = target.with_suffix(".tex")
                for parent in (path.parent, source.parent, root):
                    resolved = (parent / target).resolve()
                    if resolved in docs and resolved not in file_stack:
                        walk(resolved)
                        spans.append((line + 1, current_section))
                        break
            elif match["appendix"]:
                appendix = True
                for heading in _HEADING_LEVELS:
                    counters[heading] = 0
                headings[:] = [None] * len(headings)
            elif match["counter"]:
                counters[match["counter"]] = int(match["count"])
            elif heading := match["heading"]:
                try:
                    title, end = _argument(text, match.end())
                except ValidationError:
                    continue
                level = _HEADING_LEVELS.index(heading)
                headings[level:] = [None] * (len(headings) - level)
                number = ""
                if not match["star"] and heading not in changed:
                    counters[heading] += 1
                    for name, env in declarations.items():
                        if within.get(name, env["within"]) == heading:
                            counters[name] = 0
                    for child in _HEADING_LEVELS[level + 1:]:
                        counters[child] = 0
                    base = counters[heading]
                    headings[level] = chr(64 + base) if appendix and base <= 26 and not any(headings[:level]) else str(base)
                    number = ".".join(n for n in headings if n)
                label = _LABEL.match(text, end)
                current_section = {"title": " ".join(title.split()), "number": number,
                                   "label": label[1] if label else ""}
                spans.append((line, current_section))
            elif match["env_kind"] == "begin" and (declaration := declarations.get(match["env"])):
                counter = declaration["shared"] or declaration["env"]
                base = declarations.get(counter)
                owner = within.get(counter, base["within"] if base else None)
                if base is None or counter in changed or declaration["env"] in changed:
                    continue
                if owner and (owner not in _HEADING_LEVELS or not headings[_HEADING_LEVELS.index(owner)]):
                    continue
                counters[counter] += 1
                prefix = ".".join(n for n in headings[:_HEADING_LEVELS.index(owner) + 1] if n) if owner else ""
                number = ".".join(filter(None, (prefix, str(counters[counter]))))
                label = _LABEL.match(text, match.end())
                opened.append((match["env"], line, {"kind": declaration["title"], "number": number,
                                                       "label": label[1] if label else ""}))
            elif match["env_kind"] == "end" and opened and opened[-1][0] == match["env"]:
                _, first, item = opened.pop()
                statements.append((first, line, item))
        file_stack.pop()

    walk(source)
    return front, {rel: value for rel, value in by_file.items() if rel not in repeated}


def bind_tex_locations(items, source, root, documents):
    front, sections = tex_citation_catalog(source, root, documents)
    for item in items:
        locators = sections.get(item.get("file"))
        if not locators:
            continue
        start, end = item["line"], item["end_line"]
        headings, statements = locators
        applicable = [heading for line, heading in headings if line <= start]
        applicable = applicable[-1:] + [heading for line, heading in headings if start < line <= end]
        if len(applicable) == 1 and applicable[0]:
            item["citation_section"] = applicable[0]
        item["citation_statements"] = [{**row, "contains_span": first <= start and end <= last}
                                        for first, last, row in statements if first <= end and start <= last]
    return front
