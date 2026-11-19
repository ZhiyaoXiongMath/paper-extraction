"""Read the notation used for citations without executing an imported manuscript."""
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path
import re

from .admission import printed_statement_kind
from .profile_assets.tools.v13_check import ValidationError, balanced


_DECLARE = re.compile(r"\\newtheorem(?P<star>\*)?\s*\{(?P<env>[\w@]+)\}\s*"
                      r"(?:\[(?P<shared>[\w@]+)\]\s*)?\{(?P<title>[A-Za-z ]+)\}"
                      r"(?:\s*\[(?P<within>[\w@]+)\])?")
_DECLARE_SOURCE = re.compile(_DECLARE.pattern.replace(r"(?P<env>[\w@]+)", r"(?P<env>[\w*@]+)")
                             .replace("[A-Za-z ]+", r"(?:[^{}]|\{[^{}]*\})*"))
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
_BOLD_SWITCH = re.compile(r"^\s*\\(?:bfseries|bf)(?![A-Za-z])")
_BOLD_ARGUMENT = re.compile(r"^\s*\\(?:textbf|textsc)(?![A-Za-z])")
_DISPLAYED_NAME = re.compile(r"^\s*([^\W\d_]+)(?=\s|[.:(\\]|$)")
_MANUAL_HEAD = re.compile(r"(?m)^[ \t]*(?:\\(?:noindent|smallskip|medskip|bigskip|par)\b[ \t]*)*"
                          r"\{*?(?=\{\s*\\(?:bf|bfseries)\b|\\(?:textbf|textsc)\b)")
_MANUAL_TITLE = re.compile(r"\s*(?P<name>[^\W\d_]+)(?:\s+(?P<number>[A-Z]?(?:\.?\d+(?:\.\d+)*[a-zA-Z]?)|[A-Z]))?\.?\s*")
_NOT_STATEMENTS = re.compile(r"\\begin\{(thebibliography|proof|comment|verbatim\*?)\}.*?\\end\{\1\}", re.S)
_ITALIC_BODY = re.compile(r"\s*(?:\{\s*\\(?:it|itshape)\b|\\textit\b)")
_CAPTION_REF = re.compile(r"\s*(?:\\protect\s*)?\\([A-Za-z@]+)\s*")
_CAPTION_DECLARATION = re.compile(r"\\(?:(?:providecommand|newcommand|renewcommand)\*?\s*(?:"
                                  r"\{\s*\\(?P<group>[A-Za-z@]+)\s*\}|\\(?P<bare>[A-Za-z@]+))"
                                  r"(?:\s*\[0\])?|(?:g?def)\s*\\(?P<definition>[A-Za-z@]+))")
_TEX_TOKEN = re.compile(r"\\(?P<control>[A-Za-z@]+)\*?|\\[^\r\n]|%")
_LITERAL_ENVIRONMENT = re.compile(
    r"\s*\{(verbatim\*?|Verbatim\*?|lstlisting|minted|comment|filecontents\*?)\}"
)
_DOCUMENT_START = re.compile(r"\\begin\s*\{\s*document\s*\}")
_MACRO_DECLARATION = re.compile(
    r"\\(?:(?P<command>newcommand|providecommand|renewcommand|DeclareRobustCommand)\*?\s*"
    r"(?:\{\s*\\(?P<braced>[A-Za-z@]+)\s*\}|\\(?P<bare>[A-Za-z@]+))"
    r"|(?P<primitive>[gex]?def|let)\s*\\(?P<defined>[A-Za-z@]+))"
)
_ENVIRONMENT_WRAPPER = re.compile(r"\s*\\(begin|end)\s*\{([\w*@]+)\}\s*")
_SCOPE_TOKEN = re.compile(r"\\([A-Za-z@]+)|\\[^\r\n]|[{}]")


def _mask_tex(text, ranges):
    chunks, start = [], 0
    for left, right in ranges:
        if left < start:
            continue
        chunks.extend((text[start:left], re.sub(r"[^\r\n]", " ", text[left:right])))
        start = right
    return "".join((*chunks, text[start:]))


def tex_syntax(text):
    """Mask comments and literal material without moving decoded source positions."""
    ranges = []
    cursor = 0
    while match := _TEX_TOKEN.search(text, cursor):
        cursor = match.end()
        control = match["control"]
        if match[0] == "%":
            newline = text.find("\n", cursor)
            cursor = len(text) if newline < 0 else newline
        elif control == "verb" and cursor < len(text) and not text[cursor].isspace():
            delimiter = text[cursor]
            newline = text.find("\n", cursor)
            line_end = len(text) if newline < 0 else newline
            delimiter_end = text.find(delimiter, cursor + 1, line_end)
            cursor = line_end if delimiter_end < 0 else delimiter_end + 1
        elif control == "begin" and (env := _LITERAL_ENVIRONMENT.match(text, cursor)):
            closer = re.compile(r"\\end\s*\{" + re.escape(env[1]) + r"\}").search(text, env.end())
            cursor = closer.end() if closer else len(text)
        else:
            continue
        ranges.append((match.start(), cursor))
    return _mask_tex(text, ranges)


def top_level_tex_commands(text):
    """Report controls outside groups and unresolved TeX conditionals."""
    depth = conditional = 0
    defining_conditional = False
    for token in _SCOPE_TOKEN.finditer(text):
        if token[0] == "{":
            depth += 1
        elif token[0] == "}":
            depth = max(0, depth - 1)
        elif not depth and (name := token[1]):
            if not conditional:
                yield token.start()
            if defining_conditional:
                defining_conditional = False
            elif name == "newif":
                defining_conditional = True
            elif name == "fi":
                conditional = max(0, conditional - 1)
            elif name.startswith("if") and name != "ifthenelse":
                conditional += 1


def _macro_definitions(text):
    """Read command bodies and identify literal wrappers without evaluating them."""
    top_level = set(top_level_tex_commands(text))
    declarations = []
    ranges = []
    for match in _MACRO_DECLARATION.finditer(text):
        if ranges and match.start() < ranges[-1][1]:
            continue
        name = match["braced"] or match["bare"] or match["defined"]
        cursor = match.end()
        if match["primitive"] == "let":
            declarations.append((name, match.start(), None))
            continue
        options = []
        while text[cursor:cursor + 1].isspace():
            cursor += 1
        if match["primitive"] and text[cursor:cursor + 1] != "{":
            parameters = re.match(r"(?:#\d+)+\s*", text[cursor:])
            if parameters:
                options.append(parameters[0])
                cursor += parameters.end()
        try:
            while text[cursor:cursor + 1] == "[":
                option, cursor = balanced(text, cursor, "[", "]")
                options.append(option.strip())
                while text[cursor:cursor + 1].isspace():
                    cursor += 1
            body, end = balanced(text, cursor)
        except ValidationError:
            declarations.append((name, match.start(), None))
            continue
        ranges.append((match.start(), end))
        wrapper = _ENVIRONMENT_WRAPPER.fullmatch(body) if (
            match.start() in top_level and options in ([], ["0"])) else None
        declarations.append((name, match.start(), wrapper.groups() if wrapper else None))
    return declarations, _mask_tex(text, ranges)


def _resolve_include(filename, path, source, root, documents, stack=()):
    target = Path(filename.strip())
    if not target.suffix:
        target = target.with_suffix(".tex")
    for parent in (path.parent, source.parent, root):
        resolved = (parent / target).resolve()
        if resolved in documents and resolved not in stack:
            return resolved
    return None


def _environment_aliases(source, root, documents, support, theorem_declarations):
    scanned = {path: _macro_definitions(text) for path, text in documents.items()}
    preambles = {}

    def preamble(path):
        if path in preambles:
            return
        _, scan = scanned[path]
        begin = _DOCUMENT_START.search(scan)
        limit = begin.start() if begin else len(scan)
        preambles[path] = limit
        visible = set(top_level_tex_commands(scan[:limit]))
        for match in _EVENT.finditer(scan, 0, limit):
            if match["include"] and match.start() in visible:
                included = _resolve_include(match["file"], path, source, root, documents)
                if included:
                    preamble(included)

    preamble(source)
    definitions = defaultdict(list)
    for path, (macros, _) in scanned.items():
        for name, offset, wrapper in macros:
            definitions[name].append(wrapper if offset < preambles.get(path, 0) else None)
    for text in support:
        macros, _ = _macro_definitions(text)
        for name, _, wrapper in macros:
            definitions[name].append(wrapper)
    reserved = {*theorem_declarations, *_HEADING_LEVELS, "appendix", "include", "input", "subfile",
                "begin", "end", "setcounter"}
    aliases = {name: choices[0] for name, choices in definitions.items()
               if name not in reserved and choices[0] and choices[0][1] in theorem_declarations
               and all(choice == choices[0] for choice in choices)}
    return aliases, {path: scan for path, (_, scan) in scanned.items()}


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


def _formatted_heading(text, start):
    """Read the source text of a bold caption, allowing a literal wrapper."""
    try:
        direct = _BOLD_ARGUMENT.match(text[start:])
        if direct:
            word, end = balanced(text, start + direct.end())
        else:
            group, end = balanced(text, start)
            if switch := _BOLD_SWITCH.match(group):
                word = group[switch.end():]
            elif argument := _BOLD_ARGUMENT.match(group):
                word, _ = balanced(group, argument.end())
            else:
                return "", start
    except ValidationError:
        return "", start
    return word, end


def _manual_statements(text, start, statements, newlines):
    """Recognize old TeX papers that lay out numbered statements by hand."""
    protected = [(m.start(), m.end()) for m in _NOT_STATEMENTS.finditer(text, start)]
    for found in _MANUAL_HEAD.finditer(text, start):
        offset = found.end()
        line = bisect_left(newlines, offset) + 1
        if any(first <= line <= last for first, last, _ in statements) or any(
                first <= offset < last for first, last in protected):
            continue
        caption, end = _formatted_heading(text, offset)
        title = _MANUAL_TITLE.fullmatch(caption)
        if not title or not printed_statement_kind(title["name"]):
            continue
        number = title["number"] or ""
        if not number and not _ITALIC_BODY.match(text, end):
            continue
        label = _LABEL.match(text, end)
        yield line, line, {"kind": title["name"], "number": number, "label": label[1] if label else "",
                           "offset": offset, "unnumbered": not number}


def _counter_display_overrides(texts, declarations, within):
    """Recognize literal Arabic numbering the existing counter simulation can reproduce."""
    overrides = {}
    for text in texts:
        for match in _RENAMED_COUNTER.finditer(text):
            name = match[1]
            declaration = declarations.get(name)
            owner = within.get(name, declaration["within"] if declaration else None)
            expected = (fr"\the{owner}." if owner else "") + fr"\arabic{{{name}}}"
            cursor = match.end()
            if text[cursor:cursor + 1] == "}":
                cursor += 1
            try:
                body, _ = balanced(text, cursor)
            except ValidationError:
                continue
            overrides[name] = re.sub(r"\s+", "", body) == expected
    return {name for name, recognized in overrides.items() if recognized}


def _literal_captions(texts, declarations):
    """Resolve source-defined caption aliases to one unambiguous printed type."""
    references = {m[1] for d in declarations.values() if (m := _CAPTION_REF.fullmatch(d["title"]))}
    if not references:
        return {}
    values = defaultdict(list)
    for text in texts:
        for match in _CAPTION_DECLARATION.finditer(text):
            name = match["group"] or match["bare"] or match["definition"]
            if name not in references:
                continue
            try:
                caption, _ = balanced(text, match.end())
            except ValidationError:
                caption = ""
            literal = " ".join(caption.split())
            values[name].append((printed_statement_kind(literal), literal))
    # Locale declarations may spell the same formal type differently. Any
    # competing type, parameterized definition or arbitrary command stays unknown.
    return {name: choices[0][1] for name, choices in values.items()
            if choices[0][0] and all(kind == choices[0][0] for kind, _ in choices)}


def tex_citation_catalog(source, root, documents, *, admission_statements=False, declaration_files=(),
                         environment_aliases=False):
    """Return front matter and section/statement ranges keyed by original relative file.

    Number common LaTeX counters in manuscript order, including literal includes.
    Counter formats redefined by the author are cited through their sections.
    """
    docs = dict(documents)
    if environment_aliases:
        docs = {path: tex_syntax(text) for path, text in docs.items()}
        declaration_files = tuple(tex_syntax(text) for text in declaration_files)
    entry = docs[source]
    front = _front_matter(source.name, entry)
    declarations = {}
    changed, within = set(), {}
    declaration_pattern = _DECLARE_SOURCE if admission_statements else _DECLARE
    declaration_sources = (*docs.values(), *declaration_files)
    for text in declaration_sources:
        declarations.update((m["env"], m) for m in declaration_pattern.finditer(text)
                            if admission_statements or not m["star"])
        changed.update(_RENAMED_COUNTER.findall(text))
        within.update(_COUNTER_WITHIN.findall(text))
    if admission_statements:
        changed.difference_update(_counter_display_overrides(declaration_sources, declarations, within))
    captions = _literal_captions(declaration_sources, declarations) if admission_statements else {}
    if environment_aliases:
        aliases, scans = _environment_aliases(source, root, docs, declaration_files, declarations)
    else:
        aliases, scans = {}, docs
    event_pattern = _EVENT
    if aliases:
        names = "|".join(map(re.escape, aliases))
        event_pattern = re.compile(_EVENT.pattern + r"|\\(?P<alias>" + names + r")(?![A-Za-z@])")
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
        scan = scans[path]
        alias_controls = set(top_level_tex_commands(scan)) if aliases else set()
        newlines = [mark.start() for mark in re.finditer("\n", text)]
        spans, statements, opened = [], [], []
        by_file[rel] = spans, statements
        file_stack.append(path)
        spans.append((1, current_section))
        document = _DOCUMENT_START.search(scan) if environment_aliases else None
        offset = document.start() if document else text.find(r"\begin{document}")
        for match in event_pattern.finditer(scan, max(0, offset)):
            line = bisect_left(newlines, match.start()) + 1
            env_kind, env_name = match["env_kind"], match["env"]
            if aliases and (alias := match["alias"]):
                if match.start() not in alias_controls:
                    continue
                env_kind, env_name = aliases[alias]
            if match["include"]:
                if resolved := _resolve_include(match["file"], path, source, root, docs, file_stack):
                    walk(resolved)
                    spans.append((line + 1, current_section))
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
            elif env_kind == "begin" and (declaration := declarations.get(env_name)):
                counter = declaration["shared"] or declaration["env"]
                base = declarations.get(counter)
                owner = within.get(counter, base["within"] if base else None)
                counted = (not declaration["star"] and base is not None and
                           counter not in changed and declaration["env"] not in changed and
                           (not owner or (owner in _HEADING_LEVELS and headings[_HEADING_LEVELS.index(owner)])))
                if not counted and not admission_statements:
                    continue
                number = ""
                if counted:
                    counters[counter] += 1
                    prefix = ".".join(n for n in headings[:_HEADING_LEVELS.index(owner) + 1] if n) if owner else ""
                    number = ".".join(filter(None, (prefix, str(counters[counter]))))
                label_start = match.end()
                title = declaration["title"]
                if admission_statements:
                    title = " ".join(title.split())
                    if reference := _CAPTION_REF.fullmatch(title):
                        title = captions.get(reference[1], title)
                    if not title:
                        formatted, label_start = _formatted_heading(text, label_start)
                        title = (match_title[1] if (match_title := _DISPLAYED_NAME.match(formatted)) else "")
                label = _LABEL.match(text, label_start)
                metadata = {"kind": title, "number": number, "label": label[1] if label else ""}
                if admission_statements:
                    metadata.update(offset=match.start(), unnumbered=bool(declaration["star"]))
                opened.append((env_name, line, metadata))
            elif env_kind == "end" and opened and opened[-1][0] == env_name:
                _, first, item = opened.pop()
                statements.append((first, line, item))
        if admission_statements:
            statements.extend(list(_manual_statements(scan, max(0, offset), statements, newlines)))
        file_stack.pop()

    walk(source)
    return front, {rel: value for rel, value in by_file.items() if rel not in repeated}


def bind_tex_locations(items, source, root, documents, *, admission_statements=False, declaration_files=(),
                       environment_aliases=False):
    front, sections = tex_citation_catalog(source, root, documents,
                                           admission_statements=admission_statements,
                                           declaration_files=declaration_files,
                                           environment_aliases=environment_aliases)
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
        item["citation_statements"] = [{"kind": row["kind"], "label": row["label"], "number": row["number"],
                                         "contains_span": first <= start and end <= last}
                                        for first, last, row in statements
                                        if row["number"] and row["kind"] and first <= end and start <= last]
        if admission_statements:
            # Only the opening span asserts the original heading. Its ID does
            # not depend on which subsequent chunks overlap this statement.
            prefix = item["id"].split("-", 1)[0]
            item["source_statements"] = [{**row, "id": f"{prefix}.s{row['offset']}", "line": first, "end_line": last}
                                         for first, last, row in statements if row["kind"] and start <= first <= end]
    return front
