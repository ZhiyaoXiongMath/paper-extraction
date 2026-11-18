"""Source projections for the cost-optimized workflow. The stored index is immutable."""
import re
from collections import defaultdict


_ENV = re.compile(r"\\(begin|end)\s*\{(proof|thebibliography)\}")
_COMMENT = re.compile(r"(?<!\\)%[^\n]*")
_FORMAL = re.compile(r"\\begin\s*\{(?:theorem|thm|proposition|prop|lemma|lem|corollary|cor|definition|defn|example|remark)\*?\}", re.I)


def _proof_ranges(text):
    # Replace comments with spaces to keep offsets. A commented-out \end{proof}
    # cannot close a real proof. Leave unfamiliar literal environments intact.
    scan = _COMMENT.sub(lambda match: " " * len(match[0]), text)
    if r"\verb" in scan or r"\begin{verbatim" in scan or r"\iffalse" in scan:
        return []
    stack, ranges = [], []
    for mark in _ENV.finditer(scan):
        if mark[1] == "begin":
            stack.append(mark)
        elif stack and stack[-1][2] == mark[2]:
            start = stack.pop().start()
            if not stack and not _FORMAL.search(scan, start, mark.end()):
                ranges.append((start, mark.end()))
    return ranges


def projected_sources(catalog, ids, *, abridge=False):
    """Abbreviate only complete TeX proof/bibliography environments, even over spans.

    Keep the exact text, order and source ID of the surrounding statements and
    exceptions. A reader can use an abridged row's ID to fetch its verbatim text.
    PDF formulae, including scanned text, are always supplied whole.
    """
    ids = set(ids)
    masks = defaultdict(list)
    if abridge:
        files = defaultdict(list)
        for row in catalog:
            if row.get("file"):
                files[row["file"]].append(row)
        for rows in files.values():
            if not any(r["id"] in ids for r in rows):
                continue
            text = "\n".join(r["text"] for r in rows)
            spans = _proof_ranges(text)
            offset = 0
            for r in rows:
                end = offset + len(r["text"])
                if r["id"] in ids:
                    masks[r["id"]] = [(max(a, offset)-offset, min(b, end)-offset)
                                      for a, b in spans if a < end and b > offset]
                offset = end + 1
    result = []
    for row in catalog:
        if row["id"] not in ids:
            continue
        item = {k: row[k] for k in ("id", "text", "label", "section", "page") if row.get(k)}
        item.setdefault("text", "")
        if section := row.get("citation_section"):
            item["section"] = " ".join(str(v) for v in (section.get("number"), section.get("title")) if v)
        if locators := row.get("citation_statements"):
            item["statements"] = [f"{s['kind']} {s['number']}" + (f" ({s['label']})" if s.get("label") else "")
                                  for s in locators]
        for start, end in reversed(masks.get(row["id"], [])):
            item["text"] = item["text"][:start] + "\n[proof or bibliography omitted here]\n" + item["text"][end:]
            item["abridged"] = True
        result.append(item)
    return result


def writer_candidates(rows):
    """Only editorial decisions bearing on the selected mathematical content."""
    fields = ("candidate_id", "kind", "source_number", "claim", "assumptions", "dependencies",
              "evidence_ids", "needed_by", "independent_question", "b2_source_facts", "independent_role")
    return [{key: row[key] for key in fields if row.get(key)} for row in rows]


def repair_context(targets, context, records):
    """Send the full evidence for the changes, not the entire neighbour bibliography."""
    scope = {target["block_id"] for target in targets}
    neighbors = {key: value for key, value in context.items() if key not in scope}
    ids = {i for target in targets for i in target["bound_evidence_ids"] + target["required_evidence_ids"]}
    return neighbors, projected_sources(records.values(), ids)
