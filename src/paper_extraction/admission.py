"""Resolve current results to formal headings in the same source snapshot."""
import re
import unicodedata
import logging
from difflib import SequenceMatcher


_TITLES = {
    "theorem": "theorem", "theoreme": "theorem", "satz": "theorem", "teorema": "theorem",
    "定理": "theorem",
    "corollary": "corollary", "corollaire": "corollary", "corolario": "corollary",
    "corollario": "corollary", "korollar": "corollary", "推论": "corollary",
    "proposition": "proposition", "proposicion": "proposition", "proposizione": "proposition",
    "proposicao": "proposition", "命题": "proposition",
    "lemma": "lemma", "lemme": "lemma", "lema": "lemma", "引理": "lemma",
    "remark": "remark", "remarque": "remark", "example": "example", "exemple": "example",
    "definition": "definition", "conjecture": "conjecture",
}
_ELIGIBLE_TYPES = {"theorem", "corollary"}
_CURRENT_ROLES = {"current", "application"}
_DISPLAY_TITLE = (r"Theorem|Théorème|Theoreme|Satz|Teorema|定理|Corollary|Corollaire|Corolário|Corolario|"
                  r"Corollario|Korollar|推论|Proposition|Proposición|Proposizione|Proposição|命题|"
                  r"Lemma|Lemme|Lema|引理|Remark|Remarque|Example|Exemple|Definition|Définition|Conjecture")
_TITLE = re.compile(rf"^(?P<title>{_DISPLAY_TITLE})(?![^\W\d_])", re.I)
_NUMBER = r"(?:[A-Z]\.\d+|[0-9]+)(?:[.·]\d+)*[a-zA-Z]?|(?:[IVXLCDM]{2,}|[A-Z])"
_LOCATOR_NUMBER = re.compile(rf"(?P<number>{_NUMBER})(?![\w·]|\.\d)")
_MODEL_TITLE = re.compile(rf"^(?P<title>{_DISPLAY_TITLE}|Thm\.?|Cor\.?|Prop\.?|Lem\.?)(?![^\W\d_])", re.I)
_CITATION_HEADING = re.compile(rf"(?<!\w)(?P<title>{_DISPLAY_TITLE})\s+(?P<number>{_NUMBER})(?![\w·]|\.\d)", re.I)
_LABEL_REF = re.compile(r"\\(?:ref|eqref|autoref|cref)\s*\{([^{}]+)\}")
_CAPTION_END = re.compile(r"(?:\s*\([^()\n]+\))?\s*(?:\.(?!\d)|:|[—–]|-(?=\s|[A-Z]))")
_CLAIM_START = re.compile(r"^(?:Let|Suppose|Assume|For|If|Then|Every|Any|There|The|Given|Whenever|"
                          r"Soit|Soient|Pour|Si|Tout|Toute|Sei|Seien|Ist|Sia|Se|Sea)\b")


def _normalized(value):
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def printed_statement_kind(title):
    # The source may spell printed Latin accents using plain TeX commands.
    title = re.sub(r"\\['`^~\x22=.uvHckbdtr]\s*\{?([A-Za-z])\}?", r"\1", title)
    title = title.replace("{", "").replace("}", "")
    title = re.sub(r"(?i)^(?:main|principal)\s+", "", title)
    title = title.strip()
    parts = title.split(maxsplit=1)
    if len(parts) == 2 and _LOCATOR_NUMBER.fullmatch(parts[1]):
        title = parts[0]
    normalized = unicodedata.normalize("NFKD", title.casefold())
    name = "".join(c for c in normalized if not unicodedata.combining(c))
    return _TITLES.get(name)


def _pdf_statement(line):
    """Recognize headings at the start of text lines, with a visible boundary."""
    line = _normalized(line).replace("\x02", "")
    match = _TITLE.match(line)
    if not match or match["title"][0].islower():
        return None
    title, tail = match["title"], line[match.end():].strip()
    number = _LOCATOR_NUMBER.match(tail)
    if number:
        rest = tail[number.end():].strip()
        # A running-text cross-reference can begin the line. It is not the heading.
        if rest and not _CAPTION_END.match(rest) and not _CLAIM_START.match(rest):
            return None
        return printed_statement_kind(title), number["number"].replace("·", "."), f"{title} {number['number']}"
    if not tail or _CAPTION_END.match(tail) or (tail.startswith("(") and tail.endswith(")")):
        return printed_statement_kind(title), "", title
    return None


def source_statement_options(record):
    """Typed, source-local statement identities usable in model requests and checks."""
    if record.get("page"):
        return [{"id": f"{record['id']}.l{line_no}", "kind": kind, "number": number, "heading": heading,
                 "unnumbered": not number}
                for line_no, line in enumerate(record["text"].splitlines(), 1)
                if (parts := _pdf_statement(line))
                for kind, number, heading in (parts,)]
    return [{"id": statement["id"], "kind": printed_statement_kind(statement["kind"]) or "other",
             "number": statement["number"], "heading": statement["kind"], "label": statement["label"],
             "unnumbered": statement["unnumbered"]}
            for statement in record.get("source_statements", ())]


def admission_source(record):
    """Add a compact heading directory without changing the source text or location."""
    statements = source_statement_options(record)
    result = {k: v for k, v in record.items() if k != "source_statements"}
    if statements:
        result["formal_statement_options"] = [
            {"id": s["id"], "kind": s["kind"],
             "heading": " ".join(filter(None, (s["heading"], s["number"] if record.get("line") else ""))),
             **({"tex_label": s["label"]} if s.get("label") else {})}
            for s in statements]
    return result


def _candidate_locator(value):
    text = _normalized(value)
    match = _MODEL_TITLE.match(text)
    kind = None
    if match:
        title = match["title"].rstrip(".").casefold()
        # Common abbreviations in the model's locator are not source headings.
        kind = printed_statement_kind(title) or {
            "thm": "theorem", "cor": "corollary", "prop": "proposition", "lem": "lemma",
        }.get(title)
        text = text[match.end():].strip()
    number = _LOCATOR_NUMBER.match(text)
    label_ref = _LABEL_REF.search(text)
    label = label_ref[1] if label_ref else text
    return kind, number["number"].replace("·", ".") if number else "", label


def _cited_heading(citation):
    cite = re.fullmatch(r"\\cite\s*\[([^\[\]]+)\]\s*\{[^{}]+\}", citation.strip())
    if not cite:
        return None
    locator = _normalized(cite[1].replace("~", " "))
    if re.search(r"(?i)\b(?:proof of|before|after|following|compare)\b", locator):
        return None
    references = list(_CITATION_HEADING.finditer(locator))
    if len(references) != 1:
        return None
    ref = references[0]
    return printed_statement_kind(ref["title"]), ref["number"].replace("·", ".")


def current_statement(row, records, heading_citation=""):
    """Resolve exactly one compatible original heading from the cited evidence."""
    # Require the assigned evidence to contain the actual heading. Setting and
    # proof passages supplied when finding leads do not justify a new locator.
    ids = row.get("selection_evidence_ids") or row.get("evidence_ids", ())
    options = {s["id"]: s for i in ids if i in records for s in source_statement_options(records[i])}
    kind, number, label = _candidate_locator(row.get("source_number"))
    manual_id = row.get("source_statement_id")
    if manual_id:
        option = options.get(manual_id)
        if option and (not kind or option["kind"] == kind) and (
                not number or option["number"] == number or (not option["number"] and not option["unnumbered"])):
            return option
        return None
    if not kind and not number and not label:
        return None
    matches = [s for s in options.values() if (not kind or s["kind"] == kind)
               and ((number and s["number"] == number) or
                    (label and s.get("label") == label) or
                    (kind and not number and not label and not s["number"]))]
    if len(matches) > 1 and not kind and (hint := _cited_heading(heading_citation)):
        # A bound formal citation can distinguish Theorem 1 from Corollary 1
        # on the same page. It cannot supply a new heading or number.
        matches = [s for s in matches if (s["kind"], s["number"]) == hint]
    return matches[0] if len(matches) == 1 else None


def nomination_heading_label(value):
    """A chapter or attribution may precede the single explicit numbered heading."""
    headings = list(_CITATION_HEADING.finditer(_normalized(value).replace("~", " ")))
    return headings[0].group(0) if len(headings) == 1 else value


def scoped_image_readings(active, preserved, records):
    """Find a saved page reading for the same formal statement in this paper's catalog."""
    # A scoped selector may receive a second observation of an already photographed
    # result. A shared page number alone identifies neither the claim nor the crop.
    wanted = set()
    for row in active:
        statement = current_statement(row, records)
        if not statement or not statement["number"]:
            continue
        crops = {(r["source_id"], r["page"], tuple(r.get("region") or ()))
                 for r in row.get("source_readings", [])}
        for loc in row["evidence_ids"]:
            page = records.get(loc, {}).get("page")
            if page and statement["id"].startswith(loc + ".l"):
                bound = [crop for crop in crops if crop[:2] == (loc, page)]
                wanted.update((statement["id"], *crop) for crop in bound or [(loc, page, ())])
    if not wanted:
        return []
    result = []
    for row in preserved:
        if not row.get("source_readings"):
            continue
        statement = current_statement(row, records)
        if not statement:
            continue
        for reading in row["source_readings"]:
            key = (statement["id"], reading["source_id"], reading["page"], tuple(reading.get("region") or ()))
            if key not in wanted:
                continue
            # Legacy candidates sometimes carry every observation on their page.
            # Require the reading itself to identify the same numbered source heading.
            lines = [line.strip().lstrip("# ").replace("**", "") for line in reading["transcription"].splitlines()
                     if line.strip()]
            if not lines or not _CITATION_HEADING.match(_normalized(lines[0])):
                continue
            captions = {(printed_statement_kind(hit["title"]), hit["number"].replace("·", "."))
                        for line in lines if (hit := _CITATION_HEADING.match(_normalized(line)))}
            if captions == {(statement["kind"], statement["number"])} and reading not in result:
                result.append(reading)
    return result


def damaged_heading_page(row, records):
    """Find a unique possible OCR-damaged heading for an image lookup, never for admission."""
    kind, number, _ = _candidate_locator(row.get("source_number"))
    if not number or kind and kind not in _ELIGIBLE_TYPES:
        return None
    # A printed title starts its own line and ends with punctuation. Running
    # prose such as “Proof of Theorem 4” cannot identify the assertion's page.
    header = re.compile(r"^\s*([^\W\d_]{4,})\s+" + re.escape(number) + r"(?:\.(?!\d)|\s*[—–:])")
    words = [word for word, role in _TITLES.items() if role in _ELIGIBLE_TYPES and (not kind or role == kind)]
    matches = set()
    for loc in row.get("selection_evidence_ids") or row.get("evidence_ids", ()):
        rec = records.get(loc, {})
        if not rec.get("page"):
            continue
        for line in rec["text"].splitlines():
            match = header.match(line)
            if not match or not match[1][0].isupper():
                continue
            word = "".join(c for c in unicodedata.normalize("NFKD", match[1].casefold()) if not unicodedata.combining(c))
            if word not in _TITLES and any(SequenceMatcher(None, word, title).ratio() >= .75 for title in words):
                matches.add(loc)
    return next(iter(matches)) if len(matches) == 1 else None


def verify_damaged_headings(rows, records, resolve):
    """Verify previously selected central results once, at the suspected statement page."""
    from .models import SourceCheck
    checks = {}
    for row in rows:
        if row["kind"] not in _CURRENT_ROLES or row["decision"] != "admitted" or current_statement(row, records):
            continue
        page = damaged_heading_page(row, records)
        if page:
            checks.setdefault(page, set()).add(_candidate_locator(row.get("source_number"))[1])
    if not checks:
        return []
    requests = [SourceCheck(source_id=page, read_image=True, question=(
        "Transcribe the exact printed heading (type and number) of statement " + ", ".join(sorted(numbers))
        + " on this page, on its own line exactly as printed. Follow with its opening words."
          " If the caption is absent or illegible, say so; a proof or mention is not its heading."))
        for page, numbers in checks.items()]
    from openai import APIError
    from httpx import HTTPError
    try:
        return resolve(requests, {loc: records[loc] for loc in checks})["source_images"]
    except (APIError, HTTPError, OSError) as exc:
        # This optional caption check must not lose a usable manuscript and the
        # independently verified results when a vision provider is unavailable.
        logging.getLogger(__name__).warning("Caption image unavailable (%s)", type(exc).__name__)
        return []


def constrain_current_results(rows, records, *, heading_citations=None):
    """Exclude ineligible headings and collapse identical assertions under one source."""
    seen = {}
    for row in rows:
        if row["decision"] != "admitted" or row["kind"] not in _CURRENT_ROLES:
            continue
        reported_kind, _, _ = _candidate_locator(row.get("source_number"))
        statement = current_statement(row, records, (heading_citations or {}).get(row["candidate_id"], ""))
        if (reported_kind is not None and reported_kind not in _ELIGIBLE_TYPES) or (
                statement and statement["kind"] not in _ELIGIBLE_TYPES):
            row.update(decision="excluded", decision_reason="原文为命题、引理或其他材料，作为支持证据保留。")
            continue
        if not statement:
            row.update(decision="unresolved", scope_central_open=True,
                       decision_reason="尚无法在所引原文定位此精确断言的 Theorem 或 Corollary 标题。")
            continue
        # Parts of one source theorem can have materially different hypotheses
        # and conclusions. Merge exact repeats of the assertion only.
        claim = " ".join(row["claim"].split())
        assumptions = " ".join((row.get("assumptions") or "").split())
        identity = (statement["id"], claim, assumptions, row.get("nomination_scope"))
        if duplicate := seen.get(identity):
            row.update(decision="excluded", represented_by=duplicate,
                       decision_reason=f"与 {duplicate} 同源且结论与条件相同。")
            continue
        seen[identity] = row["candidate_id"]
        row.update(source_statement_id=statement["id"], source_statement_kind=statement["kind"])
