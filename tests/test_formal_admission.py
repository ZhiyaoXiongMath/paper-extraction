"""Source heading admission is shared by first drafts and later nominations."""
from paper_extraction.admission import admission_source, constrain_current_results, source_statement_options
from paper_extraction.lean_workflow import direct_admission
from paper_extraction.selection import select_contributions
from paper_extraction.sources import SourceIndex
from test_core_admission_calibration import _direct_decision


def _pdf_record(text, *, ident="p1-1", page=1):
    return {"id": ident, "page": page, "label": f"p. {page}", "text": text, "section": ""}


def test_actual_headings_ignore_running_prose_cross_references():
    record = _pdf_record(
        "Théorème 4 entraîne cette propriété.\n"
        "THÉORÈME 3 (nom du résultat). — Toute variété considérée satisfait la condition.\n"
        "Lemma 2. For each object take an auxiliary cover.\n"
        "Corollary 1.2. For every integer n at least ten a distinct application holds.\n"
        "Theorem. Let the group be noncompact."
    )
    options = source_statement_options(record)
    assert [(s["kind"], s["number"]) for s in options] == [
        ("theorem", "3"), ("lemma", "2"), ("corollary", "1.2"), ("theorem", ""),
    ]
    assert [s["id"] for s in options] == ["p1-1.l2", "p1-1.l3", "p1-1.l4", "p1-1.l5"]
    assert admission_source(record)["text"] == record["text"]


def test_tex_environment_name_does_not_override_title(tmp_path):
    main = tmp_path / "main.tex"
    main.write_text(r"""
\documentclass{article}
\usepackage{local}
\begin{document}
\input{part}
\end{document}
""", "utf-8")
    (tmp_path / "local.sty").write_text(r"""
\newtheorem{mainclaim}{Théorème}
\newtheorem{theoremish}[mainclaim]{Proposition}
\newtheorem*{special}{Corollaire}
""", "utf-8")
    (tmp_path / "part.tex").write_text(r"""
\begin{mainclaim}\label{thm:actual} Content.\end{mainclaim}
\begin{theoremish} Auxiliary statement.\end{theoremish}
\begin{special} Independent consequence.\end{special}
""", "utf-8")
    catalog = SourceIndex(main, tmp_path, tmp_path / "new", admission_statements=True).catalog()
    section = next(r for r in catalog if r["file"] == "part.tex")
    options = source_statement_options(section)
    assert [(s["kind"], s["number"]) for s in options] == [
        ("theorem", "1"), ("proposition", "2"), ("corollary", ""),
    ]
    assert options[0]["label"] == "thm:actual"
    assert len({s["id"] for s in options}) == 3
    assert not any(r.get("source_statements") for r in SourceIndex(main, tmp_path, tmp_path / "old").catalog())


def test_shared_empty_environment_uses_each_printed_heading_and_original_counter(tmp_path):
    source = tmp_path / "fano.tex"
    source.write_text(
        r"\documentclass{article}\newtheorem{theorem}{Theorem}"
        r"\renewcommand{\thetheorem}{\thesection.\arabic{theorem}}\numberwithin{theorem}{section}"
        r"\newtheorem{shared}[theorem]{}\newtheorem{case}[theorem]{}" "\n"
        r"\begin{document}\section{Geometry}" "\n"
        r"\begin{case}Unnamed introductory case.\end{case}" "\n"
        r"\begin{shared}{\bf Proposition.} Auxiliary statement.\end{shared}" "\n"
        r"\begin{shared}" "\n" r"{\bf Theorem.}\label{main-theorem} Classification.\end{shared}" "\n"
        r"\begin{shared}{\textbf{Corollary.}} A consequence.\end{shared}" "\n"
        r"\end{document}", encoding="utf-8",
    )
    options = [s for r in SourceIndex(source, None, tmp_path / "work", admission_statements=True).catalog()
               for s in source_statement_options(r)]
    assert [(s["kind"], s["number"], s.get("label")) for s in options] == [
        ("proposition", "1.2", ""), ("theorem", "1.3", "main-theorem"), ("corollary", "1.4", ""),
    ]


def test_literal_caption_aliases_and_their_locale_declarations(tmp_path):
    source = tmp_path / "captions.tex"
    source.write_text(
        r"\documentclass{article}\newtheorem{thm}{\protect\theoremname}[section]" "\n"
        r"\newtheorem{cor}[thm]{\protect\corollaryname}\newtheorem{amb}[thm]{\namedheading}" "\n"
        r"\providecommand{\theoremname}{Theorem}\providecommand{\corollaryname}{Corollary}" "\n"
        r"\addto\captionsfrench{\renewcommand{\theoremname}{Théorème}}" "\n"
        r"\providecommand{\namedheading}{Theorem}\renewcommand{\namedheading}{Proposition}" "\n"
        r"\begin{document}\section{Flow}\begin{thm}\label{main} A classification.\end{thm}" "\n"
        r"\begin{cor}An application.\end{cor}\begin{amb}An ambiguous title.\end{amb}" "\n"
        r"\end{document}", encoding="utf-8",
    )
    options = [s for r in SourceIndex(source, None, tmp_path / "work", admission_statements=True).catalog()
               for s in source_statement_options(r)]
    assert [(s["kind"], s["number"]) for s in options] == [
        ("theorem", "1.1"), ("corollary", "1.2"), ("other", "1.3"),
    ]
    assert options[0]["label"] == "main"


def test_direct_admission_excludes_non_theorem_sources_and_keeps_independent_application():
    direct, raw = _direct_decision([
        ("p1-1", "Theorem 1.1", "current", "The classification holds.", "Gives a classification."),
        ("p2-1", "Proposition 5.1", "application", "A localized estimate holds.", "An estimate."),
        ("p3-1", "Lemma 5.14", "current", "An auxiliary object exists.", "Used in the proof."),
        ("p4-1", "Corollary 1.2", "application", "The conclusion applies for every integer n at least 10.",
         "Resolves a separate dimension question."),
    ])
    records = {r["id"]: {**r, "page": i} for i, r in enumerate(raw, 1)}
    rows, draft, note = direct_admission(direct, records, strict_types=True)
    assert [r["decision"] for r in rows] == ["admitted", "excluded", "excluded", "admitted"]
    assert draft.core_bindings == {"groups.0.theorem": "c1", "groups.1.theorem": "c4"}
    assert "at least 10" in draft.groups[1].theorem and note is None


def test_same_label_supports_distinct_parts_but_duplicate_cannot_acquire_new_id():
    act = _pdf_record("Theorem 1.1. Let X be compact.\nProposition 3. Let Y be nonempty.")
    theorems = [
        ("c1", "Theorem 1.1", "Classifies X.", "X compact"),
        ("c2", "Thm. 1.1", "Classifies X.", "X compact"),
        ("c3", "Theorem 1.1(a)", "Gives a volume bound.", "X compact"),
        ("c4", "Theorem 1.1", "Refers to a different heading.", ""),
        ("c5", "Lemma 2", "Uses an old argument.", ""),
    ]
    rows = [{"candidate_id": ident, "kind": "current", "decision": "admitted", "source_number": label,
             "claim": claim, "assumptions": assumptions, "evidence_ids": [act["id"]]}
            for ident, label, claim, assumptions in theorems]
    rows[3]["source_statement_id"] = "p1-1.l2"
    constrain_current_results(rows, {act["id"]: act})
    assert [r["decision"] for r in rows] == [
        "admitted", "excluded", "admitted", "unresolved", "excluded",
    ]
    assert rows[1]["represented_by"] == "c1"


def test_missing_original_title_returns_evidence_gap_instead_of_negative_screening():
    decision, raw = _direct_decision([
        ("p1-1", "Theorem 4", "current", "Classifies the possible domains.", "Solves the main problem."),
    ])
    record = {**raw[0], "page": 1, "text": "Theorem 4 implies a related estimate."}
    rows, draft, note = direct_admission(decision, {record["id"]: record}, strict_types=True)
    assert rows[0]["decision"] == "unresolved"
    assert draft is None and note["kind"] == "insufficient_evidence"


def test_bound_citation_resolves_same_number_theorem_and_corollary_in_one_tex_span(tmp_path):
    source = tmp_path / "shrinker.tex"
    source.write_text(
        r"\documentclass{article}\newtheorem{thm}{Theorem}[section]" "\n"
        r"\begin{document}\section{Self-shrinkers}" "\n"
        r"\medskip\noindent" "\n"
        r"{\bf Theorem 1.1} {\it An interior curvature bound yields classification.}" "\n"
        r"\medskip\noindent {{\bf Corollary 1.1} {\it A strict bound implies flatness.}}" "\n"
        r"We also recall {\bf Theorem 4.1} from an earlier work." "\n"
        r"\begin{proof}" "\n" r"{\bf Theorem 5.1} is used within the proof." "\n" r"\end{proof}" "\n"
        r"\end{document}",
        encoding="utf-8",
    )
    record = next(r for r in SourceIndex(source, None, tmp_path / "work", admission_statements=True).catalog()
                  if r.get("source_statements"))
    assert [(s["kind"], s["number"]) for s in source_statement_options(record)] == [
        ("theorem", "1.1"), ("corollary", "1.1"),
    ]
    result, _ = _direct_decision([
        (record["id"], "1.1", "current", "An interior curvature bound yields classification.",
         "The equality case is classified."),
    ])
    formal = result.draft.formal_sources["groups.0.theorem"]
    for bad in (r"\cite[1.1]{source}", r"\cite[Theorem~1.2]{source}",
                r"\cite[Theorem~1.1 and Corollary~1.1]{source}",
                r"\cite[proof of Theorem~1.1]{source}"):
        formal.source = bad
        rows, draft, note = direct_admission(result, {record["id"]: record}, strict_types=True)
        assert rows[0]["decision"] == "unresolved"
        assert draft is None and note["kind"] == "insufficient_evidence"
    formal.source = r"\cite[Theorem~1.1]{source}"
    rows, draft, note = direct_admission(result, {record["id"]: record}, strict_types=True)
    assert rows[0]["source_statement_kind"] == "theorem"
    assert draft.core_bindings == {"groups.0.theorem": "c1"} and note is None


def test_scoped_nomination_cannot_reinsert_duplicate_or_promote_proposition():
    records = [
        _pdf_record("Theorem 1.1. Let X be compact."),
        _pdf_record("Proposition 4. Auxiliary construction.\nCorollary 1.2. For every integer n at least ten.",
                    ident="p2-1", page=2),
    ]
    rows = []
    for i, (label, claim, source) in enumerate([
            ("Theorem 1.1", "Classifies compact X.", "p1-1"),
            ("Thm. 1.1", "Classifies compact X.", "p1-1"),
            ("Proposition 4", "Builds auxiliary data.", "p2-1"),
            ("Corollary 1.2", "Settles the dimension bound.", "p2-1"),
    ], 1):
        row = {"kind": "current", "source_number": label, "claim": claim, "assumptions": "",
               "significance": "Checked locally", "decision": "admitted", "decision_reason": "Found in source.",
               "evidence_ids": [source], "dependencies": []}
        if i == 1:
            row["candidate_id"] = "c1"
        rows.append(row)

    class Selector:
        def structured(self, model, _, payload):
            assert payload["retained_cores"][0]["candidate_id"] == "c1"
            selected = [{"candidate_id": r["candidate_id"], "independent_question": "What changes?",
                         "source_reason": "An allegedly distinct result", "evidence_ids": r["evidence_ids"],
                         "b2_source_facts": [{"source_id": r["evidence_ids"][0], "fact": "Source framing"}]}
                        for r in payload["candidate_register"]]
            return model.model_validate({"selected": selected}, context=payload)

    result, note = select_contributions(records, rows, Selector(), current=True, lean=True, optimized=True,
                                        nomination_sources=["p1-1", "p2-1"], strict_types=True)
    assert note is None
    assert [r["decision"] for r in result] == ["admitted", "excluded", "excluded", "admitted"]


def test_whole_paper_selection_receives_b1_and_accepts_a_justified_fifth():
    records = [_pdf_record(f"Theorem {n}. An independent conclusion for case {n}.",
                           ident=f"p{n}-1", page=n) for n in range(1, 6)]
    candidates = [
        {"kind": "current", "source_number": f"Theorem {n}", "claim": f"An independent conclusion for case {n}.",
         "assumptions": "", "significance": "Answers a separate question.",
         "decision": "admitted", "decision_reason": "Found in source.", "evidence_ids": [record["id"]],
         "dependencies": []}
        for n, record in enumerate(records, 1)
    ]
    fifth_reason = "Omitting this result would remove the classification in case 5."

    class Selector:
        def structured(self, schema, instructions, packet):
            assert "**B1.** Aim to select **at most four current main results** per paper." in instructions
            assert "If one or two results suffice, select only those." in instructions
            assert "Only results explicitly labeled *Theorem* or *Corollary*" in instructions
            assert "Count historical theorems separately." in instructions
            assert "what would be lost by omitting it." in instructions
            assert "existing source_reason" in instructions
            selected = [
                {"candidate_id": row["candidate_id"], "independent_question": "What happens in this case?",
                 "source_reason": fifth_reason if row["candidate_id"] == "c5" else "Resolves this case.",
                 "evidence_ids": row["evidence_ids"],
                 "b2_source_facts": [{"source_id": row["evidence_ids"][0], "fact": "The source proves this case."}]}
                for row in packet["candidate_register"]
            ]
            return schema.model_validate({"selected": selected}, context=packet)

    result, note = select_contributions(records, candidates, Selector(), current=True, lean=True,
                                        optimized=True, strict_types=True)
    assert note is None and len(result) == 5
    assert all(row["decision"] == "admitted" for row in result)
    assert result[-1]["decision_reason"] == fifth_reason
