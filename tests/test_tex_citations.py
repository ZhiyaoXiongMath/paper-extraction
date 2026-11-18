import shutil

import pytest

from paper_extraction import profile
from paper_extraction.sources import SourceIndex
from test_profile import synthetic_draft


def test_tex_locator_keeps_evidence_lines_but_prints_source_section_and_labels(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    entry = root / "0705.0766v3.tex"
    entry.write_text(r"""\documentclass{article}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{definition}[theorem]{Definition}
\title[Short title]{Examples in $X$}
\author{A. Name\thanks{A grant}\and B. Author}
\date{March 2007}
\begin{document}
\section{Introduction}\label{sec:intro}
\begin{definition}\label{def:thing}
A definition whose meaning is explained here.
\end{definition}
\begin{theorem}\label{thm:front}
The initial result holds.
\end{theorem}
\input{second}
\subsection{After the included text}
\begin{theorem}\label{thm:after}
The subsequent result holds.
\end{theorem}
\end{document}""", encoding="utf8")
    (root / "second.tex").write_text(r"""\section{Further results}
\begin{theorem}\label{thm:included}
The included result holds.
\end{theorem}""", encoding="utf8")
    index = SourceIndex(entry, root, tmp_path / "work")
    records = {record['id']: record for record in index.catalog()}
    before = records['t1-2']
    assert before['file'] == entry.name and before['line'] == 8
    assert before['citation_section']['number'] == '1'
    statements = {row['label']: row['number'] for record in records.values() for row in record['citation_statements']}
    assert statements == {'def:thing': '1.1', 'thm:front': '1.2', 'thm:included': '2.1', 'thm:after': '2.2'}

    draft = synthetic_draft()
    draft['history'] = [r"A historical statement. \cite[Introduction, lines 8--14]{source}"]
    draft['shared_definitions'] = [r"A concept. \cite[\ref{def:thing}]{source}"]
    draft['groups'][0]['theorem'] = r"A source statement. \cite[\ref{thm:front}]{source}"
    draft['references'][0]['text'] = r"Bibliography copied in error. Historical details in \cite[p.1]{source}."
    draft['evidence_links'] = [dict(block_id=block, evidence_ids=['t1-2']) for block in (
        'history.0', 'shared_definitions.0', 'groups.0.theorem')]
    formatted = profile.bind_source_reference(profile.bind_formal_citations(draft, records), index.citation_metadata)
    assert formatted['evidence_links'] == draft['evidence_links']
    assert formatted['history'] == [r"A historical statement. \cite[\S~1]{source}"]
    assert formatted['shared_definitions'] == [r"A concept. \cite[Definition~1.1]{source}"]
    assert formatted['groups'][0]['theorem'] == r"A source statement. \cite[Theorem~1.2]{source}"
    bibliography = formatted['references'][0]['text']
    assert bibliography == (r"A. Name and B. Author, \emph{Examples in $X$}, arXiv:0705.0766v3, March 2007. "
                            "Historical attributions follow this manuscript.")


def test_inline_attribution_is_retained_once_without_a_duplicated_heading():
    draft = synthetic_draft()
    locator = r"\cite[Theorem~5.11]{source}"
    attribution = r"\textup{(Cegrell, " + locator + ".)} Every nonnegative measure admits a decomposition."
    draft['history'] = [attribution]
    rendered = profile.render_tex(draft)
    assert r"\begin{historical}" + "\n" + attribution in rendered
    assert rendered.count(locator) == 1
    assert profile.render_tex(draft, _legacy_citations='copy-inline').count(locator) == 2
    draft['history'] = [r"\begin{historical}[" + locator + "]" + attribution + r"\end{historical}"]
    assert profile.render_tex(draft).count(locator) == 1


def test_bare_source_does_not_replace_a_bibliography_date_with_an_inference():
    draft = synthetic_draft()
    draft['references'][0]['text'] = 'A. Author, Observed results (2014).'
    assert profile.bind_source_reference(draft, {'title': '', 'authors': []}) == draft


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="uses the v13 TeX toolchain")
def test_uncaptioned_theorem_with_inline_citation_keeps_a_flat_heading(tmp_path):
    draft = synthetic_draft()
    draft['groups'][0]['theorem'] = r"According to \cite[Theorem~3.7]{source}, every positive real has a positive square."
    source = tmp_path / "paper" / "source.tex"
    source.parent.mkdir()
    source.write_text("Every positive real has a positive square.", encoding="utf8")
    case = profile.build_case(tmp_path / "cases", source, source.parent, draft)
    assert (case / "introduction.pdf").is_file()
    assert r"\begin{theorem}" + "\n" in (case / "introduction.tex").read_text("utf8")
