"""Regressions for the web/TeX source attribution discrepancies in paper 0158."""
from copy import deepcopy

import pytest

from paper_extraction.citation_types import FormalSource
from paper_extraction.profile import reader_draft_view, render_tex
from paper_extraction.profile_variants import CURRENT_PROFILE, PREVIOUS_PROFILE, current_check
from test_profile import synthetic_draft


def paper_with_comparison():
    draft = synthetic_draft()
    draft.update(editorial_profile=CURRENT_PROFILE, shared_definitions=[],
                 references=[{"key": "source", "text": "The supplied paper."},
                             {"key": "bt", "text": "Bedford and Taylor."}])
    draft["history"] = [r"An earlier comparison holds. (Bedford and Taylor \cite{bt}, "
                        r"as stated in \cite[Theorem~5.12]{source}.)"]
    draft["groups"] = [
        {"theorem": r"There is a unique solution for these data. \cite[Lemma~5.14]{source}",
         "definitions": [r"Let $I=[0,1]$. \cite[Definition~5.13]{source}"]},
        {"theorem": r"The measure comparison implies $v\leq u$. \cite[Theorem~5.15]{source}",
         "definitions": [], "paragraph_position": "before",
         "paragraph": r"This extends the bounded comparison. \cite[Theorems~5.12 and 5.15]{source}"},
    ]
    return draft


def test_single_shared_projection_for_statement_history_and_different_comparison():
    draft = paper_with_comparison()
    original = deepcopy(draft)
    tex = render_tex(draft)
    blocks = {b["id"]: b for b in reader_draft_view(draft)["reader_blocks"]}
    assert original == draft
    assert blocks["history.0"]["text"] == "An earlier comparison holds."
    assert blocks["history.0"]["caption"] == r"Bedford and Taylor \cite{bt}, as stated in \cite[Theorem~5.12]{source}"
    assert blocks["groups.0.definitions.0"]["text"] == r"Let $I=[0,1]$."
    assert blocks["groups.0.theorem"]["caption"] == r"\cite[Lemma~5.14]{source}"
    assert blocks["groups.1.theorem"]["caption"] == r"\cite[Theorem~5.15]{source}; compare \cite[Theorem~5.12]{source}"
    assert blocks["groups.1.paragraph"]["text"] == "This extends the bounded comparison."
    assert list(blocks).index("groups.1.theorem") < list(blocks).index("groups.1.paragraph")
    for row in blocks.values():
        if row["caption"]:
            assert "[{" + row["caption"] + "}]\n" + row["text"] in tex
    assert tex.index("implies") < tex.index("extends")

    # Old export keeps the served bytes/layout; only the read view is normalized.
    draft["editorial_profile"] = PREVIOUS_PROFILE
    old_tex = render_tex(draft)
    assert r"\begin{historical}" + "\n" + original["history"][0] in old_tex
    assert old_tex.index("extends") < old_tex.index("implies")
    assert reader_draft_view(draft)["reader_blocks"] == list(blocks.values())


def test_explicit_sources_do_not_relocate_math_or_overwrite_comparison_owner():
    draft = paper_with_comparison()
    draft["groups"][0]["theorem"] = r"For every $x\in [0,1]$, the conclusion holds."
    draft["formal_sources"] = {
        "groups.0.theorem": FormalSource(source=r"\cite[Lemma~5.14]{source}").model_dump(),
        "history.0": FormalSource(prior=r"Bedford and Taylor \cite{bt}",
            reporting="as stated in", source=r"\cite[Theorem~5.12]{source}").model_dump(),
    }
    tex = render_tex(draft)
    assert r"\begin{theorem}[{\cite[Lemma~5.14]{source}}]" in tex
    assert r"For every $x\in [0,1]$, the conclusion holds." in tex
    assert tex.count("as stated in") == 1
    assert tex.count("Bedford and Taylor") == 2  # includes bibliography


def test_checker_requires_heading_and_rejects_duplicate_source_in_formal_body(tmp_path):
    draft = paper_with_comparison()
    tex = render_tex(draft)
    primary = r"\cite[Lemma~5.14]{source}"
    needle = r"\begin{theorem}[{" + primary + "}]"
    corrupt = tex.replace(needle, r"\begin{theorem}" + "\n" + primary, 1)
    report = current_check.format_contract(corrupt, "", tmp_path / "not_compiled.pdf")
    assert "theorem_1_missing_heading_source" in report["errors"]
    assert "theorem_1_body_source_outside_heading" in report["errors"]


def test_disagreement_between_heading_metadata_and_statement_cannot_silently_drop_a_locator():
    draft = paper_with_comparison()
    draft["formal_sources"] = {"groups.0.theorem": {"source": r"\cite[Lemma~5.16]{source}"}}
    with pytest.raises(ValueError, match=r"groups.0.theorem.*differs from formal_sources"):
        render_tex(draft)
    saved = next(row for row in reader_draft_view(draft)["reader_blocks"] if row["id"] == "groups.0.theorem")
    assert "5.16" in saved["caption"] and "5.14" in saved["text"]
