from copy import deepcopy

import pytest

from paper_extraction.citation_layout import reading_blocks, visual_page_context
from paper_extraction.profile import render_tex
from paper_extraction.profile_inputs import WORKFLOW_REVISION
from paper_extraction.profile_variants import CURRENT_PROFILE
from paper_extraction.reference_patch import clean_comparison_locators
from test_profile import synthetic_draft


def test_saved_source_roles_define_the_same_reading_pdf_and_visual_order():
    draft = synthetic_draft()
    source = r"\cite[Corollary~42.6]{source}"
    independent = r"\cite[\S~8]{source}"
    prior = r"\cite[Theorem~Q]{source}"
    draft.update(editorial_profile=CURRENT_PROFILE, workflow_revision=WORKFLOW_REVISION, history=[],
                 shared_definitions=[], formal_sources={
                     "groups.0.theorem": {"source": source},
                     "groups.1.theorem": {"source": source, "comparison": prior}},
                 groups=[
                     {"definitions": [], "paragraph": "Fix the coordinate system. " + independent,
                      "paragraph_position": "before",
                      "theorem": r"The domain has two boundaries. \begin{conditions}& x=2\\ & y=3\end{conditions}"},
                     {"definitions": [], "paragraph": "This permits curved boundaries. " + prior,
                      "paragraph_position": "before", "theorem": "A curved domain has the same property."},
                 ])
    original = deepcopy(draft)
    blocks = reading_blocks(draft)
    fields = [b["id"] for b in blocks]
    assert fields == ["paragraph", "groups.0.paragraph", "groups.0.theorem", "groups.1.theorem", "groups.1.paragraph"]
    tex = render_tex(draft)
    assert tex.index("Fix the coordinate") < tex.index("two boundaries") < tex.index("curved domain") < tex.index("This permits")
    assert blocks[1]["text"].endswith(independent) and tex.count(independent) == 1
    assert prior in blocks[3]["caption"] and tex.count(prior) == 1
    assert source == blocks[2]["caption"]
    packet = visual_page_context(blocks, "Theorem 1. The domain has two boundaries.\nTheorem 2. A curved domain.")
    assert packet["paragraph_positions"] == {"groups.0.paragraph": "before", "groups.1.paragraph": "after"}
    assert packet["display_environments"] == {"groups.0.theorem": ["conditions"]}
    assert list(packet["field_index"]) == fields[1:]
    assert visual_page_context(blocks, "References\n1. Original work.")["display_environments"] == {}
    assert original == draft
    # Snapshots from an earlier workflow retain their original reading and PDF order.
    draft.pop("workflow_revision")
    assert render_tex(draft).index("two boundaries") < render_tex(draft).index("Fix the coordinate")


@pytest.mark.parametrize("separator", [" and ", "; ", ", ", ", and "])
def test_compound_locators_respect_sources_across_files_and_preserve_math(separator):
    good, bad = r"\eqref{eq:kept}", r"Equation~(\ref{eq:absent})"
    mixed = r"\cite[Corollary~83.1" + separator + bad + separator + good + "]{source}"
    expected = r"\cite[Corollary~83.1" + separator + good + "]{source}"
    draft = {"groups": [{"paragraph": mixed, "theorem": mixed}],
             "formal_sources": {"groups.0.theorem": {"source": mixed, "comparison": mixed}},
             "evidence_links": [{"block_id": "groups.0.paragraph", "evidence_ids": ["passage"]}]}
    records = {
        "passage": {"id": "passage", "file": "statements.tex", "line": 4, "text": "The corollary.",
                    "source_statements": [{"id": "s1", "label": "c", "kind": "corollary",
                                           "number": "83.1", "unnumbered": False}]},
        "eqs1": {"id": "eqs1", "file": "extra.tex", "line": 1, "text": r"\label"},
        "eqs2": {"id": "eqs2", "file": "extra.tex", "line": 2, "text": r"{eq:kept} % \label{eq:absent}"},
    }
    old = deepcopy(draft)
    cleaned = clean_comparison_locators(draft, records, structured=True)
    assert cleaned["groups"][0]["paragraph"] == cleaned["formal_sources"]["groups.0.theorem"]["comparison"] == expected
    assert cleaned["groups"][0]["theorem"] == cleaned["formal_sources"]["groups.0.theorem"]["source"] == mixed
    assert old == draft
    uncertain = [
        mixed.replace("{source}", "{other}"),
        mixed.replace("83.1", "99.7"),
        mixed.replace("Corollary~83.1", "Corollary~83.1, with an assumption at infinity"),
        r"\cite[Equation~(\ref{eq:absent})]{source}",
        r"\cite[Equation~(\ref{eq:absent}); Corollary~83.1]{source}",
        r"\cite[Corollary~83.1; (\ref{unknown:object})]{source}",
    ]
    draft.pop("formal_sources")
    for text in uncertain:
        draft["groups"][0]["paragraph"] = text
        assert clean_comparison_locators(draft, records, structured=True) is draft
