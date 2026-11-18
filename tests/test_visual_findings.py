from paper_extraction.models import Review
from paper_extraction.visual_findings import actionable_page_review


# The exact historical observations from the 1010 pilot. In both rounds all
# text remained visible; only the start/end positions on adjacent lines differed.
_WRAPS = [
    "On page 2, Definition 3 breaks after “normal vector,” leaving its symbol ξ at the start of the next line.",
    "On page 2, Theorem 3 breaks after “holomorphic sectional curvature,” leaving μ and the sentence-ending period at the start of the next line.",
    "On page 2, Theorem 3 breaks the dimension hypothesis across lines: “of dimension” ends one line and “n > 1” starts the next.",
    "On page 2, the paragraph following Theorem 3 leaves the single word “form.” on its last line, splitting the phrase “second fundamental form.”",
]


def test_ordinary_continuations_pass_without_a_second_draft():
    original = Review(passed=False, issues=_WRAPS)
    assert actionable_page_review(original) == Review(passed=True, issues=[])
    assert original.issues == _WRAPS


def test_real_or_ambiguous_defects_remain_actionable_even_with_a_prose_wrap():
    hard = [
        _WRAPS[0] + " Its subscript is clipped.",
        "The displayed formula wraps across lines; its ending is outside the margin.",
        "The bibliography begins on the final body page.",
        "The printed expression has a separated superscript.",
    ]
    original = Review(passed=False, issues=_WRAPS + hard,
                      locations=[{"draft_field": "history.0", "issue": issue} for issue in _WRAPS + hard])
    final = actionable_page_review(original)
    assert not final.passed and final.issues == hard
    assert [loc.issue for loc in final.locations] == hard
