"""Check expanded prose and page budgets with the actual TeX and PDF toolchain."""

import json
import shutil

import pytest

from paper_extraction.profile import BuildFailure, build_case
from paper_extraction.profile_assets.tools import v13_check
from test_profile import synthetic_draft


pytestmark = pytest.mark.skipif(not shutil.which("pdflatex"), reason="requires pdfLaTeX")


def _reader_draft(result_count):
    draft = synthetic_draft()
    draft["editorial_profile"] = "reader"
    draft["paragraph"] = (
        "We study how the sign of a product depends on the sign of each factor. The elementary case of "
        "a square gives a useful point of comparison. A product with an even number of negative factors "
        "can be positive, while the results below impose positivity on the factors themselves. This "
        "lets us compare products before and after one factor is replaced by a sum of positive terms. "
        "The point is to separate the role of the sign condition from the choice of which factor to "
        "change. The same distinction applies when several factors are grouped before taking the product. "
        "This will matter in the comparison below."
    )
    draft["groups"] = [
        {"definitions": [], "theorem": (
            r"Let $x,y,z,w$ be real numbers with "
            r"\begin{conditions}&x>0,\\&y>0,\\&z>0,\\&w>0.\end{conditions} "
            rf"Then $xyzw>0$ and $xy(z+{index}w)>0$ for this positive integer ${index}$. "
            r"The sign of the product is unchanged when the third factor is replaced by the stated sum. "
            rf"\cite[Theorem {index}]{{source}}"
        )}
        for index in range(1, result_count + 1)
    ]
    draft["evidence_links"] = [
        {"block_id": field, "evidence_ids": ["p1"]}
        for field in ["paragraph", "history.0", *(f"groups.{i}.theorem" for i in range(result_count))]
    ]
    return draft


def _read_status(case):
    return json.loads((case / "build-status.json").read_text(encoding="utf-8"))


def test_reader_build_keeps_longer_opening_and_more_than_three_body_pages(tmp_path):
    source = tmp_path / "source.md"
    source.write_text("Synthetic statements about signs of products.", encoding="utf-8")
    draft = _reader_draft(18)

    case = build_case(tmp_path / "reader", source, None, draft)
    status = _read_status(case)
    opening = status["format_contract"]["blocks"][0]
    assert status["engineering_accepted"] and status["release"] == v13_check.EXPANDED_RELEASE
    assert 3 < status["main_text_pages"] <= 5 and status["references_separate"]
    assert opening["prose_words"] > 100 and opening["height_limit_units"] == 8
    assert any("introparagraph_1_word_target" in note for note in status["format_contract"]["advisories"])
    assert len(status["returncodes"]) >= 2 and set(status["returncodes"]) == {0}

    del draft["editorial_profile"]
    with pytest.raises(BuildFailure, match=r"Block 1 \(P\) exceeds 4U"):
        build_case(tmp_path / "legacy", source, None, draft)


def test_reader_still_rejects_a_document_beyond_five_body_pages(tmp_path):
    source = tmp_path / "source.md"
    source.write_text("Synthetic statements about signs of products.", encoding="utf-8")

    with pytest.raises(BuildFailure) as failure:
        build_case(tmp_path / "long", source, None, _reader_draft(35))

    status = _read_status(failure.value.case_dir)
    assert status["compiled"] and status["references_separate"]
    assert status["main_text_pages"] > status["max_body_pages"] == 5
    assert status["format_contract"]["accepted"] and not status["body_page_limit"]
