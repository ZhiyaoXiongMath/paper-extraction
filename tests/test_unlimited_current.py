import json
import shutil

import pytest

from paper_extraction.models import Draft
from paper_extraction.profile import ASSETS, build_case, recover_draft_case, render_tex, v13_check
from paper_extraction.profile_variants import COUNT_OVERRIDE, UNLIMITED_CURRENT
from paper_extraction.scope_reconciliation import saved_count_exception, scope_failure_reason
from paper_extraction.selection import SELECTION_VERSION, WholePaperSelection, select_contributions
from test_profile import synthetic_draft
from test_whole_selection import finding, pick


def test_model_selection_can_pass_seven_current_results_with_no_second_count_review():
    records = [{"id": f"p{i}", "text": f"Original statement {i}."} for i in range(1, 9)]
    candidates = [finding("historical" if i == 1 else "current", f"Theorem {i}", p["text"], p["id"])
                  for i, p in enumerate(records, 1)]

    class Model:
        def structured(self, schema, prompt, packet):
            original = (ASSETS / "intro-rules-v13.txt").read_text("utf-8")
            assert schema is WholePaperSelection and prompt.startswith(original) and COUNT_OVERRIDE in prompt
            return schema(selected=[pick(f"c{i}", p["id"]) for i, p in enumerate(records, 1)])

    selected = select_contributions(records, candidates, Model(), unlimited_current=True)
    assert len([r for r in selected if r["kind"] == "current" and r["decision"] == "admitted"]) == 7
    assert scope_failure_reason(selected, unlimited_current=True) == ""
    assert saved_count_exception({"status": "failed", "selection_version": SELECTION_VERSION,
                                  "candidates": selected, "reason": "v13 范围例外：旧版四项限制"}, SELECTION_VERSION)


@pytest.mark.skipif(not shutil.which("pdflatex"), reason="requires the v13 TeX toolchain")
def test_seven_theorems_build_with_bound_count_variant_and_keep_standard_limits(tmp_path):
    source = tmp_path / "source.md"
    source.write_text("Synthetic source for exercising theorem counts.", encoding="utf-8")
    draft = synthetic_draft()
    draft["groups"] = [{"definitions": [], "theorem": rf"If $x>0$, then ${n}x>0$. \cite[Theorem {n}]{{source}}"}
                       for n in range(1, 8)]
    draft["evidence_links"] = [{"block_id": k, "evidence_ids": ["p1"]}
                               for k in ("paragraph", "history.0", *(f"groups.{i}.theorem" for i in range(7)))]
    assert len(Draft.model_validate(draft).groups) == 7

    case = build_case(tmp_path / "work", source, None, draft)
    state = json.loads((case / "build-status.json").read_text("utf-8"))
    manifest = json.loads((case / "input-manifest.json").read_text("utf-8"))
    assert state["engineering_accepted"] and state["current_theorem_limit"] is None
    assert state["release"] == manifest["release"] == UNLIMITED_CURRENT
    assert state["three_page_limit"] and state["references_separate"]
    assert recover_draft_case(tmp_path / "work", source, None, draft) == case

    base_profile = v13_check.verify(ASSETS)
    assert base_profile["release"] == v13_check.RELEASE
    assert base_profile["sha256"]["preamble-v13.tex"] == "c363b47a067f7206c1c7ac56e77a913ff39f20f206d77760cf8ab02fad902590"
    invalid_case = v13_check.allocate(ASSETS, source, tmp_path / "standard")
    (invalid_case / "introduction.tex").write_text(render_tex(draft).replace(UNLIMITED_CURRENT, v13_check.RELEASE), encoding="utf-8")
    rejected = v13_check.build(ASSETS, invalid_case)
    assert not rejected["engineering_accepted"]
    assert "More than four current theorems" in (invalid_case / "introduction.log").read_text("utf-8", errors="replace")
