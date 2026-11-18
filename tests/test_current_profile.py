"""Exercise the shipped model/TeX release with the actual offline compiler."""

import json
import re
import shutil
import subprocess

import pytest

from paper_extraction.delivery import deliver_case
from paper_extraction.exporting import standalone_tex
from paper_extraction.pdfops import page_count
from paper_extraction.profile_inputs import installed_profile, is_optimized_workflow, new_batch_profile
from paper_extraction.profile import ASSETS, BuildFailure, build_case, recover_draft_case, render_tex
from paper_extraction.profile_variants import (
    CURRENT_ASSETS, CURRENT_PROFILE, current_check, current_profile_fingerprint,
)
from paper_extraction.profile_assets.tools import v13_check
from test_reader_profile import _reader_draft


def _without_history(result_count):
    draft = _reader_draft(result_count)
    draft.update(history=[], editorial_profile=CURRENT_PROFILE)
    draft["evidence_links"] = [item for item in draft["evidence_links"] if item["block_id"] != "history.0"]
    return draft


def test_new_batch_changes_the_workflow_without_rebinding_archived_rules():
    from paper_extraction.profile_variants import PREVIOUS_PROFILE

    saved = installed_profile(CURRENT_PROFILE)
    new = new_batch_profile()
    assert new["profile_fingerprint"] == saved["profile_fingerprint"]
    assert is_optimized_workflow(new)
    assert not is_optimized_workflow(saved)
    assert "workflow_revision" not in new_batch_profile(PREVIOUS_PROFILE)


def test_current_release_files_and_legacy_isolation(tmp_path):
    info = current_check.verify(CURRENT_ASSETS)
    assert set(info["sha256"]) == {"rules.txt", "prompt.txt", "template-v13.tex", "preamble-v13.tex"}
    assert info["checker_sha256"] == current_check.digest(CURRENT_ASSETS / current_check.TOOL)
    assert info["release"] == CURRENT_PROFILE
    assert len(current_profile_fingerprint()) == 64
    assert v13_check.verify(ASSETS)["release"] == v13_check.RELEASE
    draft = _without_history(1)
    assert r"\begin{historical}" not in render_tex(draft)
    draft["editorial_profile"] = "reader"
    with pytest.raises(ValueError, match="history"):
        render_tex(draft)
    tampered = tmp_path / "current"
    shutil.copytree(CURRENT_ASSETS, tampered)
    (tampered / "prompt.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Hash mismatch: prompt.txt"):
        current_check.verify(tampered)


def test_model_budget_table_agrees_with_checker_and_tex_measurements():
    rules = (CURRENT_ASSETS / "rules.txt").read_text("utf-8")
    preamble = (CURRENT_ASSETS / "preamble-v13.tex").read_text("utf-8")
    labels = {"Historical theorem": "historical", "Current theorem": "theorem",
              "Definition": "definition", "Connecting paragraph": "introparagraph"}
    rows = [line.strip("|").split("|") for line in rules.splitlines() if line.startswith("| ")]
    table = {row[0].strip(): [field.strip() for field in row[2:]] for row in rows}
    for label, name in labels.items():
        limit = current_check.BLOCK_LIMITS[name]
        assert table[label] == [str(limit["words"]), f'{limit["units"]}U',
                                str(limit["displays"]), str(limit["rows"])]
        assert rf"\IntroMeasured{{{limit['kind']}}}{{{limit['units']}}}" in preamble
    measured = [int(amount) for amount in re.findall(r"\\IntroLimit=(\d+)\\IntroUnit", preamble)]
    assert measured == [current_check.GROUP_HEIGHT, current_check.PROSE_HEIGHT, current_check.SHARED_HEIGHT]


@pytest.mark.skipif(not shutil.which("pdflatex"), reason="requires pdfLaTeX")
def test_current_release_builds_and_exports_with_zero_history(tmp_path):
    source = tmp_path / "products.txt"
    source.write_text("Synthetic sign and product identities.", encoding="utf-8")
    draft = _without_history(21)
    work = tmp_path / "builds"
    case = build_case(work, source, None, draft)
    status = json.loads((case / "build-status.json").read_text("utf-8"))
    assert status["engineering_accepted"] and status["returncodes"] == [0, 0]
    assert status["release"] == CURRENT_PROFILE and status["input_binding_checked"]
    counts = status["format_contract"]["counts"]
    assert counts["historical"] == 0 and counts["theorem"] == 21
    assert status["main_text_pages"] == status["max_body_pages"] == 5
    assert status["references_separate"] and status["current_theorem_limit"] is None
    assert recover_draft_case(work, source, None, draft) == case
    exported = standalone_tex((case / "introduction.tex").read_bytes(), (case / "preamble-v13.tex").read_bytes)
    empty = tmp_path / "download"
    empty.mkdir()
    downloaded = empty / "products-extraction.tex"
    downloaded.write_bytes(exported)
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=batchmode", "-halt-on-error", "-no-shell-escape", downloaded.name],
                       cwd=empty, check=True, capture_output=True, timeout=60)
    assert page_count(downloaded.with_suffix(".pdf")) == status["pdf_pages"]


@pytest.mark.skipif(not shutil.which("pdflatex"), reason="requires pdfLaTeX")
def test_current_release_rejects_more_than_five_body_pages(tmp_path):
    source = tmp_path / "products.txt"
    source.write_text("Synthetic product identities.", encoding="utf-8")
    with pytest.raises(BuildFailure) as exc:
        build_case(tmp_path / "runs", source, None, _without_history(35))
    status = json.loads((exc.value.case_dir / "build-status.json").read_text("utf-8"))
    assert status["compiled"] and status["format_contract"]["accepted"]
    assert status["main_text_pages"] > 5 and "body_page_limit" in status["failed_checks"]


@pytest.mark.skipif(not shutil.which("pdflatex"), reason="requires pdfLaTeX")
def test_final_draft_uses_current_preamble_and_source_binding(tmp_path):
    source = tmp_path / "products.txt"
    source.write_text("Synthetic sign and product identities.", encoding="utf-8")
    delivered = deliver_case(tmp_path / "runs", source, None, _without_history(1))
    assert delivered.kind == "final" and delivered.compiler_verified
    receipt = json.loads((delivered.path / "delivery.json").read_text("utf-8"))
    binding = json.loads((delivered.path / "input-manifest.json").read_text("utf-8"))
    assert binding["release"] == CURRENT_PROFILE and receipt["compiler_returncodes"] == [0, 0]
    assert binding["rules_sha256"] == current_check.verify(CURRENT_ASSETS)["sha256"]
    assert receipt["files"]["preamble-v13.tex"] == binding["rules_sha256"]["preamble-v13.tex"]
