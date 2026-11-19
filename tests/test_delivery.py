import json
import hashlib
from pathlib import Path
import shutil

import pytest

from paper_extraction import delivery, pdfops, profile
from paper_extraction.profile_variants import CURRENT_PROFILE
from test_profile import synthetic_draft


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="requires pdfLaTeX")
def test_final_delivery_compiles_budget_overrun_with_original_companion(tmp_path):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"An immutable input snapshot.")
    draft = synthetic_draft()
    draft["paragraph"] = "We consider positive real numbers. " * 24
    assert "exceeds" in str(_build_failure(tmp_path, source, draft))

    result = delivery.deliver_case(tmp_path / "work", source, None, draft)
    assert result.kind == "final"
    assert (result.path / "preamble-v13.tex").read_bytes() == (profile.ASSETS / "preamble-v13.tex").read_bytes()
    receipt = json.loads((result.path / "delivery.json").read_text("utf-8"))
    assert receipt["compiler_returncodes"] == [0, 0]
    assert receipt["input_binding"]["checked"]
    assert "Theorem 2" in " ".join(pdfops.read_pages(result.path / "introduction.pdf"))


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="requires pdfLaTeX")
def test_final_delivery_without_draft_can_use_extracted_claims(tmp_path):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"Input for deterministic delivery")
    excerpt = "Excerpt in place of an unwritten draft."
    result = delivery.deliver_case(tmp_path / "work", source, None,
                                   {"paragraph": excerpt, "source_excerpt": excerpt}, candidates=[
        {"claim": r"The formula $x^2>0$ holds for $x>0$.", "evidence_ids": ["p2-1"],
         "source_number": "Theorem 2", "decision": "admitted"},
    ])
    assert result.kind == "readable"
    assert "Theorem 2" in pdfops.read_pages(result.path / "introduction.pdf")[0]
    assert json.loads((result.path / "delivery.json").read_text("utf-8"))["compiler_returncodes"] == [0, 0]
    assert (result.path / "preamble-v13.tex").is_file()


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="requires pdfLaTeX")
def test_real_tex_error_uses_verified_previous_or_recompiles_literal(tmp_path):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"Another input snapshot.")
    draft = synthetic_draft()
    work = tmp_path / "work"
    prior = profile.build_case(work, source, None, draft)
    # This passes the fragment allowlist, but the missing denominator fails pdfLaTeX.
    draft["paragraph"] = r"We consider the term $\frac{1}$ here."
    result = delivery.deliver_case(work, source, None, draft, previous_case=str(prior))
    assert result.path == prior
    assert result.kind == "prior"
    assert "pdfLaTeX" in result.diagnostic

    (prior / "preamble-v13.tex").write_text("Modified companion", "utf-8")
    safe = delivery.deliver_case(work, source, None, draft, previous_case=prior)
    assert safe.kind == "readable"
    assert safe.path != prior
    assert r"\textbackslash{}frac" in (safe.path / "introduction.tex").read_text("utf-8")
    assert "frac" in pdfops.read_pages(safe.path / "introduction.pdf")[0]
    assert json.loads((safe.path / "delivery.json").read_text("utf-8"))["compiler_returncodes"] == [0, 0]


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="requires pdfLaTeX")
def test_prior_may_be_a_compiled_snapshot_archived_after_layout_rejection(tmp_path):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"An immutable input snapshot.")
    draft = synthetic_draft()
    draft["paragraph"] = "We consider " + "positive " * 44 + "numbers."
    failed = _build_failure(tmp_path, source, draft)
    status = json.loads((failed.case_dir / "build-status.json").read_text("utf-8"))
    assert status["compiled"]
    assert not (failed.case_dir / "introduction.pdf").is_file()
    draft["paragraph"] = r"We consider the term $\frac{1}$ here."
    recovered = delivery.deliver_case(tmp_path / "work", source, None, draft, previous_case=failed.case_dir)
    assert recovered.kind == "prior"
    assert recovered.path == failed.case_dir / status["diagnostic_artifacts"]["directory"]


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="requires pdfLaTeX")
def test_compile_error_restores_only_a_uniquely_bound_prior_formal_block(tmp_path):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"An immutable input snapshot.")
    previous = synthetic_draft()
    previous["evidence_links"] = [{"block_id": "groups.0.theorem", "evidence_ids": ["p2-1"]}]
    work = tmp_path / "work"
    prior = profile.build_case(work, source, None, previous)
    draft = {**previous, "paragraph": "We consider the preservation of positivity under addition.",
             "groups": [{**previous["groups"][0],
                         "theorem": r"For $x>0$, the term $\frac{1}$ is positive. \cite[Theorem 2]{source}"}]}
    result = delivery.deliver_case(work, source, None, draft, previous_case=prior, previous_draft=previous)
    assert result.kind == "repaired", result
    assert result.compiler_verified
    assert result.draft == {**draft, "groups": previous["groups"]}
    assert "preservation of positivity" in pdfops.read_pages(result.path / "introduction.pdf")[0]
    assert json.loads((result.path / "delivery.json").read_text("utf-8"))["compiler_returncodes"] == [0, 0]

    different = {**draft, "evidence_links": [{"block_id": "groups.0.theorem", "evidence_ids": ["p3-1"]}]}
    result = delivery.deliver_case(work, source, None, different, previous_case=prior, previous_draft=previous)
    assert result.kind == "prior"


def test_missing_compiler_keeps_document_and_does_not_execute_untrusted_tex(tmp_path, monkeypatch):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"Input for deterministic delivery")
    draft = synthetic_draft()
    draft["paragraph"] = r"Example: \input{outside.tex}."
    monkeypatch.setattr(delivery.shutil, "which", lambda command: None)
    result = delivery.deliver_case(tmp_path / "work", source, None, draft)
    assert result.kind == "readable"
    assert not result.compiler_verified
    assert "未找到 pdfLaTeX" in result.diagnostic
    tex = (result.path / "introduction.tex").read_text("utf-8")
    assert r"\input{outside.tex}" not in tex
    assert r"\textbackslash{}input" in tex
    assert pdfops.page_count(result.path / "introduction.pdf") == 2


def test_readable_fallback_keeps_attributions_and_bibliography_numbers(tmp_path):
    from test_heading_sources import paper_with_comparison

    source = tmp_path / "paper.pdf"
    source.write_bytes(b"Input snapshot")
    draft = paper_with_comparison()
    draft["formal_sources"] = {"groups.0.theorem": {"source": r"\cite[Lemma~5.14]{source}"}}
    titles = dict(delivery._sections(tmp_path, source, None, draft, ()))
    assert "Theorem A (Bedford and Taylor [2], as stated in [1, Theorem 5.12])" in titles
    assert "Theorem 1 ([1, Lemma 5.14])" in titles
    assert "Theorem 2 ([1, Theorem 5.15]; compare [1, Theorem 5.12])" in titles
    assert "[1] The supplied paper." in titles["References"]


def test_repair_matches_typed_heading_and_core_id_without_requiring_body_cite(tmp_path):
    from copy import deepcopy
    prior = synthetic_draft()
    path = "groups.0.theorem"
    prior.update(editorial_profile=CURRENT_PROFILE, core_bindings={path: "c2"},
                 evidence_links=[{"block_id": path, "evidence_ids": ["p2-1"]}],
                 formal_sources={path: {"source": r"\cite[Theorem 2]{source}"}})
    prior["groups"][0]["theorem"] = "Positive terms have positive sum."
    (tmp_path / "introduction.tex").write_text(profile.render_tex(prior), "utf-8")
    draft = deepcopy(prior)
    draft["groups"][0]["theorem"] = r"The term $\frac{1}$ is positive."
    blocks = []
    profile.render_tex(draft, block_map=blocks)
    line = next(b["last"] for b in blocks if b["draft_field"] == path)
    assert delivery._patch_from_prior(draft, blocks, prior, tmp_path, line) == prior
    for field, changed in (("core_bindings", "c1"), ("formal_sources", {"source": r"\cite[Theorem 3]{source}"})):
        other = deepcopy(draft)
        other[field][path] = changed
        assert delivery._patch_from_prior(other, blocks, prior, tmp_path, line) is None


def test_relations_realign_only_simple_known_rows():
    text = r"""\[
\begin{alignedat}{2}
&\det A &&{}=0\quad\text{on }D,\\
&u_Z(z) &&{}\longrightarrow 0,\\
&u_Z(z)-\log|z-Z| &&{}=O(1).
\end{alignedat}
\]"""
    result = delivery.normalize_alignment({"groups": [{"theorem": text, "source": "p7-1"}]})
    assert result["groups"][0] == {"source": "p7-1", "theorem": r"""\begin{relations}
\det A &{}=0\quad\text{on }D, \\
u_Z(z) &{}\longrightarrow 0, \\
u_Z(z)-\log|z-Z| &{}=O(1).
\end{relations}"""}
    uncertain = text.replace("O(1).", "O(1). & y.")
    assert delivery.normalize_alignment({"paragraph": uncertain})["paragraph"] == uncertain


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="requires pdfLaTeX")
def test_strict_final_records_actual_pdf_and_layout_defects(tmp_path):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"An immutable input snapshot.")
    draft = synthetic_draft()
    work = tmp_path / "work"
    good = delivery.deliver_case(work, source, None, draft, strict=True)
    assert good.compiler_verified and good.layout_check == "passed"
    draft["paragraph"] = "We consider positive real numbers. " * 24
    fallback = delivery.deliver_case(work, source, None, draft, strict=True)
    assert fallback.compiler_verified and fallback.layout_check == "failed"
    assert fallback.layout_issues and pdfops.page_count(fallback.path / "introduction.pdf") > 0
    receipt = json.loads((fallback.path / "delivery.json").read_text("utf-8"))
    assert receipt["files"]["introduction.pdf"] == delivery.v13_check.digest(fallback.path / "introduction.pdf")
    assert receipt["layout_check"] == "failed"


def test_completed_checks_require_review_of_the_published_pair(tmp_path, monkeypatch):
    from paper_extraction import agent

    monkeypatch.setattr(agent, "_render_pages", lambda _pdf: [(1, b"first"), (2, b"second")])
    same_page = hashlib.sha256(b"first").hexdigest()
    other_page = hashlib.sha256(b"different second").hexdigest()
    old_draft = {"paragraph": "Original statement."}
    revised = {"paragraph": "Revised statement."}
    case = delivery.DeliveryCase(tmp_path, "final", layout_check="passed")
    review = agent._record({"draft": old_draft, "attempt": 2}, "review", True, [])
    visuals = [
        {"stage": "visual", "page": 1, "page_sha256": same_page, "passed": True},
        {"stage": "visual", "page": 2, "page_sha256": other_page, "passed": True},
    ]
    statuses, issues = agent._completed_checks(case, revised, [review, *visuals])
    assert statuses == {"content": "not_checked", "layout": "passed", "visual": "not_checked"}
    assert not issues
    visuals[1]["page_sha256"] = hashlib.sha256(b"second").hexdigest()
    statuses, _ = agent._completed_checks(case, old_draft, [review, *visuals])
    assert all(check == "passed" for check in statuses.values())


def _build_failure(tmp_path: Path, source: Path, draft: dict) -> Exception:
    with pytest.raises(profile.BuildFailure) as failed:
        profile.build_case(tmp_path / "work", source, None, draft)
    return failed.value
