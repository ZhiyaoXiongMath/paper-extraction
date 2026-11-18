import json
from pathlib import Path
import shutil

import pytest

from paper_extraction import delivery, pdfops, profile
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


def _build_failure(tmp_path: Path, source: Path, draft: dict) -> Exception:
    with pytest.raises(profile.BuildFailure) as failed:
        profile.build_case(tmp_path / "work", source, None, draft)
    return failed.value
