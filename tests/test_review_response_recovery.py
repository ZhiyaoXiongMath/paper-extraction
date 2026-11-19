"""Replay the 0147 reviewer contract and exercise its local-delivery edge."""
import json
from pathlib import Path
import shutil
import time

import pytest
from fastapi.testclient import TestClient

from paper_extraction import agent, worker
from paper_extraction.api import create_app
from paper_extraction.lean_workflow import EvidenceReview
from paper_extraction.models import AdmissionReview
from paper_extraction.profile_inputs import new_batch_profile
from paper_extraction.profile_variants import TOKEN_PROFILE
from test_agent_json import adapter
from test_lean_workflow import APPLICATION, MAIN, Replay


_PASS = {"route": "pass", "issues": [], "evidence_ids": []}
_UNKNOWN = {"comment": "The reviewer has not supplied a verdict."}
_EXHAUSTED = ("Invalid EvidenceReview response after three attempts: 1: finish=stop, output_tokens=2317, document: value_error | "
              "2: finish=stop, output_tokens=1865, document: value_error | 3: finish=stop, output_tokens=2378, document: value_error")


def test_flat_verdict_and_provisional_verdict_with_real_source_question():
    assert EvidenceReview.model_validate(_PASS).verdict.route == "pass"
    assert AdmissionReview.model_validate({"verdict": _PASS, "source_checks": []}).route == "pass"
    query = {"source_id": "p3-1", "question": "What is the sign of the exponent?", "read_image": True}
    choice = EvidenceReview.model_validate({"verdict": _PASS, "source_checks": [query]},
                                           context={"passages": [{"id": "p3-1"}]})
    assert choice.verdict is None and choice.source_checks[0].read_image
    requests = []
    model = adapter([(json.dumps(document), "stop", 10) for document in (_UNKNOWN, _PASS)], requests)
    assert model.structured(EvidenceReview, "Review this paper.", {}).verdict.route == "pass"
    retry = json.loads(requests[1]["messages"][-1]["content"])
    assert retry["previous_complete_json"] == _UNKNOWN
    assert "under verdict" in retry["correction"] and "source_checks" in retry["correction"]


@pytest.mark.skipif(not shutil.which("pdflatex"), reason="requires local compiler")
@pytest.mark.parametrize("failure", ["invalid", "checkpoint", "transport"])
def test_saved_source_bound_draft_review_failure_paths(failure, tmp_path, monkeypatch):
    source = tmp_path / "source" / "352.tex"
    source.parent.mkdir()
    source.write_text(r"\documentclass{article}\begin{document}" + "\n" + r"\section{Theorem 1.1}"
                      + "\n" + MAIN + "\n" + r"\section{Corollary 1.2}" + "\n" + APPLICATION
                      + "\n" + r"\end{document}", "utf-8")
    work = tmp_path / "run"
    replay = Replay("first")
    wire_requests = []
    offline = adapter([(json.dumps(_UNKNOWN), "stop", 35)] * 3, wire_requests)
    original = replay.structured
    def reply(schema, instructions, packet, image=None):
        if schema is EvidenceReview:
            if failure == "checkpoint":
                # This was the exception saved before exhaustion gained its own typed class.
                raise ValueError(_EXHAUSTED)
            if failure == "transport":
                raise ConnectionError("provider unreachable during content review")
            return offline.structured(schema, instructions, packet)
        return original(schema, instructions, packet, image)
    replay.structured = reply
    monkeypatch.setattr(agent, "Models", lambda _: replay)
    def run():
        return agent.run_paper(work, source, source.parent, new_batch_profile(TOKEN_PROFILE), lambda *_: None)
    result = run()
    if failure in {"checkpoint", "transport"}:
        assert result["status"] == "failed" and result["draft_round"] == 1
        assert not (work / "published.json").exists()
        if failure == "transport":
            assert "provider unreachable" in result["reason"]
            return
        # The exact saved exhaustion must resume locally without another provider operation.
        replay.structured = lambda *_args, **_kwargs: pytest.fail("unexpected model call after checkpoint recovery")
        result = run()
    else:
        assert len(wire_requests) == 3
    assert result["status"] == "completed", result.get("reason")
    assert result["delivery"]["status"] == "delivered" and "内容复核未能取得有效意见" in result["delivery"]["note"]
    assert result["draft_round"] == 1 and len(result["draft"]["core_bindings"]) == 2
    checked = result["reviews"][-1]
    assert checked["review_status"] == "unavailable" and checked["route"] == "final_delivery" and not checked["passed"]
    assert any(c["stage"] == "compile" for c in result["reviews"])
    assert "after three attempts" in checked["diagnostic"]
    case = Path(result["result_dir"])
    preflight = json.loads((case / "build-status.json").read_text("utf-8"))
    receipt = json.loads((work / "published.json").read_text("utf-8"))
    assert preflight["engineering_accepted"] and preflight["tex_sha256"] == receipt["hashes"]["introduction.tex"]
    assert preflight["compiled_pdf_sha256"] == receipt["hashes"]["introduction.pdf"]
    assert (case / "introduction.pdf").read_bytes().startswith(b"%PDF")


@pytest.mark.skipif(not shutil.which("pdflatex"), reason="requires local compiler")
def test_review_transport_failure_delivers_saved_formal_draft(tmp_path, monkeypatch):
    source = r"""\documentclass{article}
\usepackage{amsthm}
\newtheorem{maintheorem}{Theorem}[section]
\newtheorem{consequence}[maintheorem]{Corollary}
\begin{document}
\section{Main results}
\begin{maintheorem}\label{thm:main}
""" + MAIN + r"""
\end{maintheorem}
\begin{consequence}\label{cor:application}
""" + APPLICATION + r"""
\end{consequence}
\end{document}
"""
    replay = Replay("first")
    original = replay.structured
    attempts = []

    def reply(schema, instructions, packet, image=None):
        if schema is EvidenceReview:
            attempts.append(schema)
            raise ConnectionError("fixture connection interrupted during review")
        return original(schema, instructions, packet, image)

    replay.structured = reply
    monkeypatch.setattr(agent, "Models", lambda _: replay)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "offline-fixture-key")
    with TestClient(create_app(tmp_path, runner=worker.process)) as client:
        response = client.post("/api/imports", data={"paths": json.dumps(["352.tex"])},
                               files={"files": ("352.tex", source.encode("utf-8"), "application/x-tex")})
        assert response.status_code == 200, response.text
        imported = response.json()
        response = client.post("/api/batches", json={
            "import_id": imported["id"], "provider": "deepseek",
            "papers": [{"id": paper["id"], "entry": paper["entry"]} for paper in imported["papers"]],
        })
        assert response.status_code == 201, response.text
        batch = response.json()["id"]
        deadline = time.monotonic() + 30
        while True:
            paper, = client.get(f"/api/batches/{batch}").json()["papers"]
            if paper["status"] in {"completed", "failed"}:
                break
            assert time.monotonic() < deadline, paper
            time.sleep(0.05)
        assert paper["status"] == "completed", paper["reason"]
        assert paper["delivery"]["status"] == "delivered"
        assert paper["delivery"]["draft_round"] == 1
        assert len(paper["draft"]["groups"]) == 2
        assert attempts == [EvidenceReview]
        path = f"/api/papers/{paper['id']}/artifacts"
        tex = client.get(f"{path}/introduction.tex")
        pdf = client.get(f"{path}/introduction.pdf")
        assert tex.status_code == 200 and b"\\input{preamble-v13.tex}" not in tex.content
        assert APPLICATION.encode("utf-8") in tex.content
        assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
