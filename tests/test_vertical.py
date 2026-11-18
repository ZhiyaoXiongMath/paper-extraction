"""Local provider fixture exercising the full batch path without a model account."""
from __future__ import annotations

import base64
from contextlib import contextmanager
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
import shutil
import threading
import time
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from paper_extraction.api import create_app
from fixture_selection import selection_reply

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "compactness"
TITLE = "Compactness for Equicontinuous Families on an Interval"
HISTORY = "Every bounded sequence of real numbers admits a convergent subsequence."
CURRENT = (r"A family $\mathcal F\subset C([0,1],\mathbb R)$ is relatively compact in the supremum "
           "norm if and only if it is uniformly bounded and equicontinuous.")
DEFINITION = (r"A family $\mathcal F$ is equicontinuous if for every $\varepsilon>0$ some $\delta>0$ "
              r"satisfies $|f(x)-f(y)|<\varepsilon$ whenever $|x-y|<\delta$, for all $x,y\in[0,1]$ "
              r"and $f\in\mathcal F$.")
INCOMPLETE = (r"\documentclass{article}\begin{document}\section{An incomplete teaching note} "
              r"This note asks a question. No historical result is stated.\end{document}")


def provider_answer(request, visual_pages):
    """Use actual IDs from the submitted snapshot. Replies are deterministic test fixtures."""
    sys_msg = request["messages"][0]["content"]
    payload = request["messages"][1]["content"]
    is_visual = isinstance(payload, list)
    if is_visual:
        data = json.loads(next(part["text"] for part in payload if part["type"] == "text"))
        uri = next(part["image_url"]["url"] for part in payload if part["type"] == "image_url")
        assert uri.startswith("data:image/png;base64,")
        assert base64.b64decode(uri.split(",", 1)[1]).startswith(b"\x89PNG\r\n\x1a\n")
        visual_pages.append(data["page"])
    else:
        data = json.loads(payload)

    if request.get("tools"):
        rules = files("paper_extraction").joinpath("profile_assets", "intro-rules-v13.txt").read_text("utf-8")
        assert sys_msg.startswith(rules + "\n") and "Return the original location IDs" in sys_msg[len(rules):]
        spans = data.get("passages") or data.get("original_abstract_intro_results_and_applications")
        if spans and not any(m["role"] == "tool" for m in request["messages"]):
            args = json.dumps({"location": spans[0]["id"]})
            return {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_fixture", "type": "function", "function": {"name": "read_span", "arguments": args},
            }]}
        return {"role": "assistant", "content": "Read the supplied evidence locations."}

    # The adapter supplies the live Pydantic schema in its system message.
    schema = json.loads(sys_msg.rsplit("\n", 1)[1])["title"]
    if schema == "Findings":
        spans = data["passages"]
        results = []
        for kind, needle, claim, original in [
            ("historical", "Bolzano--Weierstrass", HISTORY, "Background and question"),
            ("definition", r"\label{def:equicontinuity}", DEFINITION, "Definition 1"),
            ("current", r"\label{thm:criterion}", CURRENT, "Theorem 1"),
        ]:
            if match := next((s for s in spans if needle in s["text"]), None):
                results.append({"kind": kind, "claim": claim, "assumptions": "As stated in the fixture.",
                                "significance": "The manuscript's core comparison and criterion.",
                                "source_number": original, "dependencies": [], "decision": "admitted",
                                "decision_reason": "Fixed candidate from the teaching sample.",
                                "evidence_ids": [match["id"]]})
        answer = {"candidates": results, "uncertainties": []}
    elif schema == "WholePaperSelection":
        answer = selection_reply(data["candidate_register"])
    elif schema == "Draft":
        ids = {c["kind"]: c["evidence_ids"] for c in data["candidates"]}
        answer = {
            "title": TITLE, "authors": ["Paper Extraction Contributors"],
            "paragraph": "We ask when sequences of continuous functions on the unit interval converge along subsequences.",
            "history": [HISTORY + r" \cite[Background]{source}"],
            "shared_definitions": [],
            "groups": [{"definitions": [DEFINITION + r" \cite[Definition~1]{source}"],
                        "theorem": CURRENT + r" \cite[Theorem~1]{source}", "paragraph": ""}],
            "references": [{"key": "source", "text": f"Paper Extraction Contributors, {TITLE}. Teaching sample."}],
            "evidence_links": [{"block_id": name, "evidence_ids": ids[kind]} for name, kind in [
                ("paragraph", "historical"), ("history.0", "historical"), ("groups.0.theorem", "current"),
                ("groups.0.definitions.0", "definition")]],
        }
    elif schema == "Review":
        answer = {"passed": True, "issues": []}
    elif schema == "AdmissionReview":
        answer = {"route": "pass", "issues": [], "evidence_ids": []}
    else:
        raise ValueError(f"Unexpected model response schema {schema}")
    return {"role": "assistant", "content": json.dumps(answer)}


@contextmanager
def fixture_provider():
    visual_pages, errors = [], []

    class FixtureHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                assert self.path == "/v1/chat/completions"
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                reply = provider_answer(request, visual_pages)
                response = {"id": "fixture-completion", "object": "chat.completion", "created": 1,
                            "model": request["model"], "choices": [{"index": 0, "message": reply,
                            "finish_reason": "tool_calls" if reply.get("tool_calls") else "stop"}]}
                code = 200
            except Exception as exc:
                errors.append(repr(exc))
                response, code = {"error": {"message": repr(exc)}}, 400
            data = json.dumps(response).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", visual_pages, errors
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.skipif(not shutil.which("pdflatex"), reason="requires the v13 TeX toolchain")
def test_full_batch_with_loopback_model_fixture(tmp_path, monkeypatch):
    paths = ["compactness/main.tex", "compactness/sections/criterion.tex", "incomplete/main.tex"]
    sources = [(EXAMPLE / "main.tex").read_bytes(), (EXAMPLE / "sections" / "criterion.tex").read_bytes(),
               INCOMPLETE.encode("utf-8")]
    files = [("files", (Path(path).name, data, "application/x-tex")) for path, data in zip(paths, sources)]
    with fixture_provider() as (base_url, images, errors):
        monkeypatch.setenv("DEEPSEEK_BASE_URL", base_url)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "local-test-fixture-only")
        monkeypatch.setenv("DEEPSEEK_TEXT_MODEL", "fixture-text")
        monkeypatch.setenv("DEEPSEEK_VISION_MODEL", "fixture-vision")
        monkeypatch.setenv("LANGSMITH_TRACING", "false")
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
        with TestClient(create_app(tmp_path)) as client:
            response = client.post("/api/imports", data={"paths": json.dumps(paths)}, files=files)
            assert response.status_code == 200, response.text
            imported = response.json()
            choices = [{"id": p["id"], "entry": p["entry"]} for p in imported["papers"]]
            response = client.post("/api/batches", json={"import_id": imported["id"], "provider": "deepseek",
                                                         "papers": choices})
            assert response.status_code == 201, response.text
            bid = response.json()["id"]
            deadline = time.monotonic() + 120
            while True:
                report = client.get(f"/api/batches/{bid}").json()
                if all(p["status"] in {"completed", "failed"} for p in report["papers"]):
                    break
                assert time.monotonic() < deadline, (report, errors)
                time.sleep(.15)
            papers = {p["name"]: p for p in report["papers"]}
            completed = papers["compactness"]
            failed = papers["incomplete"]
            assert not errors
            assert completed["status"] == "completed", completed["reason"]
            assert failed["status"] == "failed"
            assert "历史定理" in failed["reason"] and "当前核心结果" in failed["reason"], failed["reason"]
            assert sorted(images) == [1, 2]
            assert [r["page"] for r in completed["reviews"] if r["stage"] == "visual"] == [1, 2]
            assert any(e["data"].get("tool") == "read_span" for e in completed["events"])
            pid = completed["id"]
            locations = next(link["evidence_ids"] for link in completed["links"] if link["block_id"]=="groups.0.theorem")
            result = client.get(f"/api/papers/{pid}/evidence/{locations[0]}").json()
            assert result["file"] == "sections/criterion.tex" and r"\label{thm:criterion}" in result["text"]
            pdf = client.get(f"/api/papers/{pid}/artifacts/introduction.pdf")
            assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
            assert pdf.headers["content-disposition"] == 'inline; filename="main-extraction.pdf"'
            tex = client.get(f"/api/papers/{pid}/artifacts/introduction.tex")
            assert tex.headers["content-disposition"] == 'inline; filename="main-extraction.tex"'
            assert b"\\documentclass" in tex.content and b"\\input{preamble-v13.tex}" not in tex.content
            with ZipFile(io.BytesIO(client.get(f"/api/papers/{pid}/download").content)) as archive:
                assert set(archive.namelist()) == {"main-extraction.tex", "main-extraction.pdf"}
                assert archive.read("main-extraction.tex") == tex.content
                assert archive.read("main-extraction.pdf") == pdf.content
            success = next((tmp_path / "runs" / bid / pid).rglob("build-status.json"))
            status = json.loads(success.read_text(encoding="utf-8"))
            assert status["input_binding_checked"] and status["engineering_accepted"]
            assert len(status["returncodes"]) >= 2 and not any(status["returncodes"])
            assert client.get(f"/api/papers/{failed['id']}/download").status_code == 400
            with ZipFile(io.BytesIO(client.get(f"/api/batches/{bid}/download").content)) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                assert {p["status"] for p in manifest["papers"]} == {"completed", "failed"}
                assert {Path(n).name for n in archive.namelist()} == {
                    "manifest.json", "main-extraction.tex", "main-extraction.pdf",
                }

