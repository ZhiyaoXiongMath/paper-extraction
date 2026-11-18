import io
import json
from pathlib import Path
import shutil
import zipfile

import pytest
from fastapi.testclient import TestClient

from paper_extraction.api import create_app
from paper_extraction.draft_artifacts import draft_digest
from paper_extraction.reviewed_revision import accept_revision, prepare_revision
from test_draft_artifacts import prepare
from test_profile import synthetic_draft


@pytest.mark.skipif(not shutil.which("pdflatex"), reason="requires the v13 TeX toolchain")
def test_operator_audit_is_bound_to_inspected_pages_and_changed_source_fields(tmp_path):
    store, paper = prepare(tmp_path)
    draft = synthetic_draft()
    draft["paragraph"] = " ".join(["Setting"] * 54) + "."
    draft["evidence_links"] = [{"block_id": name, "evidence_ids": ["t1-1"]}
                               for name in ("paragraph", "history.0", "groups.0.theorem")]
    verdict = {"status": "failed", "draft": draft, "draft_round": 3, "reason": "Word limit exceeded", "reviews": [
        {"stage": "review", "attempt": 3, "passed": True, "route": "pass", "issues": []},
        {"stage": "compile", "attempt": 3, "passed": False, "issues": ["Word limit exceeded"], "repair_targets": [
            {"draft_field": "paragraph", "current_text": draft["paragraph"], "diagnostics": ["54 words > 45"]}]}]}
    store.record_draft(paper, 3, draft)
    store.finish(paper, verdict)
    digest = draft_digest(draft)
    patch = tmp_path / "patch.json"
    patch.write_text(json.dumps({"replacements": [{"block_id": "paragraph", "text": "We consider positive real numbers.",
                                                  "evidence_ids": ["t1-1"]}]}), encoding="utf8")
    with pytest.raises(ValueError):
        prepare_revision(store, "a", 1, 3, digest + "x", patch)
    made = prepare_revision(store, "a", 1, 3, digest, patch)
    case, audit_file = Path(made["case"]), Path(made["audit_template"])
    before = store.paper("a")
    audit = json.loads(audit_file.read_text("utf8"))
    audit["operator"] = "offline test fixture"
    for item in audit["changes"] + audit["pages"]:
        item.update(passed=True, observation="Deterministic test fixture for recording the inspection.")
    valid = json.dumps(audit)
    audit["pages"][0]["sha256"] = "wrong image"
    audit_file.write_text(json.dumps(audit), encoding="utf8")
    with pytest.raises(ValueError, match="图像哈希"):
        accept_revision(store, "a", case, audit_file)
    assert store.paper("a") == before
    audit_file.write_text(valid, encoding="utf8")
    accepted = accept_revision(store, "a", case, audit_file)
    assert accepted == {"paper_id": "a", "attempt": 2, "round": 4, "status": "completed"}
    assert accept_revision(store, "a", case, audit_file) == accepted
    with TestClient(create_app(tmp_path, runner=lambda *args: None)) as client:
        p = client.get("/api/batches/b").json()["papers"][0]
        fourth, third = p["draft_artifacts"]
        assert fourth["current"] and p["status"] == "completed"
        assert third["attempt_status"] == "failed" and third["attempt_reason"] == verdict["reason"]
        assert client.get(third["files"]["tex"]["url"]).status_code == 200
        archive = zipfile.ZipFile(io.BytesIO(client.get("/api/papers/a/download").content))
        assert set(archive.namelist()) == {"source-extraction.tex", "source-extraction.pdf"}
        assert [r["method"] for r in p["reviews"][2:]] == ["operator_delta_review", "v13_checker"] + [
            "operator_rendered_page"] * len(made["page_images"])
