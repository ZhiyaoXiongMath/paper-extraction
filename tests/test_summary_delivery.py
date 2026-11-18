import hashlib
import io
import json
from zipfile import ZipFile
from urllib.parse import unquote

import pytest
from fastapi.testclient import TestClient

from paper_extraction.api import create_app
from paper_extraction.store import Store
from paper_extraction.summary_delivery import FORMAL_NAMES, SUMMARY_NAME, write_summary


def test_summary_only_and_mixed_exports(tmp_path):
    store = Store(tmp_path / "home")
    source = tmp_path / "原文.tex"
    source.write_text("A discussion of the problem.", encoding="utf-8")
    items = [{"id": key, "name": name, "kind": "tex", "source": str(source), "root": str(tmp_path)}
             for key, name in (("formal", "Formal"), ("screened", "Screened"), ("unverified", "Unverified"))]
    store.create("batch", {"provider": "openai"}, items, name="测试批次")
    formal = store.paper("formal")
    case = store.work(formal) / "formal"
    case.mkdir(parents=True)
    for name, data in zip(FORMAL_NAMES, [b"formal TeX", b"formal PDF", b"preamble"], strict=True):
        (case / name).write_bytes(data)
    store.finish(formal, {"status": "completed", "result_dir": str(case)})
    for key, kind, locs in (("screened", "no_qualifying_result", ["t1", "t1"]),
                            ("unverified", "insufficient_evidence", [])):
        paper = store.paper(key)
        evidence = [{"id": "t1", "label": "main.tex L1–8"}]
        case, delivery = write_summary(store.work(paper), source, kind=kind, question="本文研究什么？",
                                       reason="样例的具体筛选理由。", evidence_ids=locs, catalog=evidence)
        assert {f.name for f in case.iterdir()} == {SUMMARY_NAME}
        store.finish(paper, {"status": "completed", "result_dir": str(case), "delivery": delivery})

    with TestClient(create_app(store.home, runner=lambda *_: pytest.fail("Unexpected task launch"))) as client:
        batch = client.get("/api/batches/batch").json()
        assert batch["counts"] == {"completed": 3} and batch["summary_count"] == 2
        archive_response = client.get("/api/batches/batch/download")
        assert archive_response.status_code == 200 and "测试批次.zip" in unquote(archive_response.headers["content-disposition"])
        with ZipFile(io.BytesIO(archive_response.content)) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            assert set(archive.namelist()) == {
                "001-Formal/原文-extraction.tex", "001-Formal/原文-extraction.pdf",
                "002-Screened/原文-extraction.md", "003-Unverified/原文-extraction.md", "manifest.json"}
            for ordinal, item in enumerate(batch["papers"][1:], 2):
                key, delivery = item["id"], item["delivery"]
                assert delivery["kind"] == "summary" and delivery["draft_round"] == 0
                assert not item["draft"] and not item["draft_artifacts"]
                url = f"/api/papers/{key}/artifacts/"
                note = client.get(url + SUMMARY_NAME)
                assert note.status_code == 200 and "text/markdown" in note.headers["content-type"]
                assert "原文-extraction.md" in unquote(note.headers["content-disposition"])
                assert delivery["summary"]["question"] in note.text and delivery["summary"]["reason"] in note.text
                if key == "screened":
                    assert delivery["outcome"] == "no_qualifying_result"
                    assert note.text.count("main.tex L1–8") == 1 and "已核对原文" in note.text
                else:
                    assert delivery["outcome"] == "insufficient_evidence"
                    assert "无法核实" in note.text and "已核对原文" not in note.text
                for formal_name in FORMAL_NAMES:
                    assert client.get(url + formal_name).status_code == 400
                with ZipFile(io.BytesIO(client.get(f"/api/papers/{key}/download").content)) as single:
                    assert single.namelist() == ["原文-extraction.md"] and single.read(single.namelist()[0]) == note.content
                assert archive.read(f"{ordinal:03}-{item['name']}/原文-extraction.md") == note.content
                row = manifest["papers"][ordinal - 1]
                assert row["sha256"] == {"原文-extraction.md": hashlib.sha256(note.content).hexdigest()}
                assert row["delivery"]["outcome"] == delivery["outcome"]
        assert client.get("/api/papers/formal/artifacts/summary.md").status_code == 400
        paper = store.paper("screened")
        store.artifact(paper, SUMMARY_NAME).write_text("changed", encoding="utf-8")
        assert client.get("/api/papers/screened/artifacts/summary.md").status_code == 400


def test_summary_requires_valid_original_evidence(tmp_path):
    for kind, ids in (("no_qualifying_result", []), ("insufficient_evidence", ["unknown"])):
        with pytest.raises(ValueError):
            write_summary(tmp_path, tmp_path / "paper.pdf", kind=kind, question="问题", reason="说明",
                          evidence_ids=ids, catalog=[])
