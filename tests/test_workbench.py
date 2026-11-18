import io
import json
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from paper_extraction.api import create_app
from paper_extraction.sources import SourceIndex
from paper_extraction.store import Store
from paper_extraction import worker


def upload(client, pairs):
    files = [("files",(name.split("/")[-1],contents)) for name,contents in pairs]
    response = client.post("/api/imports",files=files,data={"paths":json.dumps([p[0] for p in pairs])})
    assert response.status_code == 200,response.text
    return response.json()


def wait_for(client, bid):
    for _ in range(150):
        result = client.get(f"/api/batches/{bid}").json()
        if all(p["status"] in {"completed","failed"} for p in result["papers"]):
            return result["papers"]
        time.sleep(.05)
    raise AssertionError("worker did not complete")


def test_batch_import_evidence_download_and_retry(tmp_path,monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY","offline-unit-test")
    def runner(home,pid):
        store = Store(home)
        p = store.paper(pid)
        if p["kind"] == "pdf":
            store.finish(p,{"status":"failed","reason":"fixture: unreadable PDF"})
            return
        index = SourceIndex(Path(p["source"]),Path(p["root"]),store.work(p))
        rows = index.catalog()
        store.event(pid,p["attempt"],"evidence",{"locations":[r["id"] for r in rows]})
        dest = store.work(p)/"result"
        dest.mkdir()
        for name in ("introduction.tex","introduction.pdf","preamble-v13.tex"):
            (dest/name).write_bytes(name.encode())
        store.finish(p,{"status":"completed","result_dir":str(dest),"evidence_links":[{"block_id":"groups.0.theorem","evidence_ids":[rows[-1]["id"]]}]})

    app = create_app(tmp_path,runner=runner)
    with TestClient(app) as client:
        batch = upload(client,[("study/main.tex",rb"\documentclass{article}\input{sections/a}"),
                               ("study/sections/a.tex",b"Source theorem."),("bad.pdf",b"bad")])
        assert len(batch["papers"]) == 2
        res = client.post("/api/batches",json={"import_id":batch["id"],"provider":"deepseek",
            "papers":[{"id":p["id"],"entry":p["entry"]} for p in batch["papers"]]})
        assert res.status_code == 201,res.text
        bid = res.json()["id"]
        tex,bad = sorted(wait_for(client,bid),key=lambda p:p["kind"],reverse=True)
        assert tex["status"] == "completed" and bad["status"] == "failed"
        evidence = client.get(f"/api/papers/{tex['id']}/evidence/t2-1").json()
        assert (evidence["file"],evidence["line"],evidence["text"]) == ("sections/a.tex",1,"Source theorem.")
        assert client.get(f"/api/papers/{bad['id']}/artifacts/introduction.pdf").status_code == 400
        assert "event: update" in client.get(f"/api/batches/{bid}/events").text
        zipped = zipfile.ZipFile(io.BytesIO(client.get(f"/api/batches/{bid}/download").content))
        assert len(zipped.namelist()) == 4 and "manifest.json" in zipped.namelist()
        assert client.post(f"/api/papers/{bad['id']}/retry").status_code == 200
        assert next(p for p in wait_for(client,bid) if p["id"]==bad["id"])["attempt"] == 2
    with TestClient(create_app(tmp_path,runner=runner)) as client:
        assert client.get("/api/batches").json()[0]["counts"] == {"failed":1,"completed":1}


def test_missing_key_explicit_and_ambiguous_root(tmp_path,monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY","")
    with TestClient(create_app(tmp_path)) as client:
        pack = upload(client,[("project/main.tex",rb"\documentclass{article}\input{x}"),
                              ("project/revised.tex",rb"\documentclass{article}\input{x}"),
                              ("project/x.tex",b"an included chapter")])
        assert len(pack["papers"]) == 1 and pack["papers"][0]["entry"] == ""
        config = client.get("/api/config").json()
        assert not config["providers"][0]["configured"]
        res = client.post("/api/batches",json={"import_id":pack["id"],"provider":"deepseek",
                          "papers":[{"id":pack["papers"][0]["id"],"entry":"project/main.tex"}]})
        assert res.status_code == 400 and "DEEPSEEK_API_KEY" in res.text


def test_missing_source_dependency_is_diagnosed(tmp_path):
    import pytest
    source = tmp_path/"a.tex"
    source.write_text(r"\input{missing}","utf-8")
    with pytest.raises(ValueError, match="missing"):
        SourceIndex(source,tmp_path,tmp_path/"work").catalog()


def test_retry_preserves_interrupted_graph_work_but_allocates_new_content_attempt(tmp_path):
    store = Store(tmp_path)
    store.create("batch", {"provider":"deepseek"}, [
        dict(id="paper",name="test",kind="tex",source="main.tex",root=None)])
    first = store.paper("paper")
    work = store.work(first)
    work.mkdir(parents=True)
    (work/"checkpoint.sqlite").touch()
    store.finish(first, {"status":"failed","reason":"network interrupted","resumable":True})
    store.retry("paper")
    resumed = store.paper("paper")
    assert resumed["attempt"] == 2 and store.work(resumed) == work
    store.finish(resumed, {"status":"failed","reason":"source has no supported history"})
    store.retry("paper")
    assert store.work(store.paper("paper")) != work


def test_pasted_paths_cannot_leave_staging(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        res=client.post("/api/imports",files=[("files",("a.tex",b"A"))],data={"paths":'["../a.tex"]'})
        assert res.status_code == 400


def test_configured_import_file_limit_counts_files_in_one_tex_project(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_EXTRACTION_MAX_IMPORT_FILES", "2")
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/config").json()["max_import_files"] == 2
        schema = client.get("/openapi.json").json()
        multipart = schema["paths"]["/api/imports"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
        form = schema["components"]["schemas"][multipart["$ref"].split("/")[-1]]
        assert set(form["required"]) == {"files", "paths"}
        assert form["properties"]["files"]["type"] == "array" and form["properties"]["paths"]["type"] == "string"
        papers = upload(client, [("main.tex", rb"\documentclass{article}\input{chapter}"), ("chapter.tex", b"chapter")])
        assert len(papers["papers"]) == 1
        parts = [("files", (name, b"data")) for name in ("main.tex", "chapter.tex", "image.png")]
        result = client.post("/api/imports", files=parts, data={"paths": '["main.tex","chapter.tex","image.png"]'})
        assert result.status_code == 400 and "最多允许 2 个文件" in result.json()["detail"]


@pytest.mark.parametrize(("limit", "expected"), [(None, 400), ("1001", 200), ("0", 200)])
def test_large_multipart_folder_and_paths_metadata(tmp_path, monkeypatch, limit, expected):
    monkeypatch.setenv("PAPER_EXTRACTION_ENV_FILE", str(tmp_path / "isolated.env"))
    monkeypatch.delenv("PAPER_EXTRACTION_MAX_IMPORT_FILES", raising=False)
    if limit is not None:
        monkeypatch.setenv("PAPER_EXTRACTION_MAX_IMPORT_FILES", limit)
    handles = []
    def stage(home, files, paths):
        assert len(files) == len(paths) == 1001 and paths[0].endswith("0.pdf")
        handles.extend(item.file for item in files)
        assert not any(handle.closed for handle in handles)
        return {"papers": []}
    monkeypatch.setattr("paper_extraction.api.stage", stage)
    parts = [("files", (f"{i}.pdf", b"x")) for i in range(1001)]
    # Long nested source-folder paths together exceed Starlette's default 1 MiB text-field limit.
    prefix = ("long-section-name/" * 60)
    paths = json.dumps([f"{prefix}{i}.pdf" for i in range(len(parts))])
    assert len(paths.encode()) > 1024 * 1024
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/config").json()["max_import_files"] == (int(limit) if limit else 1000)
        response = client.post("/api/imports", files=parts, data={"paths": paths})
    assert response.status_code == expected, response.text
    if expected == 200:
        assert all(handle.closed for handle in handles)
    else:
        assert not handles and "最多允许 1000 个文件" in response.json()["detail"]


@pytest.mark.parametrize("invalid", ["-1", "2.5", "none"])
def test_invalid_import_limit_reports_key_at_startup(tmp_path, monkeypatch, invalid):
    monkeypatch.setenv("PAPER_EXTRACTION_MAX_IMPORT_FILES", invalid)
    with pytest.raises(ValueError, match="PAPER_EXTRACTION_MAX_IMPORT_FILES"):
        create_app(tmp_path)
