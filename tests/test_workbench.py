import io
import time
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from paper_extraction.api import create_app
from paper_extraction.sources import SourceIndex
from paper_extraction.store import Store
from paper_extraction import worker


def upload(client, pairs):
    import json
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


def test_pasted_paths_cannot_leave_staging(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        res=client.post("/api/imports",files=[("files",("a.tex",b"A"))],data={"paths":'["../a.tex"]'})
        assert res.status_code == 400
