import json
import sqlite3
from urllib.parse import unquote

from fastapi.testclient import TestClient

from paper_extraction.api import create_app
from paper_extraction.store import Store


def test_existing_unnamed_batch_can_be_named_without_changing_its_id(tmp_path):
    with sqlite3.connect(tmp_path / "jobs.sqlite3") as db:
        db.execute("CREATE TABLE batches(id TEXT PRIMARY KEY, created_at TEXT NOT NULL, settings TEXT NOT NULL)")
        db.execute("INSERT INTO batches VALUES(?,?,?)", ("existing", "2026-01-01", json.dumps({"provider": "openai"})))
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/batches").json()[0]["name"] is None
        assert client.get("/api/batches/existing/download").headers["content-disposition"] == (
            'attachment; filename="papers-existing.zip"'
        )
        assert client.get("/api/batches/unknown/download").status_code == 404
        saved = client.patch("/api/batches/existing/name", json={"name": "  曲率流选读  "})
        assert saved.json() == {"id": "existing", "name": "曲率流选读"}
        assert client.patch("/api/batches/existing/name", json={"name": " "}).status_code == 422
        assert client.patch("/api/batches/unknown/name", json={"name": "新名称"}).status_code == 404
        assert client.get("/api/batches").json()[0]["name"] == "曲率流选读"
    assert Store(tmp_path).describe("existing")["name"] == "曲率流选读"


def test_create_and_rename_batch_keeps_original_paper_and_route(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    with TestClient(create_app(tmp_path, runner=lambda *_: None)) as client:
        preview = client.post("/api/imports", files={"files": ("example.tex", b"Example")},
                              data={"paths": '["example.tex"]'}).json()
        paper = preview["papers"][0]
        create = client.post("/api/batches", json={"import_id": preview["id"], "provider": "openai",
                             "name": "几何 / 第一组", "papers": [{"id": paper["id"], "entry": paper["entry"]}]})
        assert create.status_code == 201
        path = "/api/batches/" + create.json()["id"]
        before = client.get(path).json()
        assert before["name"] == "几何 / 第一组"
        assert unquote(client.get(path + "/download").headers["content-disposition"]) == (
            "attachment; filename*=UTF-8''几何 _ 第一组.zip"
        )
        assert client.patch(path + "/name", json={"name": "几何 / 第二组"}).is_success
        assert client.get(path).json() == {**before, "name": "几何 / 第二组"}
        assert unquote(client.get(path + "/download").headers["content-disposition"]) == (
            "attachment; filename*=UTF-8''几何 _ 第二组.zip"
        )
