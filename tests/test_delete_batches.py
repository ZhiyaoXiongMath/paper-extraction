import io
import json
from pathlib import Path
import sys
from threading import Event
from urllib.parse import unquote
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from paper_extraction import worker
from paper_extraction.api import create_app
from paper_extraction.store import NAMES, Store
from test_profile import synthetic_draft


_PREAMBLE = "\\documentclass{article}\n"
_BODY = "\\begin{document}\nSample.\n\\end{document}\n"


@pytest.fixture
def workbench(tmp_path_factory, monkeypatch):
    def offline_settings(provider, with_key):
        return {"provider": provider, "api_key": "fixture-only"}

    monkeypatch.setattr("paper_extraction.api.provider_settings", offline_settings)
    monkeypatch.setattr("paper_extraction.store.provider_settings", offline_settings)
    store = Store(tmp_path_factory.mktemp("del"))
    with TestClient(create_app(store.home, runner=lambda *_: None)) as client:
        yield client, store


def create_batch(client, *names):
    preview = client.post("/api/imports", files=[("files", (name, name.encode())) for name in names],
                          data={"paths": json.dumps(names)})
    assert preview.status_code == 200
    items = preview.json()["papers"]
    created = client.post("/api/batches", json={
        "name": "Fixture", "import_id": preview.json()["id"], "provider": "openai",
        "papers": [{"id": item["id"], "entry": item["entry"]} for item in items],
    })
    assert created.status_code == 201
    return client.get("/api/batches/" + created.json()["id"]).json()


def publish_fixture(store, paper):
    case = store.work(store.paper(paper["id"])) / "accepted"
    case.mkdir(parents=True)
    for name, data in zip(NAMES, [r"\input{preamble-v13.tex}" + "\n" + _BODY, "introduction.pdf", _PREAMBLE], strict=True):
        (case / name).write_bytes(data.encode())
    store.finish(store.paper(paper["id"]), {"status": "completed", "result_dir": str(case)})


def test_delete_one_finished_batch_removes_its_downloads_and_history(workbench, tmp_path):
    client, store = workbench
    batch = create_batch(client, "紧致性.pdf", "failed.pdf")
    batch_url = "/api/batches/" + batch["id"]
    finished, failed = batch["papers"]
    store.event(finished["id"], 1, "compile", {})
    publish_fixture(store, finished)
    store.record_draft(store.paper(failed["id"]), 1, synthetic_draft())
    store.finish(store.paper(failed["id"]), {"status": "failed", "reason": "Layout"})
    assert client.post(f"/api/papers/{failed['id']}/retry", json={"provider": "deepseek"}).is_success
    store.finish(store.paper(failed["id"]), {"status": "failed", "reason": "Layout"})
    version, = store.describe(batch["id"])["papers"][1]["draft_artifacts"]

    files = [batch_url + "/download", f"/api/papers/{finished['id']}/artifacts/introduction.tex",
             f"/api/papers/{failed['id']}/source", version["files"]["tex"]["url"]]
    assert all(client.get(path).status_code == 200 for path in files)
    bundle = client.get(f"/api/papers/{finished['id']}/download")
    assert unquote(bundle.headers["content-disposition"]) == "attachment; filename*=UTF-8''紧致性-extraction.zip"
    with ZipFile(io.BytesIO(bundle.content)) as archive:
        assert set(archive.namelist()) == {"紧致性-extraction.tex", "紧致性-extraction.pdf"}
        assert archive.read("紧致性-extraction.tex") == client.get(files[1]).content
    saved_copy = tmp_path / "my-download.tex"
    tex = client.get(files[1])
    assert unquote(tex.headers["content-disposition"]) == "inline; filename*=UTF-8''紧致性-extraction.tex"
    saved_copy.write_bytes(tex.content)
    other = create_batch(client, "unrelated.pdf")
    all_imports = list((store.home / "imports").glob("*/manifest.json"))
    assert client.delete(batch_url).json() == {"deleted_batch_ids": [batch["id"]]}
    assert client.delete(batch_url).status_code == 404
    assert all(client.get(path).status_code == 404 for path in [batch_url, *files])
    assert not (store.home / "runs" / batch["id"]).exists()
    exported = saved_copy.read_text("utf-8")
    assert _PREAMBLE in exported and exported.endswith(_BODY)
    assert r"\input{preamble-v13.tex}" not in exported
    assert all(path.exists() for path in all_imports)
    assert Path(store.paper(other["papers"][0]["id"])["source"]).is_file()
    with store.db() as db:
        for table in ("events", "archived_reviews", "paper_configs", "draft_artifacts"):
            assert db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM batch_names WHERE batch_id=?", (batch["id"],)).fetchone()[0] == 0


def test_clear_terminal_batches_skips_active_batches_and_preserves_settings(workbench):
    client, store = workbench
    mixed = create_batch(client, "already-done.pdf", "still-running.pdf")
    publish_fixture(store, mixed["papers"][0])
    store.event(mixed["papers"][1]["id"], 1, "review", {"message": "Still running"})
    queued = create_batch(client, "still-queued.pdf")
    completed = create_batch(client, "complete.pdf")
    publish_fixture(store, completed["papers"][0])
    rejected = create_batch(client, "rejected.pdf")
    store.finish(store.paper(rejected["papers"][0]["id"]), {"status": "failed", "reason": "Layout"})
    assert client.patch("/api/config/concurrency", json={"concurrency": 7}).is_success
    for active in (queued, mixed):
        assert client.delete("/api/batches/" + active["id"]).status_code == 409

    response = client.delete("/api/batches")
    assert response.status_code == 200
    assert sorted(response.json()["deleted_batch_ids"]) == sorted([completed["id"], rejected["id"]])
    assert client.delete("/api/batches").json() == {"deleted_batch_ids": []}
    assert {row["id"] for row in client.get("/api/batches").json()} == {queued["id"], mixed["id"]}
    assert {p["status"] for p in store.describe(mixed["id"])["papers"]} == {"completed", "running"}
    assert store.describe(queued["id"])["papers"][0]["status"] == "queued"
    assert client.get("/api/config").json()["concurrency"] == 7
    assert all((store.home / "runs" / b["id"]).is_dir() for b in (queued, mixed))
    assert all(not (store.home / "runs" / b["id"]).exists() for b in (completed, rejected))


@pytest.mark.parametrize("include_finished", [False, True])
def test_delete_single_batch_after_all_active_papers_pause(workbench, include_finished):
    client, store = workbench
    names = ["starting.pdf", "waiting.pdf"]
    if include_finished:
        names += ["accepted.pdf", "failed.pdf"]
    batch = create_batch(client, *names)
    bid = batch["id"]
    url = "/api/batches/" + bid
    starting, waiting, *finished = batch["papers"]
    if finished:
        publish_fixture(store, finished[0])
        store.finish(store.paper(finished[1]["id"]), {"status": "failed", "reason": "Layout"})

    # Even a worker which has not yet opened its work directory must finish pausing.
    assert store.claim_worker(starting["id"]) == bid
    assert client.post(url + "/pause").json() == {"id": bid, "affected": 2}
    assert store.paper(starting["id"])["status"] == "pausing"
    assert store.paper(waiting["id"])["status"] == "paused"
    response = client.delete(url)
    assert response.status_code == 409
    assert "全部暂停或结束" in response.json()["detail"]
    store.finish(store.paper(starting["id"]), {"status": "paused"})
    assert client.request("DELETE", "/api/batches", json={"batch_ids": [bid]}).json() == {"deleted_batch_ids": []}

    # Current status takes precedence if another tab resumes before deletion.
    assert client.post(url + "/resume").json() == {"id": bid, "affected": 2}
    assert client.delete(url).status_code == 409
    assert client.post(url + "/pause").json() == {"id": bid, "affected": 2}
    assert client.delete(url).json() == {"deleted_batch_ids": [bid]}
    assert client.get(url).status_code == 404

    worker.process(store.home, waiting["id"])
    assert not (store.home / "runs" / bid).exists()


def test_bulk_delete_only_uses_confirmed_ids_and_rechecks_status(workbench):
    client, store = workbench
    previously_finished = create_batch(client, "finished.pdf")
    publish_fixture(store, previously_finished["papers"][0])
    retried = create_batch(client, "retry.pdf")
    retried_pid = retried["papers"][0]["id"]
    store.finish(store.paper(retried_pid), {"status": "failed", "reason": "Layout"})
    finishing_now = create_batch(client, "newly-completed.pdf")
    confirmed_ids = [r["id"] for r in client.get("/api/batches").json()
                     if r["counts"].get("completed", 0) + r["counts"].get("failed", 0) == r["total"]]
    assert set(confirmed_ids) == {previously_finished["id"], retried["id"]}

    # Another tab retries a failed paper while the confirmation is open.
    assert client.post(f"/api/papers/{retried_pid}/retry").is_success
    publish_fixture(store, finishing_now["papers"][0])
    assert client.request("DELETE", "/api/batches", json={"batch_ids": []}).json() == {"deleted_batch_ids": []}
    removed = client.request("DELETE", "/api/batches",
                             json={"batch_ids": [*confirmed_ids, previously_finished["id"], "unknown"]})
    assert removed.status_code == 200
    assert removed.json() == {"deleted_batch_ids": [previously_finished["id"]]}
    assert {r["id"] for r in client.get("/api/batches").json()} == {retried["id"], finishing_now["id"]}
    assert client.get(f"/api/batches/{finishing_now['id']}/download").status_code == 200


def test_cleanup_recovers_after_a_batch_file_was_locked(workbench, monkeypatch):
    client, store = workbench
    batch = create_batch(client, "locked.pdf")
    publish_fixture(store, batch["papers"][0])
    directory = store.home / "runs" / batch["id"]

    def held_open(path):
        assert Path(path) == directory
        raise PermissionError("File is open")

    with monkeypatch.context() as patch:
        patch.setattr("paper_extraction.store.shutil.rmtree", held_open)
        assert client.delete("/api/batches/" + batch["id"]).json() == {"deleted_batch_ids": [batch["id"]]}
    assert directory.is_dir()
    assert client.get(f"/api/batches/{batch['id']}/download").status_code == 404
    with TestClient(create_app(store.home, runner=lambda *_: None)) as restarted:
        assert restarted.get("/api/batches").json() == []
        assert not directory.exists()


def test_cleanup_retries_when_a_paused_worker_exits(workbench, monkeypatch):
    client, store = workbench
    working, checkpoint, paused, exited, cleaned = (Event() for _ in range(5))

    def runner(home, pid):
        assert store.claim_worker(pid)
        working.set()
        assert checkpoint.wait(5)
        store.finish(store.paper(pid), {"status": "paused"})
        paused.set()
        assert exited.wait(5)

    import shutil
    remove = shutil.rmtree

    def remove_when_closed(path):
        if not exited.is_set():
            raise PermissionError("Worker lock is still open")
        remove(path)
        cleaned.set()

    monkeypatch.setattr("paper_extraction.store.shutil.rmtree", remove_when_closed)
    client.app.state.dispatch.runner = runner
    try:
        batch = create_batch(client, "pausing.pdf")
        url = "/api/batches/" + batch["id"]
        directory = store.home / "runs" / batch["id"]
        assert working.wait(5)
        assert client.post(url + "/pause").is_success
        checkpoint.set()
        assert paused.wait(5)
        assert client.delete(url).is_success
        assert directory.exists()
    finally:
        checkpoint.set()
        exited.set()
    assert cleaned.wait(5)
    assert not directory.exists()


def test_a_linked_runs_root_cannot_delete_original_files(workbench, tmp_path):
    client, store = workbench
    external = tmp_path / "papers"
    original = external / "old-batch" / "source.tex"
    original.parent.mkdir(parents=True)
    original.write_text("Original paper", encoding="utf-8")
    linked_runs = store.home / "runs"
    try:
        if sys.platform == "win32":
            import _winapi
            _winapi.CreateJunction(str(external), str(linked_runs))
        else:
            linked_runs.symlink_to(external, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"Directory links unavailable: {exc}")
    store.create("old-batch", {"provider": "openai"}, [
        {"id": "old-paper", "name": "Paper", "kind": "tex", "source": str(original), "root": str(original.parent)}])
    store.finish(store.paper("old-paper"), {"status": "failed", "reason": "Layout"})
    assert client.delete("/api/batches/old-batch").status_code == 400
    assert original.read_text(encoding="utf-8") == "Original paper"
    assert client.get("/api/batches/old-batch").status_code == 200


def test_deleting_a_legacy_record_with_missing_run_files_keeps_its_original(workbench, tmp_path):
    client, store = workbench
    source = tmp_path / "supplied.tex"
    source.write_text("Original paper", encoding="utf-8")
    store.create("legacy", {"provider": "openai"}, [
        {"id": "paper", "name": "Old paper", "kind": "tex", "source": str(source), "root": str(tmp_path)}])
    store.finish(store.paper("paper"), {"status": "failed", "reason": "No draft"})
    assert not (store.home / "runs" / "legacy").exists()
    assert client.delete("/api/batches/legacy").json() == {"deleted_batch_ids": ["legacy"]}
    assert source.read_text(encoding="utf-8") == "Original paper"
