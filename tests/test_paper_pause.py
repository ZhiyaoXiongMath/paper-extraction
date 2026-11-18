"""Exercise HTTP pause and resume with checkpointed, offline paper workers."""

from threading import Event
from time import monotonic, sleep

from fastapi.testclient import TestClient

from paper_extraction import worker
from paper_extraction.api import create_app
from paper_extraction.models import Draft
from paper_extraction.store import Store
from test_agent import paper  # noqa: F401 - offline model and compiler fixtures


def wait_for_status(store, pid, status):
    start = monotonic()
    while True:
        current = store.paper(pid)
        if current["status"] == status:
            return current
        assert monotonic() - start < 5, (status, current)
        sleep(0.01)


def test_queued_pause_survives_service_restart_and_resumes_same_attempt(tmp_path):
    store = Store(tmp_path)
    store.set_concurrency(1)
    store.create("batch", {"provider": "openai"}, [
        dict(id=pid, name=pid, kind="tex", source=str(tmp_path / f"{pid}.tex"), root=None)
        for pid in ("occupy-slot", "target")
    ])
    occupied = Event()
    release = Event()
    starts = []

    def runner(home, pid):
        starts.append(pid)
        if pid == "occupy-slot":
            occupied.set()
            assert release.wait(5)
        store.finish(store.paper(pid), {"status": "failed", "reason": "offline runner finished"})

    try:
        with TestClient(create_app(tmp_path, runner=runner)) as client:
            assert occupied.wait(5)
            response = client.post("/api/papers/target/pause")
            assert response.status_code == 200 and response.json()["status"] == "paused"
            assert store.paper("target")["attempt"] == 1 and not store.work(store.paper("target")).exists()
            assert ("target", 1) not in store.pending_attempts()
            assert client.post("/api/papers/target/pause").json()["status"] == "paused"
            assert client.delete("/api/batches/batch").status_code == 409
            release.set()
            wait_for_status(store, "occupy-slot", "failed")
            assert starts == ["occupy-slot"]
    finally:
        release.set()

    with TestClient(create_app(tmp_path, runner=runner)) as restarted:
        detail = restarted.get("/api/batches/batch").json()
        assert next(p for p in detail["papers"] if p["id"] == "target")["status"] == "paused"
        assert starts == ["occupy-slot"]
        assert restarted.post("/api/papers/target/resume").json()["status"] == "queued"
        final = wait_for_status(store, "target", "failed")
        assert final["attempt"] == 1 and starts == ["occupy-slot", "target"]
        assert restarted.post("/api/papers/target/resume").status_code == 400


def test_running_pause_preserves_draft_checkpoint_and_does_not_replay_models(paper, tmp_path, monkeypatch):
    models, _, builds, _ = paper
    monkeypatch.setenv("OPENAI_API_KEY", "offline-fixture")
    store = Store(tmp_path)
    store.set_concurrency(1)
    settings = dict(provider="openai", base_url="https://offline.invalid", text_model="fixture", vision_model="fixture")
    store.create("batch", settings, [
        dict(id="target", name="target", kind="tex", source=str(tmp_path / "input" / "source.tex"), root=str(tmp_path / "input")),
    ])
    writer_entered = Event()
    release_writer = Event()
    first_pause_saved = Event()
    release_first_worker = Event()
    starts = []
    structured = models.structured

    def pause_during_write(schema, instructions, payload, image=None):
        if schema is Draft:
            writer_entered.set()
            assert release_writer.wait(5)
        return structured(schema, instructions, payload, image=image)

    monkeypatch.setattr(models, "structured", pause_during_write)

    def runner(home, pid):
        starts.append(pid)
        worker.process(home, pid)
        if len(starts) == 1:
            first_pause_saved.set()
            assert release_first_worker.wait(5)

    app = create_app(tmp_path, runner=runner)
    try:
        with TestClient(app) as client:
            assert writer_entered.wait(5)
            assert client.post("/api/papers/target/pause").json()["status"] == "pausing"
            assert store.paper("target")["status"] == "pausing"
            release_writer.set()
            assert first_pause_saved.wait(5)
            paused = wait_for_status(store, "target", "paused")
            checkpoint = store.work(paused) / "checkpoint.sqlite"
            assert checkpoint.is_file() and paused["result"]["draft"] and paused["result"]["draft_round"] == 1
            assert models.extracts == models.drafts == 1 and not builds

            # A resume can race with the previous worker's release of its dispatch slot.
            assert client.post("/api/papers/target/resume").json()["status"] == "queued"
            with app.state.dispatch.lock:
                assert app.state.dispatch.active == {("target", 1)}
                assert list(app.state.dispatch.queued) == [("target", 1)]
            release_first_worker.set()
            final = wait_for_status(store, "target", "completed")
            assert final["attempt"] == 1 and store.work(final) == checkpoint.parent
            assert starts == ["target", "target"]
            assert models.extracts == models.drafts == len(builds) == 1
    finally:
        release_writer.set()
        release_first_worker.set()


def test_batch_pause_and_resume_leave_individually_paused_and_other_papers_alone(tmp_path):
    store = Store(tmp_path)
    store.set_concurrency(1)

    def papers(*ids):
        return [dict(id=pid, name=pid, kind="tex", source=str(tmp_path / f"{pid}.tex"), root=None) for pid in ids]

    store.create("batch", {"provider": "openai"}, papers("running", "queued", "individual", "finished"))
    store.create("other-batch", {"provider": "openai"}, papers("other"))
    store.set_paper_paused("individual", True)
    store.finish(store.paper("finished"), {"status": "failed", "reason": "already finished"})
    entered = Event()
    settle = Event()
    starts = []

    def runner(home, pid):
        starts.append(pid)
        if pid == "running" and starts.count(pid) == 1:
            store.event(pid, 1, "draft", {"attempt": 8})
            entered.set()
            assert settle.wait(5)
            store.finish(store.paper(pid), {"status": "paused", "draft_round": 8})
        else:
            store.finish(store.paper(pid), {"status": "failed", "reason": "offline worker finished"})

    with TestClient(create_app(tmp_path, runner=runner)) as client:
        try:
            assert entered.wait(5)
            assert client.post("/api/batches/batch/pause").json() == {"id": "batch", "affected": 2}
            for pid, expected in {"running": "pausing", "queued": "paused", "individual": "paused",
                                  "finished": "failed", "other": "queued"}.items():
                assert store.paper(pid)["status"] == expected
            assert store.paper("running")["stage"] == "draft"
            assert client.get("/api/batches/batch").json()["batch_pause"] == {"pausing": 1, "paused": 1}
            assert client.post("/api/batches/batch/pause").json()["affected"] == 0
            assert client.post("/api/batches/batch/resume").status_code == 400
            assert client.post("/api/batches/missing/pause").status_code == 404
        finally:
            settle.set()
        assert wait_for_status(store, "running", "paused")["result"]["draft_round"] == 8
        wait_for_status(store, "other", "failed")
    assert starts == ["running", "other"]

    # Ownership and the previous attempt survive a new dispatcher process.
    with TestClient(create_app(tmp_path, runner=runner)) as restarted:
        assert restarted.get("/api/batches/batch").json()["batch_pause"] == {"paused": 2}
        assert restarted.post("/api/batches/batch/resume").json() == {"id": "batch", "affected": 2}
        assert restarted.post("/api/batches/batch/resume").json()["affected"] == 0
        for pid in ("running", "queued"):
            assert wait_for_status(store, pid, "failed")["attempt"] == 1
        assert store.paper("individual")["status"] == "paused"
        assert store.paper("finished")["reason"] == "already finished"
