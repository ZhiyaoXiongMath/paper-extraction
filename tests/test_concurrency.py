"""Exercise live paper-worker limits with real HTTP and SQLite, without model calls."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from time import monotonic, sleep

from fastapi.testclient import TestClient

from paper_extraction.api import create_app
from paper_extraction.store import Store


def test_fill_grow_shrink_and_persist_between_starts(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_EXTRACTION_CONCURRENCY", "1")
    store = Store(tmp_path)
    papers = [dict(id=str(i), name=str(i), kind="tex", source="source.tex", root=None) for i in range(5)]
    store.create("fixture", {"provider": "zhipu"}, papers)
    entered = {p["id"]: Event() for p in papers}
    release = {p["id"]: Event() for p in papers}
    starts, active, peak = [], set(), []
    lock = Lock()

    def runner(home, pid):
        with lock:
            starts.append(pid)
            active.add(pid)
            peak.append(len(active))
            entered[pid].set()
        try:
            assert release[pid].wait(12)
            current = store.paper(pid)
            store.finish(current, {"status": "failed", "reason": "terminal test fixture"})
        finally:
            with lock:
                active.remove(pid)

    app = create_app(tmp_path, runner=runner)
    try:
        with TestClient(app) as client:
            assert entered["0"].wait(5)
            assert client.get("/api/config").json()["concurrency"] == 1
            assert starts == ["0"]
            assert client.patch("/api/config/concurrency", json={"concurrency": 3}).json() == {"concurrency": 3}
            assert entered["1"].wait(5) and entered["2"].wait(5)
            assert set(starts) == {"0", "1", "2"}
            app.state.dispatch.submit("2")  # The active attempt is already reserved.
            assert client.post("/api/config/concurrency", json={"concurrency": 1}).json() == {"concurrency": 1}
            assert client.get("/api/config").json()["concurrency"] == 1
            release["1"].set()
            release["2"].set()
            deadline = monotonic() + 5
            while True:
                with app.state.dispatch.lock:
                    if app.state.dispatch.active == {("0", 1)}:
                        break
                assert monotonic() < deadline
                sleep(.01)
            assert not entered["3"].is_set() and not entered["4"].is_set()
            release["0"].set()
            assert entered["3"].wait(5)
            assert not entered["4"].is_set()
            release["3"].set()
            assert entered["4"].wait(5)
            release["4"].set()
            assert max(peak) == 3 and len(starts) == 5
            monkeypatch.setenv("PAPER_EXTRACTION_CONCURRENCY", "invalid after initial setup")
    finally:
        for event in release.values():
            event.set()

    deadline = monotonic() + 5
    while store.pending():
        assert monotonic() < deadline
        sleep(.01)
    with TestClient(create_app(tmp_path, runner=runner)) as restarted:
        assert restarted.get("/api/config").json()["concurrency"] == 1
        assert starts.count("4") == 1


def test_positive_integers_only_and_no_arbitrary_upper_bound(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_EXTRACTION_CONCURRENCY", "3")
    with TestClient(create_app(tmp_path)) as client:
        for invalid in (0, -1, True, "4", 3.25, None):
            response = client.patch("/api/config/concurrency", json={"concurrency": invalid})
            assert response.status_code == 422
        assert client.get("/api/config").json()["concurrency"] == 3
        large = 2 ** 70
        assert client.patch("/api/config/concurrency", json={"concurrency": large}).json() == {"concurrency": large}
    monkeypatch.setenv("PAPER_EXTRACTION_CONCURRENCY", "1")
    with TestClient(create_app(tmp_path)) as restarted:
        assert restarted.get("/api/config").json()["concurrency"] == large


def test_failed_persistence_cannot_apply_an_uncommitted_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPER_EXTRACTION_CONCURRENCY", "2")
    app = create_app(tmp_path)
    with TestClient(app, raise_server_exceptions=False) as client:
        def fail(value):
            raise OSError("injected settings write failure")
        monkeypatch.setattr(app.state.dispatch.store, "set_concurrency", fail)
        assert client.patch("/api/config/concurrency", json={"concurrency": 4}).status_code == 500
        assert client.get("/api/config").json()["concurrency"] == 2
        assert Store(tmp_path).concurrency() == 2


def test_overlapping_hot_edits_commit_and_apply_in_the_same_order(tmp_path, monkeypatch):
    app = create_app(tmp_path)
    with TestClient(app) as client, ThreadPoolExecutor(max_workers=2) as calls:
        dispatcher = app.state.dispatch
        first_writing, second_arrived, resume_first = Event(), Event(), Event()
        writes = []
        write_setting = dispatcher.store.set_concurrency
        apply_setting = dispatcher.set_concurrency

        def observed_update(n):
            if n == 7:
                second_arrived.set()
            return apply_setting(n)

        def observed_write(n):
            if n == 5:
                first_writing.set()
                assert resume_first.wait(8)
            write_setting(n)
            writes.append(n)

        monkeypatch.setattr(dispatcher, "set_concurrency", observed_update)
        monkeypatch.setattr(dispatcher.store, "set_concurrency", observed_write)
        first = calls.submit(client.patch, "/api/config/concurrency", json={"concurrency": 5})
        try:
            assert first_writing.wait(5)
            second = calls.submit(client.patch, "/api/config/concurrency", json={"concurrency": 7})
            assert second_arrived.wait(5)
        finally:
            resume_first.set()
        assert first.result(timeout=5).json()["concurrency"] == 5
        assert second.result(timeout=5).json()["concurrency"] == 7
        assert writes == [5, 7]
        assert Store(tmp_path).concurrency() == client.get("/api/config").json()["concurrency"] == 7

