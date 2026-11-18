"""One process per paper keeps PDFium and SQLite graph state isolated."""
import sys
import logging
from pathlib import Path

from filelock import FileLock

from .settings import provider_settings
from .store import Store
from .usage import usage_session


def process(home, pid):
    from .agent import run_paper

    store = Store(home)
    batch = store.claim_worker(pid)
    if batch is None:
        return
    lock = Path(home)/"runs"/batch/pid/"worker.lock"
    with FileLock(str(lock)):
        try:
            paper = store.paper(pid)
        except KeyError:
            return
        if paper["status"] not in {"queued","running","pausing"}:
            return
        settings = store.paper_settings(paper)
        def read_api_key():
            return provider_settings(settings["provider"], with_key=True)["api_key"]
        def pause_requested():
            return store.paper(pid)["status"] in {"pausing", "paused"}
        def count_draft_starts():
            # Count starts, including requests interrupted before LangGraph saves a draft.
            with store.db() as db:
                return db.execute("""SELECT count(*) FROM events WHERE paper_id=? AND attempt=? AND stage='draft'
                    AND json_extract(data, '$.mode') IN ('full', 'patch')""", (pid, paper["attempt"])).fetchone()[0]
        settings["api_key"] = read_api_key()
        def notify(stage, data):
            store.event(pid, paper["attempt"], stage, data)
        def save_draft(round_no, draft, case=None):
            try:
                store.record_draft(paper, round_no, draft, case)
            except Exception as exc:
                logging.getLogger(__name__).warning("Draft artifact storage failed (%s)", type(exc).__name__)
        try:
            if not settings["api_key"] and not pause_requested():
                raise ValueError("缺少独立模型 API Key，请在模型设置查看变量名")
            failure = paper["result"].get("previous_failure")
            start = {"message": "读取论文快照并恢复执行图", "provider": settings["provider"],
                     "text_model": settings.get("text_model"), "vision_model": settings.get("vision_model")}
            if failure:
                start.update(previous_failure=failure,
                             message=f"读取论文快照，复核第 {failure['attempt']} 次任务记录的 {failure['issue_count']} 项问题")
            notify("start", start)
            work = store.work(paper)
            with usage_session(work, pid, paper["attempt"], settings):
                result = run_paper(work, Path(paper["source"]), Path(paper["root"]) if paper["root"] else None,
                                   settings, notify, retry_legacy_review=paper["result"].get("retry_legacy_review", False),
                                   read_api_key=read_api_key, save_draft=save_draft, previous_failure=failure,
                                   pause_requested=pause_requested, count_draft_starts=count_draft_starts)
            store.finish(paper, result)
        except Exception as exc:
            reason = str(exc).replace(settings["api_key"], "[redacted]") if settings["api_key"] else str(exc)
            store.finish(paper, {"status":"failed","reason":reason[:1800] or type(exc).__name__})


if __name__ == "__main__":
    process(Path(sys.argv[1]), sys.argv[2])
