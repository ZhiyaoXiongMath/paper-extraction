"""One process per paper keeps PDFium and SQLite graph state isolated."""
import sys
from pathlib import Path

from filelock import FileLock

from .settings import provider_settings
from .store import Store


def process(home, pid):
    from .agent import run_paper

    store = Store(home)
    lock = Path(home)/"runs"/store.paper(pid)["batch_id"]/pid/"worker.lock"
    with FileLock(str(lock)):
        paper = store.paper(pid)
        if paper["status"] not in {"queued","running"}:
            return
        settings = store.settings(paper["batch_id"])
        private = provider_settings(settings["provider"], with_key=True)
        settings["api_key"] = private["api_key"]
        def notify(stage, data):
            store.event(pid, paper["attempt"], stage, data)
        try:
            if not settings["api_key"]:
                raise ValueError("缺少独立模型 API Key，请在模型设置查看变量名")
            notify("start", {"message": "读取论文快照并恢复执行图"})
            result = run_paper(store.work(paper), Path(paper["source"]),
                               Path(paper["root"]) if paper["root"] else None,
                               settings, notify)
            store.finish(paper, result)
        except Exception as exc:
            reason = str(exc).replace(settings["api_key"], "[redacted]") if settings["api_key"] else str(exc)
            store.finish(paper, {"status":"failed","reason":reason[:1800] or type(exc).__name__})


if __name__ == "__main__":
    process(Path(sys.argv[1]), sys.argv[2])
