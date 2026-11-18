"""HTTP entry points for the local paper extraction workbench."""
import asyncio
import json
import subprocess
import sys
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock, Thread
from typing import Annotated, Literal
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.routing import APIRoute
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as FormHTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .importing import prepare_sources, stage
from .settings import PROVIDERS, data_home, probe, provider_settings, public_config, workbench_limits
from .sources import SourceIndex
from .store import Store


class PaperChoice(BaseModel):
    id: str
    entry: str


class BatchInput(BaseModel):
    import_id: str
    provider: Literal["openai","deepseek","zhipu"]
    papers: list[PaperChoice]


class RetryInput(BaseModel):
    provider: Literal["openai", "deepseek", "zhipu"] | None = None


class ConnectionTest(BaseModel):
    provider: Literal["openai","deepseek","zhipu"]
    mode: Literal["text","vision"]


class ConcurrencyChange(BaseModel):
    concurrency: int = Field(strict=True, gt=0)


class ImportRoute(APIRoute):
    def get_route_handler(self):
        handle = super().get_route_handler()

        async def receive_multipart(request: Request):
            limit = request.app.state.limits["max_import_files"]
            try:
                # FastAPI subsequently consumes this cached form and closes its files.
                # The JSON paths list is one text field and grows with the whole folder.
                await request.form(max_files=limit or float("inf"), max_part_size=sys.maxsize)
            except FormHTTPException as exc:
                if limit and exc.status_code == 400 and str(exc.detail).startswith("Too many files."):
                    raise HTTPException(400, f"单次导入最多允许 {limit} 个文件。请分批上传，或调高 "
                                            "PAPER_EXTRACTION_MAX_IMPORT_FILES 后重启服务。") from exc
                raise
            return await handle(request)

        return receive_multipart


def launch_worker(home, pid):
    subprocess.run([sys.executable,"-m","paper_extraction.worker",str(home),pid],check=True)


class Dispatcher:
    def __init__(self, store, runner, concurrency):
        self.store, self.runner = store, runner
        self._concurrency = concurrency
        self.queued = OrderedDict()
        self.active = set()
        self.lock = Lock()
        self.closed = False

    @property
    def concurrency(self):
        with self.lock:
            return self._concurrency

    def set_concurrency(self, value):
        if type(value) is not int or value < 1:
            raise ValueError("并发数必须是正整数")
        with self.lock:
            self.store.set_concurrency(value)
            self._concurrency = value
            self._start_ready()
            return self._concurrency

    def restore(self):
        self._enqueue(self.store.pending_attempts())

    def submit_new_batch(self, papers):
        self._enqueue((paper["id"], 1) for paper in papers)

    def submit(self, pid):
        paper = self.store.paper(pid)
        if paper["status"] in {"queued", "running"}:
            self._enqueue([(pid, paper["attempt"])])

    def _enqueue(self, tokens):
        with self.lock:
            if self.closed:
                return
            for token in tokens:
                if token not in self.active:
                    self.queued.setdefault(token, None)
            self._start_ready()

    def _start_ready(self):
        """Called under lock. Lower limits take effect as existing workers leave."""
        while not self.closed and self.queued and len(self.active) < self._concurrency:
            token, _ = self.queued.popitem(last=False)
            self.active.add(token)
            try:
                Thread(target=self._run, args=(token,), name=f"paper-worker-{token[0]}",
                       daemon=False).start()
            except RuntimeError as exc:
                self.active.remove(token)
                if self.active:
                    # Leave the paper at the head until an OS thread is released.
                    self.queued[token] = None
                    self.queued.move_to_end(token, last=False)
                    return
                self._mark_failed(token, exc)

    def _mark_failed(self, token, error):
        pid, attempt = token
        paper = self.store.paper(pid)
        if paper["attempt"] == attempt:
            self.store.finish(paper, {"status": "failed", "reason": f"工作进程退出：{type(error).__name__}"})

    def _run(self, token):
        try:
            self.runner(self.store.home, token[0])
        except Exception as exc:
            self._mark_failed(token, exc)
        finally:
            with self.lock:
                self.active.discard(token)
                self._start_ready()

    def close(self):
        with self.lock:
            self.closed = True
            self.queued.clear()


def create_app(home=None, runner=launch_worker):
    store = Store(home or data_home())
    limits = workbench_limits(concurrency=store.concurrency())
    @asynccontextmanager
    async def lifespan(app):
        app.state.dispatch = Dispatcher(store, runner, store.concurrency(default=limits["concurrency"]))
        app.state.dispatch.restore()
        try:
            yield
        finally:
            app.state.dispatch.close()

    app = FastAPI(title="Paper Extraction",version="0.1.0",lifespan=lifespan)
    app.state.limits = limits
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1","localhost","[::1]","testserver"])
    @app.middleware("http")
    async def browser_origin(req: Request, call_next):
        origin = req.headers.get("origin")
        if req.method not in {"GET","HEAD","OPTIONS"} and origin and urlsplit(origin).netloc != req.headers.get("host"):
            return JSONResponse({"detail":"请从本机工作台发起操作"}, status_code=403)
        return await call_next(req)

    @app.exception_handler(ValueError)
    async def bad_value(req, exc):
        return JSONResponse({"detail":str(exc)},status_code=400)
    @app.exception_handler(KeyError)
    async def missing(req, exc):
        return JSONResponse({"detail":exc.args[0]},status_code=404)

    @app.get("/api/config")
    def config():
        return {**public_config(), **limits, "concurrency": app.state.dispatch.concurrency}

    @app.patch("/api/config/concurrency")
    @app.post("/api/config/concurrency")
    def change_concurrency(body: ConcurrencyChange):
        return {"concurrency": app.state.dispatch.set_concurrency(body.concurrency)}

    @app.post("/api/config/test")
    def test_connection(body: ConnectionTest):
        return probe(body.provider,body.mode)

    def receive(files: Annotated[list[UploadFile],File()], paths: Annotated[str,Form()]):
        rel = json.loads(paths)
        if not isinstance(rel,list) or not all(isinstance(p,str) for p in rel):
            raise ValueError("文件路径列表无效")
        return stage(store.home,files,rel)

    app.router.add_api_route("/api/imports", receive, methods=["POST"], route_class_override=ImportRoute)

    @app.post("/api/batches",status_code=201)
    def create(body: BatchInput):
        settings = provider_settings(body.provider,with_key=True)
        if not settings.pop("api_key"):
            raise ValueError("缺少 " + PROVIDERS[body.provider][1] + "，请先配置模型")
        bid = uuid4().hex
        papers = prepare_sources(store.home,body.import_id,body.papers,bid)
        store.create(bid,settings,papers)
        app.state.dispatch.submit_new_batch(papers)
        return {"id":bid}

    @app.get("/api/batches")
    def batches():
        return store.summaries()

    @app.get("/api/batches/{bid}")
    def detail(bid: str):
        return store.describe(bid)

    @app.get("/api/batches/{bid}/events")
    async def events(bid: str, req: Request):
        store.settings(bid)
        async def stream():
            before, tick = None, 0
            while not await req.is_disconnected():
                with store.db() as db:
                    watermark = db.execute("""SELECT coalesce(max(e.seq),0) FROM papers p
                        LEFT JOIN events e ON e.paper_id=p.id WHERE p.batch_id=?""",(bid,)).fetchone()[0]
                    states = [tuple(r) for r in db.execute("SELECT id,status,attempt FROM papers WHERE batch_id=? ORDER BY rowid",(bid,))]
                signature = (watermark,states)
                if signature != before:
                    yield "event: update\ndata: " + json.dumps({"batch_id":bid,"sequence":watermark}) + "\n\n"
                    before = signature
                if all(row[1] in {"failed","completed"} for row in states):
                    return
                tick += 1
                if tick % 15 == 0:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(1)
        return StreamingResponse(stream(), media_type="text/event-stream",headers={"Cache-Control":"no-store"})

    @app.get("/api/papers/{pid}/evidence/{location}")
    def evidence(pid: str, location:str):
        p = store.paper(pid)
        return SourceIndex(Path(p["source"]),Path(p["root"]) if p["root"] else None,store.work(p)).read(location)

    @app.get("/api/papers/{pid}/source")
    def source(pid: str):
        p = store.paper(pid)
        if p["kind"] != "pdf":
            raise ValueError("TeX 来源可通过证据位置查看")
        return FileResponse(p["source"],media_type="application/pdf")

    @app.post("/api/papers/{pid}/retry")
    def retry(pid: str, body: RetryInput | None = None):
        provider = store.retry(pid, provider=body.provider if body else None)
        app.state.dispatch.submit(pid)
        return {"id": pid, "status": "queued", "provider": provider}

    @app.get("/api/papers/{pid}/artifacts/{name}")
    def artifact(pid:str,name:str):
        path = store.artifact(store.paper(pid),name)
        return FileResponse(path,media_type="application/pdf" if name.endswith(".pdf") else "text/plain; charset=utf-8")

    def zip_response(data,name):
        return Response(data, media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{name}.zip"'})

    @app.get("/api/batches/{bid}/download")
    def download(bid:str):
        return zip_response(store.archive(bid), "papers-"+bid[:10])

    @app.get("/api/papers/{pid}/download")
    def download_paper(pid:str):
        p = store.paper(pid)
        return zip_response(store.archive(p["batch_id"],pid),"introduction-"+pid[:10])

    static = Path(__file__).parent/"static"
    if static.is_dir():
        app.mount("/static",StaticFiles(directory=static),name="static")
    @app.get("/",include_in_schema=False)
    def index():
        if not (static/"index.html").is_file():
            raise HTTPException(503,"开发环境需要先运行前端构建；发行 wheel 已内置前端")
        return FileResponse(static/"index.html")
    return app
