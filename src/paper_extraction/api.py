"""HTTP entry points for the local paper extraction workbench."""
import asyncio
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal
from urllib.parse import urlsplit
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .importing import prepare_sources, stage
from .settings import PROVIDERS, data_home, env, probe, provider_settings, public_config
from .sources import SourceIndex
from .store import Store


class PaperChoice(BaseModel):
    id: str
    entry: str


class BatchInput(BaseModel):
    import_id: str
    provider: Literal["deepseek","zhipu"]
    papers: list[PaperChoice]


class ConnectionTest(BaseModel):
    provider: Literal["deepseek","zhipu"]
    mode: Literal["text","vision"]


def launch_worker(home, pid):
    subprocess.run([sys.executable,"-m","paper_extraction.worker",str(home),pid],check=True)


class Dispatcher:
    def __init__(self, store, runner):
        self.store, self.runner = store, runner
        count = max(1, int(env().get("PAPER_EXTRACTION_CONCURRENCY") or 2))
        self.pool = ThreadPoolExecutor(max_workers=count, thread_name_prefix="paper-worker")
        self.active = set()
        self.lock = Lock()

    def submit(self, pid):
        attempt = self.store.paper(pid)["attempt"]
        token = (pid,attempt)
        with self.lock:
            if token in self.active:
                return
            self.active.add(token)
        def run():
            try:
                self.runner(self.store.home,pid)
            except Exception as exc:
                p = self.store.paper(pid)
                if p["attempt"] == attempt:
                    self.store.finish(p, {"status":"failed","reason":f"工作进程退出：{type(exc).__name__}"})
            finally:
                with self.lock:
                    self.active.discard(token)
        self.pool.submit(run)

    def close(self):
        self.pool.shutdown(wait=False,cancel_futures=True)


def create_app(home=None, runner=launch_worker):
    store = Store(home or data_home())
    @asynccontextmanager
    async def lifespan(app):
        app.state.dispatch = Dispatcher(store,runner)
        for pid in store.pending():
            app.state.dispatch.submit(pid)
        try:
            yield
        finally:
            app.state.dispatch.close()

    app = FastAPI(title="Paper Extraction",version="0.1.0",lifespan=lifespan)
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
        return public_config()

    @app.post("/api/config/test")
    def test_connection(body: ConnectionTest):
        return probe(body.provider,body.mode)

    @app.post("/api/imports")
    def receive(files: Annotated[list[UploadFile],File()], paths: Annotated[str,Form()]):
        rel = json.loads(paths)
        if not isinstance(rel,list) or not all(isinstance(p,str) for p in rel):
            raise ValueError("文件路径列表无效")
        return stage(store.home,files,rel)

    @app.post("/api/batches",status_code=201)
    def create(body: BatchInput):
        settings = provider_settings(body.provider,with_key=True)
        if not settings.pop("api_key"):
            raise ValueError("缺少 " + PROVIDERS[body.provider][1] + "，请先配置模型")
        bid = uuid4().hex
        papers = prepare_sources(store.home,body.import_id,body.papers,bid)
        store.create(bid,settings,papers)
        for p in papers:
            app.state.dispatch.submit(p["id"])
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
    def retry(pid:str):
        store.retry(pid)
        app.state.dispatch.submit(pid)
        return {"id":pid,"status":"queued"}

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
