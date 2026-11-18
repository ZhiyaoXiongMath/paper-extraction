"""SQLite-backed batches and snapshots of published artifacts."""
import hashlib
import io
import json
import sqlite3
import zipfile
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

NAMES = ("introduction.tex", "introduction.pdf", "preamble-v13.tex")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def dump(value):
    return json.dumps(value, ensure_ascii=False)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


class Store:
    def __init__(self, home):
        self.home = Path(home).resolve()
        self.home.mkdir(parents=True, exist_ok=True)
        self.path = self.home / "jobs.sqlite3"
        with self.db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
              CREATE TABLE IF NOT EXISTS batches(
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL, settings TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS papers(
                id TEXT PRIMARY KEY, batch_id TEXT REFERENCES batches, name TEXT, kind TEXT,
                source TEXT, root TEXT, status TEXT, stage TEXT, reason TEXT DEFAULT '',
                attempt INTEGER DEFAULT 1, result TEXT DEFAULT '{}');
              CREATE TABLE IF NOT EXISTS events(
                seq INTEGER PRIMARY KEY, paper_id TEXT REFERENCES papers, attempt INTEGER,
                stage TEXT, time TEXT, data TEXT);
              CREATE INDEX IF NOT EXISTS events_paper ON events(paper_id,attempt,seq);
              CREATE INDEX IF NOT EXISTS papers_batch ON papers(batch_id);
            """)

    @contextmanager
    def db(self):
        with closing(sqlite3.connect(self.path, timeout=30)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            with db:
                yield db

    def create(self, batch, settings, papers):
        with self.db() as db:
            db.execute("INSERT INTO batches VALUES(?,?,?)", (batch, now(), dump(settings)))
            for p in papers:
                db.execute("""INSERT INTO papers(id,batch_id,name,kind,source,root,status,stage)
                          VALUES(?,?,?,?,?,?,'queued','queued')""",
                           (p["id"],batch,p["name"],p["kind"],p["source"],p["root"]))

    def paper(self, pid):
        with self.db() as db:
            row = db.execute("SELECT * FROM papers WHERE id=?", (pid,)).fetchone()
        if not row:
            raise KeyError("论文不存在")
        p = dict(row)
        p["result"] = json.loads(p["result"])
        return p

    def settings(self, batch):
        with self.db() as db:
            row = db.execute("SELECT settings FROM batches WHERE id=?", (batch,)).fetchone()
        if not row:
            raise KeyError("批次不存在")
        return json.loads(row["settings"])

    def work(self, p):
        work_attempt = p.get("result", {}).get("work_attempt", p["attempt"])
        return self.home/"runs"/p["batch_id"]/p["id"]/f"attempt-{work_attempt}"

    def pending(self):
        with self.db() as db:
            return [row[0] for row in db.execute("SELECT id FROM papers WHERE status IN ('queued','running') ORDER BY rowid")]

    def summaries(self):
        with self.db() as db:
            batches = [dict(r) for r in db.execute("SELECT * FROM batches ORDER BY rowid DESC")]
            totals = db.execute("SELECT batch_id,status,count(*) n FROM papers GROUP BY batch_id,status").fetchall()
        for b in batches:
            b["provider"] = json.loads(b.pop("settings"))["provider"]
            b["counts"] = {r["status"]: r["n"] for r in totals if r["batch_id"] == b["id"]}
            b["total"] = sum(b["counts"].values())
        return batches

    def describe(self, batch):
        head = next((b for b in self.summaries() if b["id"] == batch), None)
        if not head:
            raise KeyError("批次不存在")
        with self.db() as db:
            papers = [dict(r) for r in db.execute("SELECT * FROM papers WHERE batch_id=? ORDER BY rowid", (batch,))]
            events = db.execute("""SELECT e.* FROM events e JOIN papers p ON p.id=e.paper_id
                              WHERE p.batch_id=? AND e.attempt=p.attempt ORDER BY e.seq""",(batch,)).fetchall()
        for p in papers:
            result = json.loads(p.pop("result"))
            for key in ("root","source","batch_id"):
                p.pop(key)
            p["events"] = [dict(stage=r["stage"],time=r["time"],data=json.loads(r["data"])) for r in events if r["paper_id"] == p["id"]]
            p.update({k:result.get(k, []) for k in ("candidates","reviews","evidence_links")})
            p["draft"] = result.get("draft")
            p["links"] = p["evidence_links"]
        head["papers"] = papers
        return head

    def event(self, pid, attempt, stage, data):
        with self.db() as db:
            row = db.execute("UPDATE papers SET stage=?,status='running' WHERE id=? AND attempt=? AND status IN ('running','queued')",
                             (stage,pid,attempt))
            if row.rowcount:
                db.execute("INSERT INTO events(paper_id,attempt,stage,time,data) VALUES(?,?,?,?,?)",
                           (pid,attempt,stage,now(),dump(data)))

    def finish(self, p, result):
        result = {**result, "work_attempt":p["result"].get("work_attempt",p["attempt"])}
        status = result.get("status","failed")
        if status == "completed":
            directory = Path(result["result_dir"]).resolve()
            if not directory.is_relative_to(self.work(p)):
                raise ValueError("成品目录超出本次任务范围")
            hashes = {name:sha(directory/name) for name in NAMES}
            result = {**result,"hashes":hashes}
        else:
            status = "failed"
        with self.db() as db:
            db.execute("""UPDATE papers SET status=?,stage=?,reason=?,result=? WHERE id=? AND attempt=?
                          AND status IN ('running','queued')""",
                       (status, status, result.get("reason",""), dump(result), p["id"], p["attempt"]))

    def retry(self, pid):
        prior = self.paper(pid)
        result = prior["result"]
        # Early v0.1 builds stored exceptions as "ValueError: ..." without a flag.
        # Their checkpoint still contains the failed node's starting state.
        interrupted = result.get("resumable", prior["reason"].startswith(
            ("ValueError: Model did not return valid ", "APIConnectionError:", "APITimeoutError:")))
        resume = interrupted and (self.work(prior)/"checkpoint.sqlite").is_file()
        state = {"work_attempt":result.get("work_attempt",prior["attempt"])} if resume else {}
        with self.db() as db:
            row = db.execute("""UPDATE papers SET attempt=attempt+1,status='queued',stage='queued',
                          reason='',result=? WHERE id=? AND status='failed' AND attempt=?""",
                          (dump(state),pid,prior["attempt"]))
            if not row.rowcount:
                raise ValueError("仅失败任务可重试")

    def artifact(self, p, name):
        if name not in NAMES or p["status"] != "completed":
            raise ValueError("成品尚未就绪")
        result = p["result"]
        path = Path(result["result_dir"])/name
        if not path.is_file() or sha(path) != result["hashes"][name]:
            raise ValueError("成品与已登记版本不匹配")
        return path

    def archive(self, batch, pid=None):
        summary = self.describe(batch)
        output = io.BytesIO()
        rows = []
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for i, info in enumerate(summary["papers"],1):
                if pid and info["id"] != pid:
                    continue
                row = {k:info[k] for k in ("id","name","status","reason")}
                if info["status"] == "completed":
                    p = self.paper(info["id"])
                    prefix = "" if pid else f"{i:03d}-{p['name']}/"
                    for name in NAMES:
                        archive.write(self.artifact(p,name), prefix+name)
                    row["sha256"] = p["result"]["hashes"]
                elif pid:
                    raise ValueError("成品尚未就绪")
                rows.append(row)
            if not pid:
                archive.writestr("manifest.json", dump({"batch":batch,"papers":rows}))
        return output.getvalue()
