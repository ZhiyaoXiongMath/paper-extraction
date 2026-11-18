"""SQLite-backed batches and snapshots of published artifacts."""
import hashlib
import io
import json
import sqlite3
import zipfile
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .settings import PROVIDERS, provider_settings
from .selection import SELECTION_VERSION
from .scope_reconciliation import stale_scope_result
from .editorial import confirmation_only

NAMES = ("introduction.tex", "introduction.pdf", "preamble-v13.tex")
CONTENT_REVIEW_VERSION = "source-packet-v2"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def dump(value):
    return json.dumps(value, ensure_ascii=False)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def legacy_content_failure(result):
    """Recognize exhausted content reviews whose final verdict predates the active rubric."""
    reviews = result.get("reviews", [])
    if (result.get("status") != "failed" or result.get("resumable") or not result.get("draft")
            or not result.get("candidates") or not reviews):
        return False
    last = reviews[-1]
    return (last.get("stage") == "review" and last.get("passed") is False
            and last.get("attempt") == 3 and last.get("review_version") != CONTENT_REVIEW_VERSION)


def stale_selection_failure(result):
    """Explicitly retry terminal legacy cases that predate whole-paper candidate selection."""
    empty_legacy = str(result.get("reason", "")).startswith(
        "Insufficient sourced history or current contributions for standard v13.")
    return (result.get("status") == "failed" and not result.get("resumable")
            and (result.get("selection_version") != SELECTION_VERSION or stale_scope_result(result))
            and (bool(result.get("candidates")) or empty_legacy))


def _selected_terminal(result):
    if (result.get("status") != "failed" or result.get("resumable") or not isinstance(result.get("draft"), dict)
            or not result.get("reviews") or result.get("selection_version") != SELECTION_VERSION):
        return False
    admitted = [c for c in result.get("candidates", []) if c.get("decision") == "admitted"]
    current_groups = {c.get("scope_group_id") or c.get("candidate_id") or index
                      for index, c in enumerate(admitted) if c["kind"] in {"current", "application"}}
    return (any(c["kind"] == "historical" for c in admitted)
            and 1 <= len(result["draft"].get("history", [])) <= 3 and 1 <= len(current_groups) <= 4)


def legacy_confirmation_failure(result):
    """Preserve a selected final draft when every old rejection explicitly confirmed it."""
    if not _selected_terminal(result):
        return False
    last = result["reviews"][-1]
    if (last.get("stage") != "review" or last.get("passed") is not False or last.get("route") != "draft"
            or last.get("attempt") != 3 or last.get("review_version") != CONTENT_REVIEW_VERSION
            or last.get("edit_contract_version") is not None or last.get("evidence_ids")):
        return False
    defects = last.get("defects", [])
    issues = last.get("issues", [])
    if not defects or len(defects) != len(issues):
        return False
    for item, issue in zip(defects, issues):
        if not isinstance(item, dict) or item.get("kind") not in {"incorrect_claim", "incorrect_attribution_or_locator"}:
            return False
        change = item.get("required_change")
        if not isinstance(change, str) or not confirmation_only(change):
            return False
        # Extra free-form instructions in a legacy issue must remain actionable.
        expected = (f"{item.get('draft_field')}: {change} (draft: {item.get('current_excerpt')}; "
                    f"original {item.get('evidence_id')}: {item.get('source_excerpt')})")
        if issue != expected:
            return False
    return True


def repairable_citation_draft(result, index):
    """Return a source-bound correction only for the exact exhausted locator failure."""
    if not _selected_terminal(result):
        return None
    last = result["reviews"][-1]
    issues = last.get("issues", [])
    if (last.get("stage") != "compile" or last.get("passed") is not False or last.get("attempt") != 3
            or not issues or any(not isinstance(i, str) or not i.endswith(
                r" needs a citation identifying its source location, e.g. \cite[p. 1]{source}.") for i in issues)):
        return None
    from .profile import bind_formal_citations
    original = result["draft"]
    corrected = bind_formal_citations(original, {r["id"]: r for r in index.catalog()})
    return corrected if corrected != original else None


def saved_writer_targets(result, index):
    """Recover an exhausted pre-patch writer only when its exact final edits are actionable."""
    from .writer_patch import WRITER_VERSION, repair_targets
    if not _selected_terminal(result) or result.get("writer_version") == WRITER_VERSION:
        return []
    last = result["reviews"][-1]
    if last.get("attempt") != 3 or (
            last.get("stage") == "review" and last.get("review_version") != CONTENT_REVIEW_VERSION):
        return []
    return repair_targets(result["draft"], last, {str(r["id"]) for r in index.catalog()})


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
              CREATE TABLE IF NOT EXISTS archived_reviews(
                paper_id TEXT REFERENCES papers, attempt INTEGER NOT NULL, result TEXT NOT NULL,
                PRIMARY KEY(paper_id,attempt));
              CREATE TABLE IF NOT EXISTS paper_configs(
                paper_id TEXT REFERENCES papers, attempt INTEGER NOT NULL, settings TEXT NOT NULL,
                PRIMARY KEY(paper_id,attempt));
              CREATE TABLE IF NOT EXISTS workbench_settings(
                name TEXT PRIMARY KEY, value TEXT NOT NULL);
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

    def concurrency(self, default=None):
        """Read the local preference, initializing it once from the first startup."""
        with self.db() as db:
            if default is not None:
                db.execute("INSERT OR IGNORE INTO workbench_settings VALUES('concurrency', ?)",
                           (str(default),))
            row = db.execute("SELECT value FROM workbench_settings WHERE name='concurrency'").fetchone()
        return int(row[0]) if row else None

    def set_concurrency(self, value):
        with self.db() as db:
            db.execute("""INSERT INTO workbench_settings VALUES('concurrency', ?)
                          ON CONFLICT(name) DO UPDATE SET value=excluded.value""", (str(value),))

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

    def paper_settings(self, paper):
        """Return the public settings frozen for this paper and attempt."""
        with self.db() as db:
            row = db.execute("""SELECT settings FROM paper_configs WHERE paper_id=? AND attempt<=?
                                ORDER BY attempt DESC LIMIT 1""", (paper["id"], paper["attempt"])).fetchone()
        return json.loads(row["settings"]) if row is not None else self.settings(paper["batch_id"])

    def work(self, p):
        work_attempt = p.get("result", {}).get("work_attempt", p["attempt"])
        return self.home/"runs"/p["batch_id"]/p["id"]/f"attempt-{work_attempt}"

    def pending(self):
        return [pid for pid, _ in self.pending_attempts()]

    def pending_attempts(self):
        with self.db() as db:
            return [tuple(row) for row in db.execute(
                "SELECT id,attempt FROM papers WHERE status IN ('queued','running') ORDER BY rowid")]

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
            papers = [dict(r) for r in db.execute("""SELECT p.*, COALESCE(
                (SELECT pc.settings FROM paper_configs pc WHERE pc.paper_id=p.id AND pc.attempt<=p.attempt
                 ORDER BY pc.attempt DESC LIMIT 1), b.settings) AS effective_settings
                FROM papers p JOIN batches b ON b.id=p.batch_id WHERE p.batch_id=? ORDER BY p.rowid""", (batch,))]
            events = db.execute("""SELECT e.* FROM events e JOIN papers p ON p.id=e.paper_id
                              WHERE p.batch_id=? AND e.attempt=p.attempt ORDER BY e.seq""",(batch,)).fetchall()
        for p in papers:
            effective = json.loads(p.pop("effective_settings"))
            for name in ("provider", "text_model", "vision_model"):
                p[name] = effective.get(name)
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

    def retry(self, pid, provider=None):
        prior = self.paper(pid)
        if prior["status"] != "failed":
            raise ValueError("仅失败任务可重试")
        previous = self.paper_settings(prior)
        effective_provider = provider if provider is not None else previous.get("provider")
        switched = provider is not None and provider != previous.get("provider")
        if provider is not None or effective_provider == "openai":
            updated = provider_settings(effective_provider, with_key=True)
            if not updated.pop("api_key", "").strip():
                raise ValueError("缺少 " + PROVIDERS[effective_provider][1] + "，请先配置模型")
        result = prior["result"]
        # Early v0.1 builds stored exceptions as "ValueError: ..." without a flag.
        # Their checkpoint still contains the failed node's starting state.
        interrupted = result.get("resumable", prior["reason"].startswith(
            ("ValueError: Model did not return valid ", "APIConnectionError:", "APITimeoutError:",
             "Insufficient sourced history or current contributions for standard v13.")))
        has_checkpoint = not switched and (self.work(prior)/"checkpoint.sqlite").is_file()
        redraft = has_checkpoint and (legacy_content_failure(result) or stale_selection_failure(result)
                                      or legacy_confirmation_failure(result))
        if has_checkpoint and not redraft:
            from .sources import SourceIndex
            from .saved_selection_recovery import saved_selection_for_work
            index = SourceIndex(Path(prior["source"]), Path(prior["root"]) if prior["root"] else None, self.work(prior))
            redraft = (repairable_citation_draft(result, index) is not None or bool(saved_writer_targets(result, index))
                       or saved_selection_for_work(self.work(prior), result, index) is not None)
        resume = has_checkpoint and (interrupted or redraft)
        state = {"work_attempt":result.get("work_attempt",prior["attempt"])} if resume else {}
        if redraft:
            state["retry_legacy_review"] = True
        with self.db() as db:
            row = db.execute("""UPDATE papers SET attempt=attempt+1,status='queued',stage='queued',
                          reason='',result=? WHERE id=? AND status='failed' AND attempt=?""",
                          (dump(state),pid,prior["attempt"]))
            if not row.rowcount:
                raise ValueError("仅失败任务可重试")
            db.execute("INSERT INTO archived_reviews(paper_id,attempt,result) VALUES(?,?,?)",
                       (pid, prior["attempt"], dump(result)))
            if switched:
                db.execute("INSERT INTO paper_configs(paper_id,attempt,settings) VALUES(?,?,?)",
                           (pid, prior["attempt"] + 1, dump(updated)))
        return effective_provider

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
