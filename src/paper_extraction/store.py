"""SQLite-backed batches and snapshots of published artifacts."""
import hashlib
import io
import json
import logging
import re
import shutil
import sqlite3
import zipfile
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

from .settings import PROVIDERS, apply_reasoning_preferences, provider_settings
from .selection import SELECTION_VERSION
from .scope_reconciliation import saved_count_exception, stale_scope_result
from .editorial import confirmation_only
from .draft_artifacts import FILENAMES, capture_files, draft_digest, saved_rounds, snapshot_dir
from .exporting import batch_export_directory, extraction_basename, standalone_tex

NAMES = ("introduction.tex", "introduction.pdf", "preamble-v13.tex")
CONTENT_REVIEW_VERSION = "source-packet-v2"


class BatchStillActive(ValueError):
    """A batch cannot be removed while any of its papers is active."""


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


def resolved_renderer_failure(result):
    """Recognize an approved, unchanged draft rejected solely by an obsolete TeX allowlist."""
    checks = result.get("reviews") or []
    if result.get("status") != "failed" or result.get("resumable") or len(checks) < 2 or not result.get("draft"):
        return False
    review, compile_check = checks[-2:]
    round_no = compile_check.get("attempt")
    if (compile_check.get("stage") != "compile" or compile_check.get("passed") is not False
            or review.get("stage") != "review" or review.get("route") != "pass" or review.get("passed") is not True
            or review.get("issues") or review.get("defects") or not round_no or review.get("attempt") != round_no
            or result.get("draft_round", result.get("attempt", round_no)) != round_no):
        return False
    issues = compile_check.get("issues") or []
    if (len(issues) != 1 or result.get("reason") != issues[0]
            or not re.fullmatch(r"[^\r\n]+ uses unsupported TeX command \\(?:[A-Za-z]+|[^\s])\. Use standard notation\.", issues[0])):
        return False
    from .profile import render_tex
    try:
        render_tex(result["draft"])
    except (TypeError, ValueError):
        return False
    return True


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
            and 1 <= len(result["draft"].get("history", [])) <= 3 and bool(current_groups))


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
              CREATE TABLE IF NOT EXISTS batch_names(
                batch_id TEXT PRIMARY KEY REFERENCES batches(id) ON DELETE CASCADE,
                name TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS papers(
                id TEXT PRIMARY KEY, batch_id TEXT REFERENCES batches, name TEXT, kind TEXT,
                source TEXT, root TEXT, status TEXT, stage TEXT, reason TEXT DEFAULT '',
                attempt INTEGER DEFAULT 1, result TEXT DEFAULT '{}',
                paused_by_batch INTEGER NOT NULL DEFAULT 0);
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
              CREATE TABLE IF NOT EXISTS deleted_batch_files(batch_id TEXT PRIMARY KEY);
              CREATE TABLE IF NOT EXISTS draft_artifacts(
                paper_id TEXT REFERENCES papers, attempt INTEGER NOT NULL, work_attempt INTEGER NOT NULL,
                round INTEGER NOT NULL, draft_sha256 TEXT NOT NULL, files TEXT NOT NULL, restored INTEGER NOT NULL,
                recompiled INTEGER,
                PRIMARY KEY(paper_id,attempt,round,draft_sha256));
              CREATE INDEX IF NOT EXISTS events_paper ON events(paper_id,attempt,seq);
              CREATE INDEX IF NOT EXISTS papers_batch ON papers(batch_id);
            """)
            db.execute("BEGIN IMMEDIATE")
            if "recompiled" not in {column["name"] for column in db.execute("PRAGMA table_info(draft_artifacts)")}:
                db.execute("ALTER TABLE draft_artifacts ADD COLUMN recompiled INTEGER")
            if "paused_by_batch" not in {column["name"] for column in db.execute("PRAGMA table_info(papers)")}:
                db.execute("ALTER TABLE papers ADD COLUMN paused_by_batch INTEGER NOT NULL DEFAULT 0")

    @contextmanager
    def db(self):
        with closing(sqlite3.connect(self.path, timeout=30)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            with db:
                yield db

    def create(self, batch, settings, papers, *, name=None):
        with self.db() as db:
            inserted = db.execute("""INSERT INTO batches SELECT ?,?,? WHERE NOT EXISTS
                                    (SELECT 1 FROM deleted_batch_files WHERE batch_id=?)""",
                                  (batch, now(), dump(settings), batch))
            if not inserted.rowcount:
                raise ValueError("该批次编号的旧文件尚在清理")
            if name is not None:
                db.execute("INSERT INTO batch_names(batch_id,name) VALUES(?,?)", (batch, name))
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

    def reasoning_efforts(self, provider):
        with self.db() as db:
            row = db.execute("SELECT value FROM workbench_settings WHERE name=?",
                             (f"{provider}_reasoning_efforts",)).fetchone()
        return json.loads(row[0]) if row else {}

    def set_reasoning_efforts(self, provider, settings):
        with self.db() as db:
            # Merge only the submitted modes so different tabs can change text and vision independently.
            db.execute("""INSERT INTO workbench_settings VALUES(?, ?)
                          ON CONFLICT(name) DO UPDATE SET value=json_patch(value, excluded.value)""",
                       (f"{provider}_reasoning_efforts", dump(settings)))

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

    def pending_attempts(self, batch=None):
        with self.db() as db:
            return [tuple(row) for row in db.execute(
                """SELECT id,attempt FROM papers WHERE status IN ('queued','running','pausing')
                   AND (? IS NULL OR batch_id=?) ORDER BY rowid""", (batch, batch))]

    def claim_worker(self, pid):
        # Make a launched worker visible before it can create files. Concurrent pauses
        # then wait for its checkpoint, and already paused or deleted rows cannot start.
        with self.db() as db:
            row = db.execute("""UPDATE papers SET status=CASE WHEN status='queued' THEN 'running' ELSE status END
                               WHERE id=? AND status IN ('queued','running','pausing')
                               RETURNING batch_id""", (pid,)).fetchone()
        return row["batch_id"] if row else None

    @staticmethod
    def _write_pause_events(db, changes):
        messages = {"pausing": "当前环节完成后暂停", "paused": "已暂停", "queued": "从检查点继续"}
        timestamp = now()
        db.executemany("INSERT INTO events(paper_id,attempt,stage,time,data) VALUES(?,?,?,?,?)",
                       ((pid, attempt, status, timestamp, dump({"message": messages[status]}))
                        for pid, attempt, status in changes))

    def set_paper_paused(self, pid, paused):
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute("SELECT status,stage,attempt FROM papers WHERE id=?", (pid,)).fetchone()
            if current is None:
                raise KeyError("论文不存在")
            status = current["status"]
            if paused and status in {"pausing", "paused"}:
                return status, current["attempt"]
            if not paused and status in {"queued", "running"}:
                return status, current["attempt"]
            if paused and status in {"queued", "running"}:
                new_status = "pausing" if status == "running" else "paused"
            elif not paused and status == "paused":
                new_status = "queued"
            else:
                raise ValueError("当前状态无法暂停或继续")
            db.execute("UPDATE papers SET status=?,stage=?,reason=?,paused_by_batch=0 WHERE id=?",
                       (new_status, current["stage"] if new_status == "pausing" else new_status, "", pid))
            self._write_pause_events(db, [(pid, current["attempt"], new_status)])
            return new_status, current["attempt"]

    def set_batch_paused(self, batch, paused):
        """Change eligible papers together, preserving papers paused individually."""
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            if not db.execute("SELECT 1 FROM batches WHERE id=?", (batch,)).fetchone():
                raise KeyError("批次不存在")
            if paused:
                rows = db.execute("""UPDATE papers SET
                    status=CASE status WHEN 'running' THEN 'pausing' ELSE 'paused' END,
                    stage=CASE status WHEN 'running' THEN stage ELSE 'paused' END,
                    reason='',paused_by_batch=1 WHERE batch_id=? AND status IN ('queued','running')
                    RETURNING id,attempt,status""", (batch,)).fetchall()
            else:
                if db.execute("""SELECT 1 FROM papers WHERE batch_id=? AND status='pausing'
                                  AND paused_by_batch=1 LIMIT 1""", (batch,)).fetchone():
                    raise ValueError("批次正在暂停，当前环节保存后可继续")
                rows = db.execute("""UPDATE papers SET status='queued',stage='queued',reason='',paused_by_batch=0
                    WHERE batch_id=? AND status='paused' AND paused_by_batch=1
                    RETURNING id,attempt,status""", (batch,)).fetchall()
            changes = [tuple(row) for row in rows]
            self._write_pause_events(db, changes)
            return changes

    def batch_name(self, batch):
        with self.db() as db:
            row = db.execute("""SELECT n.name FROM batches b LEFT JOIN batch_names n ON n.batch_id=b.id
                                WHERE b.id=?""", (batch,)).fetchone()
        if row is None:
            raise KeyError("批次不存在")
        return row["name"]

    def rename_batch(self, batch, name):
        with self.db() as db:
            saved = db.execute("""INSERT INTO batch_names(batch_id,name)
                                  SELECT id, ? FROM batches WHERE id=?
                                  ON CONFLICT(batch_id) DO UPDATE SET name=excluded.name""", (name, batch))
            if not saved.rowcount:
                raise KeyError("批次不存在")

    def _batch_directory(self, batch):
        runs = self.home / "runs"
        path = runs / batch
        if path.parent != runs or path.resolve() != path or not batch or batch in {".", ".."}:
            raise ValueError("批次文件目录无效")
        return path

    def delete_finished_batches(self, *, batch_ids=None):
        return self._delete_batches(batch_ids)

    def delete_batch(self, batch):
        return self._delete_batches([batch], single_batch=batch)

    def _delete_batches(self, batch_ids, *, single_batch=None):
        selected = dump(batch_ids) if batch_ids is not None else None
        allowed = ["completed", "failed", "cancelled"]
        if single_batch is not None:
            allowed.append("paused")
        with self.db() as db:
            # Check eligibility in the same transaction as deletion, serialized with resume and retry.
            db.execute("BEGIN IMMEDIATE")
            eligible = db.execute("""SELECT b.id FROM batches b WHERE
                    (? IS NULL OR b.id IN (SELECT value FROM json_each(?)))
                    AND NOT EXISTS (SELECT 1 FROM papers p WHERE p.batch_id=b.id
                                    AND coalesce(p.status,'') NOT IN (SELECT value FROM json_each(?)))
                    ORDER BY b.rowid""", (selected, selected, dump(allowed))).fetchall()
            ids = [row["id"] for row in eligible]
            if single_batch is not None and not ids:
                if db.execute("SELECT 1 FROM batches WHERE id=?", (single_batch,)).fetchone():
                    raise BatchStillActive("请等批次中的论文全部暂停或结束后再删除。")
                raise KeyError("批次不存在")
            for bid in ids:
                self._batch_directory(bid)
                for table in ("events", "archived_reviews", "paper_configs", "draft_artifacts"):
                    db.execute(f"DELETE FROM {table} WHERE paper_id IN (SELECT id FROM papers WHERE batch_id=?)", (bid,))
                db.execute("DELETE FROM papers WHERE batch_id=?", (bid,))
                db.execute("DELETE FROM batch_names WHERE batch_id=?", (bid,))
                db.execute("DELETE FROM batches WHERE id=?", (bid,))
                db.execute("INSERT INTO deleted_batch_files VALUES(?)", (bid,))
        self.clean_deleted_batch_files()
        return ids

    def clean_deleted_batch_files(self):
        """Finish committed deletions, including files held open during an earlier request."""
        with self.db() as db:
            if not db.execute("SELECT 1 FROM deleted_batch_files LIMIT 1").fetchone():
                return
        # File deletion can be slow. Other workers may continue writing to SQLite throughout.
        with FileLock(self.home / ".batch-cleanup.lock"):
            with self.db() as db:
                pending = db.execute("SELECT batch_id FROM deleted_batch_files").fetchall()
            for row in pending:
                bid = row["batch_id"]
                try:
                    directory = self._batch_directory(bid)
                    if directory.exists():
                        shutil.rmtree(directory)
                except (OSError, ValueError) as exc:
                    logging.getLogger(__name__).warning("Pending cleanup for batch %s: %s", bid, exc)
                    continue
                with self.db() as db:
                    db.execute("DELETE FROM deleted_batch_files WHERE batch_id=?", (bid,))

    def summaries(self):
        with self.db() as db:
            batches = [dict(r) for r in db.execute("""SELECT b.*, n.name FROM batches b
                        LEFT JOIN batch_names n ON n.batch_id=b.id ORDER BY b.rowid DESC""")]
            totals = db.execute("SELECT batch_id,status,count(*) n,sum(paused_by_batch) owned FROM papers GROUP BY batch_id,status").fetchall()
        for b in batches:
            b["provider"] = json.loads(b.pop("settings"))["provider"]
            b["counts"] = {r["status"]: r["n"] for r in totals if r["batch_id"] == b["id"]}
            b["batch_pause"] = {r["status"]: r["owned"] for r in totals
                                if r["batch_id"] == b["id"] and r["status"] in {"pausing", "paused"} and r["owned"]}
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
            artifacts = db.execute("""SELECT a.* FROM draft_artifacts a JOIN papers p ON p.id=a.paper_id
                                      WHERE p.batch_id=? ORDER BY a.attempt DESC,a.round DESC,a.rowid DESC""",
                                   (batch,)).fetchall()
            historical = db.execute("""SELECT ar.paper_id,ar.attempt,json_extract(ar.result,'$.status') AS status,
                                       json_extract(ar.result,'$.reason') AS reason
                                       FROM archived_reviews ar JOIN papers p ON p.id=ar.paper_id WHERE p.batch_id=?""",
                                    (batch,)).fetchall()
            outcomes = {(r["paper_id"], r["attempt"]): (r["status"], r["reason"]) for r in historical}
        for p in papers:
            effective = json.loads(p.pop("effective_settings"))
            p.pop("paused_by_batch")
            for name in ("provider", "text_model", "vision_model"):
                p[name] = effective.get(name)
            result = json.loads(p.pop("result"))
            for key in ("root","source","batch_id"):
                p.pop(key)
            p["events"] = [dict(stage=r["stage"],time=r["time"],data=json.loads(r["data"])) for r in events if r["paper_id"] == p["id"]]
            p.update({k:result.get(k, []) for k in ("candidates","reviews","evidence_links")})
            p["draft"] = result.get("draft")
            p["profile_release"] = result.get("profile_release")
            if result.get("delivery"):
                p["delivery"] = result["delivery"]
            p["links"] = p["evidence_links"]
            versions = [a for a in artifacts if a["paper_id"] == p["id"]]
            p["draft_artifacts"] = self._describe_drafts(p, result, versions, outcomes)
        head["papers"] = papers
        return head

    def event(self, pid, attempt, stage, data):
        with self.db() as db:
            row = db.execute("""UPDATE papers SET stage=CASE WHEN status='pausing' THEN stage ELSE ? END,
                             status=CASE WHEN status='pausing' THEN status ELSE 'running' END
                             WHERE id=? AND attempt=? AND status IN ('running','queued','pausing')""",
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
        elif status != "paused":
            status = "failed"
        with self.db() as db:
            db.execute("""UPDATE papers SET status=?,stage=?,reason=?,result=?,
                          paused_by_batch=CASE WHEN ?='paused' THEN paused_by_batch ELSE 0 END WHERE id=? AND attempt=?
                          AND status IN ('running','queued','pausing')""",
                       (status, status, result.get("reason",""), dump(result), status, p["id"], p["attempt"]))
            if status == "paused":
                db.execute("""INSERT INTO events(paper_id,attempt,stage,time,data)
                              SELECT id,attempt,'paused',?,? FROM papers WHERE id=? AND attempt=? AND status='paused'""",
                           (now(), dump({"message": "节点检查点已保存"}), p["id"], p["attempt"]))

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
            if switched:
                apply_reasoning_preferences(updated, self.reasoning_efforts(effective_provider))
        result = prior["result"]
        # Early v0.1 builds stored exceptions as "ValueError: ..." without a flag.
        # Their checkpoint still contains the failed node's starting state.
        interrupted = result.get("resumable", prior["reason"].startswith(
            ("ValueError: Model did not return valid ", "APIConnectionError:", "APITimeoutError:",
             "Insufficient sourced history or current contributions for standard v13.")))
        has_checkpoint = not switched and (self.work(prior)/"checkpoint.sqlite").is_file()
        from .repair_progress import legacy_quota_stop, stalled_saved_repair
        redraft = has_checkpoint and (result.get("draft_limit_reached") or result.get("technical_build_failure")
                                      or saved_count_exception(result, SELECTION_VERSION) or stalled_saved_repair(result)
                                      or resolved_renderer_failure(result) or legacy_quota_stop(result)
                                      or legacy_content_failure(result) or stale_selection_failure(result)
                                      or legacy_confirmation_failure(result))
        if has_checkpoint and not redraft:
            from .sources import SourceIndex
            from .saved_selection_recovery import saved_selection_for_work
            index = SourceIndex(Path(prior["source"]), Path(prior["root"]) if prior["root"] else None, self.work(prior))
            redraft = (repairable_citation_draft(result, index) is not None or bool(saved_writer_targets(result, index))
                       or saved_selection_for_work(self.work(prior), result, index) is not None)
        resume = has_checkpoint and (interrupted or redraft)
        state = {"work_attempt":result.get("work_attempt",prior["attempt"])} if resume else {}
        if not resume:
            from .retry_context import previous_failure
            context = previous_failure(result, prior["attempt"])
            if context:
                state["previous_failure"] = context
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

    def export_tex(self, paper):
        tex = self.artifact(paper, "introduction.tex").read_bytes()
        return standalone_tex(tex, lambda: self.artifact(paper, "preamble-v13.tex").read_bytes())

    @staticmethod
    def _describe_drafts(p, result, versions, outcomes):
        digest = draft_digest(result["draft"]) if versions and result.get("draft") else None
        current = next((v for v in versions if v["attempt"] == p["attempt"] and (
            (digest and digest == v["draft_sha256"] and result.get("draft_round", v["round"]) == v["round"])
            or (not digest and p["status"] in {"running", "queued"}))), None)
        output = []
        for row in versions:
            metadata = {k: row[k] for k in ("attempt", "work_attempt", "round", "draft_sha256")}
            status, reason = outcomes.get((p["id"], row["attempt"]), (p["status"], p.get("reason") or ""))
            metadata.update(attempt_status=status, attempt_reason=reason or "")
            prefix = f'/api/papers/{p["id"]}/drafts/{row["attempt"]}/{row["round"]}/{row["draft_sha256"]}'
            metadata.update(current=row is current, restored=bool(row["restored"]), recompiled=bool(row["recompiled"]), files={
                kind: {"url": prefix + "/" + FILENAMES[kind] + "?sha256=" + hash_, "sha256": hash_}
                for kind, hash_ in json.loads(row["files"]).items()
            })
            output.append(metadata)
        return output

    def record_draft(self, paper, round_no, draft, case=None, *, restored=False):
        if not draft or type(round_no) is not int or round_no <= 0:
            return
        attempt = paper["attempt"]
        work_attempt = paper.get("result", {}).get("work_attempt", attempt)
        digest = draft_digest(draft)
        files, compiled_tex = capture_files(self.work(paper), draft, case)
        recompiled = False
        if restored and "pdf" in files and case:
            provenance_file = Path(case) / "draft-recovery.json"
            if provenance_file.is_file():
                provenance = json.loads(provenance_file.read_text("utf-8"))
                recompiled = provenance.get("kind") == "local_recompile" and provenance.get("draft_sha256") == digest
        identity = (paper["id"], attempt, round_no, digest)
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("""SELECT files FROM draft_artifacts
                                     WHERE paper_id=? AND attempt=? AND round=? AND draft_sha256=?""", identity).fetchone()
            if existing:
                # Keep an already served file immutable when compilation is replayed.
                stored = json.loads(existing["files"])
                merged = {**files, **stored}
                if compiled_tex:
                    merged["tex"] = files["tex"]
                recompiled = recompiled and files.get("pdf") == merged.get("pdf")
                db.execute("""UPDATE draft_artifacts SET files=?,restored=MAX(restored,?),recompiled=MAX(COALESCE(recompiled,0),?)
                              WHERE paper_id=? AND attempt=? AND round=? AND draft_sha256=?""",
                           (dump(merged), int(restored), int(recompiled), *identity))
            else:
                db.execute("INSERT INTO draft_artifacts VALUES(?,?,?,?,?,?,?,?)",
                           (paper["id"], attempt, work_attempt, round_no, digest, dump(files), int(restored), int(recompiled)))

    def draft_artifact(self, pid, attempt, round_no, digest, name, expected_hash=None):
        kind = next((k for k, filename in FILENAMES.items() if filename == name), None)
        with self.db() as db:
            row = db.execute("""SELECT work_attempt,files FROM draft_artifacts
                                WHERE paper_id=? AND attempt=? AND round=? AND draft_sha256=?""",
                             (pid, attempt, round_no, digest)).fetchone()
        if not row or kind not in json.loads(row["files"]):
            raise KeyError("该版本没有此草稿文件")
        file_hash = json.loads(row["files"])[kind]
        if expected_hash is not None and expected_hash != file_hash:
            raise KeyError("该文件哈希不属于登记的草稿版本")
        paper = self.paper(pid)
        work = self.work({**paper, "result": {"work_attempt": row["work_attempt"]}})
        target = (snapshot_dir(work) / f"{file_hash}.{kind}").resolve()
        if not target.is_relative_to(work) or not target.is_file() or sha(target) != file_hash:
            raise ValueError("草稿文件与登记的论文、尝试或稿件版本不匹配")
        return target

    def publish_reviewed(self, prior, result):
        """Atomically archive the exact rejected attempt and register its inspected successor."""
        new_attempt = prior["attempt"] + 1
        work_attempt = prior["result"].get("work_attempt", prior["attempt"])
        work = self.work(prior)
        case = Path(result["result_dir"]).resolve()
        if not case.is_relative_to(work) or any(sha(case / n) != result["hashes"][n] for n in NAMES):
            raise ValueError("文件已改变或不属于该论文的工作目录")
        result = {**result, "work_attempt": work_attempt}
        digest = draft_digest(result["draft"])
        files, _ = capture_files(work, result["draft"], case)
        if any(files.get(kind) != result["hashes"][name] for kind, name in FILENAMES.items()):
            raise ValueError("未能精确留存待发布的 TeX 和 PDF")
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute("SELECT attempt,status,result FROM papers WHERE id=?", (prior["id"],)).fetchone()
            if (not current or current["attempt"] != prior["attempt"] or current["status"] != "failed"
                    or json.loads(current["result"]) != prior["result"]):
                raise ValueError("该论文的保存结果已变化，请重新确认尝试与稿件")
            db.execute("INSERT INTO archived_reviews(paper_id,attempt,result) VALUES(?,?,?)",
                       (prior["id"], prior["attempt"], current["result"]))
            db.execute("""UPDATE papers SET attempt=?,status='completed',stage='completed',reason='',result=? WHERE id=?""",
                       (new_attempt, dump(result), prior["id"]))
            db.execute("INSERT INTO draft_artifacts VALUES(?,?,?,?,?,?,?,?)",
                       (prior["id"], new_attempt, work_attempt, result["draft_round"], digest, dump(files), 0, 0))
            db.execute("INSERT INTO events(paper_id,attempt,stage,time,data) VALUES(?,?,?,?,?)",
                       (prior["id"], new_attempt, "publish", now(), dump({"method": "operator_reviewed_revision",
                        "operator": result["operator_audit"]["operator"], "draft_round": result["draft_round"],
                        "basis": result["revision_basis"], "files": list(NAMES)})))

    def backfill_drafts(self, pid=None):
        """Recover existing failed attempts without changing their verdicts or contacting a provider."""
        from .profile import recover_draft_case
        with self.db() as db:
            papers = [dict(r) for r in db.execute("SELECT * FROM papers WHERE ? IS NULL OR id=?", (pid, pid))]
            archived = db.execute("SELECT * FROM archived_reviews WHERE ? IS NULL OR paper_id=?",
                                  (pid, pid)).fetchall()
        count = 0
        for p in papers:
            attempts = [(r["attempt"], json.loads(r["result"])) for r in archived if r["paper_id"] == p["id"]]
            attempts.append((p["attempt"], json.loads(p["result"])))
            previous_by_work = {}
            for attempt, result in sorted(attempts, key=lambda item: item[0]):
                work_attempt = result.get("work_attempt", attempt)
                previous_result = previous_by_work.get(work_attempt)
                previous_by_work[work_attempt] = result
                if result.get("status") != "failed" or not result.get("draft"):
                    continue
                paper = {**p, "attempt": attempt, "result": result}
                work = self.work(paper)
                for round_no, draft in saved_rounds(work, result, previous_result,
                                                     final_only=attempt != work_attempt and previous_result is None):
                    with self.db() as db:
                        existing = db.execute("""SELECT files,restored,recompiled FROM draft_artifacts
                            WHERE paper_id=? AND attempt=? AND round=? AND draft_sha256=?""",
                            (p["id"], attempt, round_no, draft_digest(draft))).fetchone()
                    if existing and existing["recompiled"] is not None and (existing["restored"] or "pdf" in json.loads(existing["files"])):
                        continue
                    try:
                        case = recover_draft_case(work, Path(p["source"]), Path(p["root"]) if p["root"] else None, draft)
                    except (OSError, ValueError, TypeError, KeyError):
                        case = None
                    self.record_draft(paper, round_no, draft, case, restored=True)
                    count += 1
        return count

    def _write_extraction(self, archive, paper, base, directory=""):
        tex = self.export_tex(paper)
        pdf = self.artifact(paper, "introduction.pdf")
        tex_name, pdf_name = base + ".tex", base + ".pdf"
        archive.writestr(directory + tex_name, tex)
        archive.write(pdf, directory + pdf_name)
        return {tex_name: hashlib.sha256(tex).hexdigest(), pdf_name: paper["result"]["hashes"]["introduction.pdf"]}

    def archive(self, batch, pid=None):
        output = io.BytesIO()
        rows = []
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            if pid is not None:
                paper = self.paper(pid)
                if paper["batch_id"] != batch:
                    raise ValueError("该论文不属于此批次")
                self._write_extraction(archive, paper, extraction_basename(paper["source"]))
            else:
                for i, info in enumerate(self.describe(batch)["papers"], 1):
                    row = {k: info[k] for k in ("id", "name", "status", "reason")}
                    if info["status"] == "completed":
                        paper = self.paper(info["id"])
                        base = extraction_basename(paper["source"])
                        directory = batch_export_directory(i, paper["name"]) + "/"
                        row["sha256"] = self._write_extraction(archive, paper, base, directory)
                        if info["profile_release"]:
                            row["profile_release"] = info["profile_release"]
                        if info.get("delivery"):
                            row["delivery"] = info["delivery"]
                    rows.append(row)
                archive.writestr("manifest.json", dump({"batch":batch,"papers":rows}))
        return output.getvalue()
