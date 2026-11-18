"""Capture the TeX and readable PDF for one particular saved writer revision."""
import hashlib
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import NamedTemporaryFile

from langgraph.checkpoint.sqlite import SqliteSaver

from .pdfops import page_count
from .profile import render_tex

FILENAMES = {"tex": "introduction.tex", "pdf": "introduction.pdf"}


def draft_digest(draft):
    canonical = json.dumps(draft, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def snapshot_dir(work):
    # One immutable content store per paper work attempt; the DB binds each digest to its writer revision.
    return work / "draft-files"


def _write_content(directory, kind, data):
    digest = hashlib.sha256(data).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / f"{digest}.{kind}"
    if not dest.is_file() or hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
        with NamedTemporaryFile(dir=directory, delete=False) as f:
            temp = Path(f.name)
            try:
                f.write(data)
            except BaseException:
                f.close()
                temp.unlink(missing_ok=True)
                raise
        try:
            os.replace(temp, dest)
        finally:
            temp.unlink(missing_ok=True)
    return digest


def _compiled_files(work, case, expected_tex):
    if not case:
        return {}
    case = Path(case).resolve()
    if not case.is_relative_to(work) or case == work:
        return {}
    status_path = case / "build-status.json"
    status = json.loads(status_path.read_text("utf-8")) if status_path.is_file() else {}
    diagnostic = status.get("diagnostic_artifacts")
    location = (case / diagnostic["directory"]).resolve() if diagnostic else case
    if not location.is_relative_to(case):
        return {}
    if diagnostic:
        run_id = status.get("run_id")
        receipt = location / "receipt.json"
        if (not run_id or diagnostic.get("run_id") != run_id or diagnostic["directory"] != f"draft-artifacts/{run_id}"
                or not receipt.is_file() or receipt.is_symlink() or json.loads(receipt.read_text("utf8")) != diagnostic
                or diagnostic.get("input_manifest_sha256") != status.get("input_binding", {}).get("manifest_sha256")):
            return {}
    tex, pdf = ((location / name).resolve() for name in FILENAMES.values())
    if not tex.is_relative_to(location) or not tex.is_file() or tex.read_text("utf-8") != expected_tex:
        return {}
    tex_data = tex.read_bytes()
    files = diagnostic.get("files", {}) if diagnostic else {}
    if diagnostic and files.get(tex.name) != hashlib.sha256(tex_data).hexdigest():
        return {}
    verified = {"tex": tex_data}
    if not pdf.is_relative_to(location) or not pdf.is_file():
        return verified
    pdf_data = pdf.read_bytes()
    bound_hash = files.get(pdf.name) if diagnostic else status.get("compiled_pdf_sha256") or status.get("pdf_sha256")
    if bound_hash != hashlib.sha256(pdf_data).hexdigest():
        return verified
    try:
        readable = page_count(pdf) > 0
    except Exception:
        return verified
    if readable:
        verified["pdf"] = pdf_data
    return verified


def capture_files(work, draft, case):
    directory = snapshot_dir(work)
    try:
        rendered = render_tex(draft)
    except (TypeError, KeyError, ValueError):
        return {}, False
    try:
        contents = _compiled_files(work, case, rendered)
    except (OSError, ValueError, TypeError, KeyError):
        contents = {}
    compiled_tex = "tex" in contents
    contents.setdefault("tex", rendered.replace("\n", os.linesep).encode("utf-8"))
    return {kind: _write_content(directory, kind, data) for kind, data in contents.items()}, compiled_tex


def saved_rounds(work, result, previous_result=None, *, final_only=False):
    """Follow a matching terminal head's ancestry; a prior terminal starts another attempt."""
    target = result.get("draft")
    if not target:
        return []
    binding, checkpoint = work / "paper-binding.json", work / "checkpoint.sqlite"
    round_from_result = result.get("draft_round")
    final_snapshot = [(round_from_result, target)] if type(round_from_result) is int and round_from_result > 0 else []
    if not binding.is_file() or not checkpoint.is_file():
        return final_snapshot
    thread = json.loads(binding.read_text("utf-8"))["identity"]
    config = {"configurable": {"thread_id": thread, "checkpoint_ns": ""}}
    with closing(sqlite3.connect(checkpoint.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        saver = SqliteSaver(db)
        def same_output(data, saved):
            return (data.get("draft", {}) == saved.get("draft", {})
                    and data.get("reviews", []) == saved.get("reviews", [])
                    and data.get("candidates", []) == saved.get("candidates", []))

        # list() holds the saver's cursor lock across yields. Release it before following ancestors.
        with closing(saver.list(config)) as entries:
            entry = next((given for given in entries if same_output(given.checkpoint["channel_values"], result)
                          and (given.checkpoint["channel_values"].get("status") == "failed" or result.get("resumable"))), None)
        if not entry:
            return final_snapshot
        versions = {}
        final_round = entry.checkpoint["channel_values"].get("attempt", 0)
        if final_only:
            return [(final_round, target)] if final_round > 0 else final_snapshot
        while entry:
            data = entry.checkpoint["channel_values"]
            round_no, draft = data.get("attempt", 0), data.get("draft")
            if draft and round_no > 0:
                versions.setdefault(round_no, draft)
            entry = saver.get_tuple(entry.parent_config) if entry.parent_config else None
            if entry:
                parent = entry.checkpoint["channel_values"]
                previous_reviews = parent.get("reviews", [])
                if (parent.get("status") in {"failed", "completed"} or parent.get("attempt", 0) > round_no
                        or (previous_result and same_output(parent, previous_result))
                        or previous_reviews != data.get("reviews", [])[:len(previous_reviews)]):
                    break
        if final_round > 0 and versions.get(final_round) == target:
            return sorted(versions.items())
    return []
