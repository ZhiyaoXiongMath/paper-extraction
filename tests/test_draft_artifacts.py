from copy import deepcopy
import hashlib
import json

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from PIL import Image

from paper_extraction import agent, worker
from paper_extraction.api import create_app
from paper_extraction.draft_artifacts import draft_digest, saved_rounds
from paper_extraction.models import DraftPatch, Review
from paper_extraction.profile import BuildFailure, render_tex
from paper_extraction.store import Store
from test_agent import OfflineModels
from test_profile import synthetic_draft


def prepare(home):
    store = Store(home)
    source = home / "input" / "source.tex"
    source.parent.mkdir()
    source.write_text("The source claim applies to compact objects.", encoding="utf-8")
    store.create("b", {"provider": "offline", "text_model": "t", "vision_model": "v", "base_url": "offline"},
                 [dict(id=i, name=i, source=str(source), kind="tex", root=str(source.parent)) for i in ("a", "b")])
    return store, store.paper("a")


def build_receipt(case, draft):
    directory = case / "draft-artifacts" / ("1" * 32)
    directory.mkdir(parents=True)
    tex = render_tex(draft).replace("\n", "\r\n").encode()
    (directory / "introduction.tex").write_bytes(tex)
    pdf = directory / "introduction.pdf"
    Image.new("RGB", (50, 50), "white").save(pdf)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
    diagnostic = {"directory": directory.relative_to(case).as_posix(), "run_id": "1" * 32, "pages": 1, "files": hashes,
                  "input_manifest_sha256": "fixture-manifest"}
    (directory / "receipt.json").write_text(json.dumps(diagnostic), encoding="utf-8")
    (case / "build-status.json").write_text(json.dumps({"engineering_accepted": False, "run_id": "1" * 32,
        "input_binding": {"manifest_sha256": "fixture-manifest"}, "diagnostic_artifacts": diagnostic}), encoding="utf-8")
    return tex, pdf.read_bytes()


def test_versions_download_only_the_registered_bytes_and_do_not_become_publications(tmp_path):
    store, paper = prepare(tmp_path)
    original, newer = synthetic_draft(), synthetic_draft()
    newer["paragraph"] = "The newly revised setting."
    original_copy = deepcopy(original)
    work = store.work(paper)
    store.record_draft(paper, 1, original)
    case = work / "original-case"
    tex, pdf = build_receipt(case, original)
    (case / "draft-recovery.json").write_text(json.dumps({"kind": "local_recompile", "draft_sha256": draft_digest(original)}))
    store.record_draft(paper, 1, original, case, restored=True)
    store.record_draft(paper, 2, newer, case)  # A case whose TeX belongs to round one.
    store.finish(paper, {"status": "failed", "draft": newer, "draft_round": 2, "reason": "Revision rejected"})
    assert original == original_copy
    with TestClient(create_app(tmp_path, runner=lambda *args: None)) as client:
        detail = client.get("/api/batches/b").json()["papers"][0]
        latest, older = detail["draft_artifacts"]
        assert latest["current"] and latest["round"] == 2 and set(latest["files"]) == {"tex"}
        assert older["restored"] and older["recompiled"] and not older["current"]
        assert older["work_attempt"] == older["attempt"] == 1
        for name, data in (("tex", tex), ("pdf", pdf)):
            reference = older["files"][name]
            assert reference["sha256"] == hashlib.sha256(data).hexdigest()
            response = client.get(reference["url"])
            assert response.content == data
            assert response.headers["content-disposition"] == f'inline; filename="source-extraction.{name}"'
            assert client.get(reference["url"].replace("/papers/a/", "/papers/b/")).status_code == 404
            assert client.get(reference["url"].replace("/drafts/1/1/", "/drafts/2/1/")).status_code == 404
            assert client.get(reference["url"].split("?")[0] + "?sha256=wrong").status_code == 404
        assert client.get("/api/papers/a/artifacts/introduction.pdf").status_code == 400
        assert client.get("/api/papers/a/download").status_code == 400
        target = store.draft_artifact("a", 1, 1, older["draft_sha256"], "introduction.pdf")
        target.write_bytes(b"changed on disk")
        assert client.get(older["files"]["pdf"]["url"]).status_code == 400


@pytest.mark.parametrize("failed_stage", ["compile", "visual"])
def test_worker_retains_writer_rounds_and_publishes_after_check_rejection(tmp_path, monkeypatch, failed_stage):
    store, _ = prepare(tmp_path)
    fixture = OfflineModels()
    fixture.visual_passes = False
    original = fixture.structured

    def respond(schema, instructions, packet, image=None):
        if schema is Review and failed_stage == "visual":
            issue = "Theorem 1 has a clipped relation."
            return Review(passed=False, issues=[issue], locations=[dict(draft_field="groups.0.theorem", issue=issue)])
        if schema is DraftPatch and failed_stage == "visual":
            fixture.patches += 1
            return DraftPatch(replacements=[dict(block_id=target["block_id"], text=target["current_text"].replace("Source claim", "Claim"),
                                               evidence_ids=target["required_evidence_ids"]) for target in packet["repair_targets"]])
        return original(schema, instructions, packet, image)

    fixture.structured = respond
    monkeypatch.setattr(agent, "Models", lambda _: fixture)
    monkeypatch.setattr(worker, "provider_settings", lambda *a, **kw: {"api_key": "fixture-only"})

    def build(work, source, root, draft):
        case = work / f"case-{fixture.drafts + fixture.patches}"
        _, pdf = build_receipt(case, draft)
        if failed_stage == "compile":
            raise BuildFailure("Fixture engineering rejection.", [], case_dir=case)
        (case / "introduction.pdf").write_bytes(pdf)
        return case

    monkeypatch.setattr(agent, "build_case", build)
    worker.process(store.home, "a")
    p = store.paper("a")
    assert p["status"] == "completed" and p["result"]["reviews"][-1]["stage"] == failed_stage
    assert p["result"]["delivery"]["status"] == "delivered"
    assert store.artifact(p, "introduction.pdf").read_bytes().startswith(b"%PDF")
    versions = store.describe("b")["papers"][0]["draft_artifacts"]
    last = p["result"]["draft_round"]
    assert last >= 2 and [v["round"] for v in versions] == list(range(last, 0, -1))
    assert all("tex" in v["files"] for v in versions)
    assert all("pdf" in v["files"] for v in versions[1:])
    assert [v["current"] for v in versions] == [True] + [False] * (last - 1)


def test_legacy_resume_does_not_attach_earlier_writer_rounds_to_the_following_attempt(tmp_path):
    store, paper = prepare(tmp_path)
    work = store.work(paper)
    work.mkdir(parents=True)
    (work / "paper-binding.json").write_text(json.dumps({"identity": "same-paper"}))
    config = {"configurable": {"thread_id": "same-paper"}}
    drafts = {i: {**synthetic_draft(), "paragraph": f"Revision {i}."} for i in (1, 2, 3)}
    old_reviews = [{"attempt": 1, "stage": "review", "passed": False}]
    final_reviews = old_reviews + [{"attempt": 2, "stage": "review", "passed": False},
                                   {"attempt": 3, "stage": "review", "passed": False}]
    with SqliteSaver.from_conn_string(str(work / "checkpoint.sqlite")) as saver:
        graph = StateGraph(agent.PaperState)
        graph.add_node("first", lambda _: {"attempt": 1, "draft": drafts[1], "candidates": [], "reviews": []})
        graph.add_node("second", lambda _: {"attempt": 2, "draft": drafts[2], "reviews": old_reviews})
        graph.add_node("resumed", lambda _: {"reviews": final_reviews[:2]})
        graph.add_node("third", lambda _: {"attempt": 3, "draft": drafts[3], "reviews": final_reviews, "status": "failed"})
        for start, finish in ((START, "first"), ("first", "second"), ("second", "resumed"), ("resumed", "third"), ("third", END)):
            graph.add_edge(start, finish)
        graph.compile(checkpointer=saver).invoke({"cursor": 1}, config)
    earlier = {"status": "failed", "resumable": True, "draft": drafts[2], "candidates": [], "reviews": old_reviews}
    latest = {"status": "failed", "draft": drafts[3], "candidates": [], "reviews": final_reviews}
    assert [n for n, _ in saved_rounds(work, earlier)] == [1, 2]
    assert [n for n, _ in saved_rounds(work, latest, earlier)] == [2, 3]
    assert saved_rounds(work, {**latest, "draft": {**drafts[3], "paragraph": "Different source."}}) == []
