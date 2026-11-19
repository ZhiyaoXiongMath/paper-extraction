"""Keep paid whole-source work across graph checkpoints."""
import pytest

from paper_extraction import agent
from paper_extraction.lean_workflow import DirectResult, ScreeningNote
from paper_extraction.profile_inputs import GROUNDED_WORKFLOW, WORKFLOW_REVISION, new_batch_profile
from paper_extraction.segment_evidence import SegmentLeads


def test_resume_reads_saved_source_action_before_finalizing(tmp_path, monkeypatch):
    source = tmp_path / "original" / "note.tex"
    source.parent.mkdir()
    source.write_text(
        "\\section{Question} An open question.\n"
        "\\section{Discussion} The statement is an open question.\n"
        "\\section{Appendix} Unrelated background text.\n",
        encoding="utf-8",
    )
    stop = {"requested": False}
    calls, source_ids, events = [], [], []

    class Responses:
        def structured(self, schema, instructions, payload):
            assert schema is DirectResult
            calls.append(schema)
            ids = [row["id"] for row in payload["passages"]]
            if len(calls) == 1:
                source_ids.extend(ids)
                assert len(source_ids) == 3
                answer = DirectResult(source_checks=[{"source_id": ids[0], "target_id": ids[1],
                                                     "question": "Is the statement a question?"}])
                answer._provisional = {"summary": {"kind": "no_qualifying_result", "question": "有定理吗？",
                                                   "reason": "仅提出问题。", "evidence_ids": [ids[0]]}}
                stop["requested"] = True
                return answer
            assert source_ids[:2] == ids
            assert payload["source_questions"][0]["target_id"] == ids[1]
            assert payload["provisional_response"]["summary"]["evidence_ids"] == [ids[0]]
            return DirectResult(summary={"kind": "no_qualifying_result", "question": "有定理吗？",
                                         "reason": "仅提出问题。", "evidence_ids": ids})

    monkeypatch.setattr(agent, "Models", lambda _: Responses())
    config = new_batch_profile()
    work = tmp_path / "work"
    notify = lambda stage, item: events.append((stage, item))
    first = agent.run_paper(work, source, source.parent, config, notify, pause_requested=lambda: stop["requested"])
    assert first["status"] == "paused", first.get("reason")
    assert calls == [DirectResult]
    assert not any(stage == "evidence" for stage, _ in events)

    stop["requested"] = False
    resumed = agent.run_paper(work, source, source.parent, config, notify, pause_requested=lambda: stop["requested"])
    assert resumed["status"] == "completed", resumed.get("reason")
    assert resumed["delivery"]["kind"] == "summary"
    assert calls == [DirectResult, DirectResult]
    assert [item["tool"] for stage, item in events if stage == "evidence"] == ["resolve_source_checks"]


def test_repeat_source_check_continues_in_existing_segment_path(tmp_path, monkeypatch):
    source = tmp_path / "original" / "note.tex"
    source.parent.mkdir()
    source.write_text("An open question. Does each example have property P?")
    calls, ids, events = [], [], []

    class Responses:
        def structured(self, schema, instructions, payload):
            calls.append(schema)
            current_ids = [row["id"] for row in payload["passages"]]
            if len(calls) == 1:
                ids.extend(current_ids)
            assert current_ids == ids
            if schema is DirectResult:
                return DirectResult(source_checks=[{"source_id": ids[0], "target_id": ids[0],
                                                    "question": "Is the statement a question?"}])
            assert schema is SegmentLeads
            assert payload["source_questions"][0]["target_id"] == ids[0]
            return SegmentLeads(candidates=[])

    def choose(*args, **kwargs):
        return [], ScreeningNote(kind="no_qualifying_result", question="有没有定理？", reason="仅提出问题。",
                                 evidence_ids=ids)

    monkeypatch.setattr(agent, "Models", lambda _: Responses())
    monkeypatch.setattr(agent, "select_contributions", choose)
    result = agent.run_paper(tmp_path / "work", source, source.parent, new_batch_profile(),
                             lambda stage, item: events.append((stage, item)))
    assert result["status"] == "completed", result.get("reason")
    assert calls == [DirectResult, DirectResult, SegmentLeads]
    assert sum(item.get("tool") == "resolve_source_checks" for _, item in events) == 1
    assert sum(item.get("reason") == "remaining_source_questions" for _, item in events) == 1


@pytest.mark.parametrize("version", [GROUNDED_WORKFLOW, WORKFLOW_REVISION])
def test_check_action_instructions_match_batch_version(tmp_path, monkeypatch, version):
    class Source:
        citation_metadata = {}
        def catalog(self):
            return [{"id": "p1-1", "page": 1, "text": "An open mathematical question."}]

    calls = []
    class Offline:
        def structured(self, schema, instructions, packet):
            calls.append(instructions)
            return DirectResult(summary={"kind": "no_qualifying_result", "question": "有定理吗？",
                                         "reason": "只提出问题。", "evidence_ids": ["p1-1"]})

    settings = {**new_batch_profile(), "workflow_revision": version}
    monkeypatch.setattr(agent, "Models", lambda _: Offline())
    graph, _ = agent._make_graph(tmp_path, tmp_path / "source.pdf", None, settings,
                                 lambda *args: None, None, "", Source(), None, None, lambda: 0, True)
    graph.nodes["direct"].bound.invoke({})
    instructions, = calls
    assert ("Leave cores empty and draft and summary null." in instructions) == (version == GROUNDED_WORKFLOW)
    assert ("Do not write a new draft just to request evidence." in instructions) == (version == WORKFLOW_REVISION)
    assert ("on the answer, return only the checks." in instructions) == (version == WORKFLOW_REVISION)


@pytest.mark.parametrize("variant,retained,reason", [
    ("complete", [1, 2, 3, 4, 5, 6, 7, 8], "validated_provisional"),
    ("misplaced", [1, 2, 3, 4, 5, 6, 7, 8], "validated_provisional"),
    ("missing", list(range(1, 10)), "no_validated_provisional"),
    ("unknown", list(range(1, 10)), "invalid_provisional"),
    ("bad-action", list(range(1, 10)), "unresolved_source_dependency"),
    ("empty-search", list(range(1, 10)), "unresolved_source_dependency"),
    ("unknown-heading", list(range(1, 10)), "unresolved_source_dependency"),
])
def test_direct_finish_preserves_all_identified_dependencies(tmp_path, monkeypatch, variant, retained, reason):
    from test_direct_group_position import native_draft

    class Source:
        citation_metadata = {}
        def catalog(self):
            return [{"id": f"p{i}-1", "page": i, "file": "paper.pdf", "section": f"§{i}",
                     "text": f"Theorem {i}.\nSupporting statement {i}."} for i in range(1, 10)]

    class Captured(Exception):
        pass

    class Offline:
        def structured(self, schema, instructions, packet):
            assert schema is DirectResult
            sent.append(packet)
            raise Captured

    sent, events = [], []
    monkeypatch.setattr(agent, "Models", lambda _: Offline())
    graph, _ = agent._make_graph(tmp_path, tmp_path / "source.pdf", None, new_batch_profile(),
                                 lambda stage, row: events.append(row), None, "", Source(), None, None, lambda: 0, True)
    provisional = native_draft()
    provisional["cores"][0]["independent_role"] = {
        "source_id": "p7-1", "role": "independent_application", "quote": "An independent application."}
    provisional["cores"][1]["source_statement_id"] = "p8-1.l1"
    if variant == "misplaced":
        group = provisional["draft"]["groups"][0]
        group["paragraph"]["paragraph_position"] = group.pop("paragraph_position")
    if variant == "unknown":
        provisional["cores"][0]["evidence_ids"] = ["p404-1"]
    if variant == "unknown-heading":
        provisional["cores"][1]["source_statement_id"] = "p8-1.l404"
    question = {"source_id": "p1-1", "target_id": "p2-1", "question": "Which range applies?"}
    if variant == "empty-search":
        question["search_hits"] = []
    evidence = {"source_questions": [question], "source_images": [], "passage_ids": ["p1-1", "p2-1"]}
    state = {"direct_provisional": {} if variant == "missing" else provisional, "direct_evidence_packet": evidence,
             "direct_invalid_checks": [{"code": "unknown_source_id"}] if variant == "bad-action" else []}
    with pytest.raises(Captured):
        graph.nodes["direct_finish"].bound.invoke(state)
    packet, = sent
    ids = [f"p{i}-1" for i in retained]
    assert [r["id"] for r in packet["passages"]] == ids
    event, = (e for e in events if e.get("mode") == "direct_finish_context")
    assert event["reason"] == reason and event["source_ids"] == ids
    if len(retained) < 9:
        assert [r["id"] for r in packet["source_locations"]] == [f"p{i}-1" for i in range(1, 10)]
        assert packet["source_locations"][-1]["page"] == 9
        # When a later correction resends just one cited passage, all nine pages
        # must still be discoverable even if their metadata was deduplicated.
        from paper_extraction.direct_response import direct_retry_packet
        correction = direct_retry_packet(packet, {"summary": {"evidence_ids": ["p1-1"]}},
                                         [{"code": "unknown_source_id"}])
        assert [r["id"] for r in correction["passages"]] == ["p1-1"]
        assert [r["page"] for r in correction["source_locations"]] == list(range(1, 10))
