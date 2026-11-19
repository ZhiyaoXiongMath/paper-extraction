"""Keep paid whole-source work across graph checkpoints."""
from paper_extraction import agent
from paper_extraction.lean_workflow import DirectResult, ScreeningNote
from paper_extraction.profile_inputs import new_batch_profile
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
