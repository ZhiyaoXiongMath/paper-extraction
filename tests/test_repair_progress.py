"""v13 repair routing follows observable findings across durable rounds."""

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from paper_extraction import agent
from paper_extraction.models import AdmissionReview, Draft, DraftPatch, Review
from paper_extraction.repair_progress import legacy_quota_stop, record_progress
from paper_extraction.selection import SELECTION_VERSION
from test_agent import paper  # noqa: F401


def word_check(counts):
    diagnostics = [{"draft_field": field, "current_text": "saved block",
                    "diagnostics": [f"introparagraph_{i}_word_limit:{count}>45"]}
                   for i, (field, count) in enumerate(counts, 1)]
    return {"passed": False, "issues": ["bad counts"], "repair_targets": diagnostics}


def test_measured_improvement_and_resolved_sibling_allow_repair_but_worsening_alone_stalls():
    state = {"draft": {"paragraph": "saved block"}}
    for counts, stalled in [([("paragraph", 54), ("groups.0.paragraph", 58)], False),
                            ([("paragraph", 53), ("groups.0.paragraph", 58)], False),
                            ([("paragraph", 53)], False), ([("paragraph", 54)], True)]:
        check = word_check(counts)
        state.update(record_progress(state, "compile", [check], check["issues"]))
        assert state["repair_stalled"] is stalled


def test_new_findings_deliver_third_draft_without_a_third_review(paper, monkeypatch):
    _, run, _, events = paper
    calls = []

    def build(*args):
        calls.append(None)
        raise ValueError("New original checking diagnostic " + str(len(calls)))

    monkeypatch.setattr(agent, "build_case", build)
    result = run()
    assert result["status"] == "completed" and result["delivery"]["status"] == "delivered"
    assert result["draft_round"] == 3 and len(calls) == 2
    assert result["reviews"][-1]["issues"] == ["New original checking diagnostic 2"]
    assert [d["attempt"] for stage, d in events if stage == "draft"] == [1, 2, 3]
    assert not any(stage == "visual" for stage, _ in events)


def test_third_draft_cannot_trigger_review_or_extraction(paper, monkeypatch):
    model, run, _, events = paper
    respond = model.structured

    post_final_calls = []

    def checks(schema, instructions, payload, image=None):
        if model.drafts == 3:
            post_final_calls.append(schema)
        return respond(schema, instructions, payload, image)

    monkeypatch.setattr(model, "structured", checks)

    def early_layout_failure(*args):
        raise ValueError(f"Layout defect from draft {model.drafts}")

    monkeypatch.setattr(agent, "build_case", early_layout_failure)
    result = run()
    assert result["status"] == "completed" and result["draft_round"] == 3
    assert not post_final_calls
    assert model.drafts == 3 and model.extracts == 1
    assert all(not data.get("targeted") for stage, data in events if stage == "extract")


def test_compiler_filesystem_failure_uses_local_delivery_without_a_new_draft(paper, monkeypatch):
    model, run, _, _ = paper

    def missing_manifest(*args):
        raise OSError("Cannot create the build manifest")

    monkeypatch.setattr(agent, "build_case", missing_manifest)
    result = run()
    assert result["status"] == "completed" and result["delivery"]["status"] == "delivered"
    assert result["draft_round"] == model.drafts == 1 and result["draft"]
    assert [(check["stage"], check["passed"]) for check in result["reviews"]] == [("review", True), ("compile", False)]


def test_third_draft_can_publish_after_layout_and_vision_repairs(paper, monkeypatch):
    models, run, _, events = paper
    original_model, original_build = models.structured, agent.build_case
    stages = []
    compiles = []

    def build(*args):
        compiles.append(None)
        if len(compiles) == 1:
            draft = args[-1]
            error = "introparagraph_1_word_limit:54>45"
            raise agent.BuildFailure(error, [{"draft_field": "paragraph", "current_text": draft["paragraph"],
                                              "diagnostics": [error]}])
        return original_build(*args)

    def respond(schema, instructions, packet, image=None):
        if schema is DraftPatch:
            rows = packet["repair_targets"]
            stages.append(("patch", tuple(r["block_id"] for r in rows)))
            return DraftPatch(replacements=[{"block_id": r["block_id"],
                "text": "Setting." if r["block_id"] == "paragraph" else r["current_text"] + r" \allowbreak",
                "evidence_ids": r["required_evidence_ids"]} for r in rows])
        if schema is Draft:
            return original_model(schema, instructions, packet, image).model_copy(update={"paragraph": "Extended initial setting."})
        if schema is AdmissionReview:
            if "changed_blocks" in packet:
                assert "draft" not in packet and "candidates" not in packet
                stages.append(("check", tuple(row["draft_field"] for row in packet["changed_blocks"])))
            return AdmissionReview(route="pass", issues=[], evidence_ids=[])
        if schema is Review and len(compiles) == 2 and packet["page"] == 1:
            return Review(passed=False, issues=["A relation is clipped."],
                          locations=[{"draft_field":"groups.0.theorem", "issue":"A relation is clipped."}])
        return original_model(schema, instructions, packet, image)
    monkeypatch.setattr(agent, "build_case", build)
    monkeypatch.setattr(agent, "_render_pages", lambda path: [(1, f"edited-page-{len(compiles)}".encode()), (2, b"stable-page")])
    models.structured = respond
    result = run()
    assert result["status"] == "completed", result.get("reason")
    assert result["delivery"]["draft_round"] == 3 and result["delivery"]["status"] == "delivered"
    assert stages == [("patch", ("paragraph",)), ("check", ("paragraph",)),
                      ("patch", ("groups.0.theorem",))]
    assert [(d["attempt"], d["page"]) for stage, d in events if stage == "visual"] == [(2, 1), (2, 2)]


def test_explicit_retry_of_a_saved_old_third_round_repairs_the_same_work_as_round_four(paper, tmp_path):
    models, run, _, _ = paper
    initial = run()
    assert initial["status"] == "completed"
    work, source_root = tmp_path / "legacy", tmp_path / "input"
    entry = source_root / "source.tex"
    work.mkdir()
    settings = {"provider": "offline", "api_key": "unused", "base_url": "http://localhost",
                "text_model": "mock", "vision_model": "mock"}
    thread = agent._thread_id(entry, source_root, settings, agent._rules())
    state = dict(cursor=1, attempt=3, draft=initial["draft"], candidates=initial["candidates"],
        selection_version=SELECTION_VERSION, reviews=[{"stage": "compile", "attempt": 3, "passed": False,
        "issues": ["Recorded new layout issue"], "repair_targets": []}], issues=["Recorded new layout issue"],
        status="failed", reason="Recorded new layout issue")
    assert legacy_quota_stop(state)
    with SqliteSaver.from_conn_string(str(work / "checkpoint.sqlite")) as saver:
        old = StateGraph(agent.PaperState)
        old.add_node("old", lambda _: state)
        old.add_edge(START, "old")
        old.add_edge("old", END)
        old.compile(checkpointer=saver).invoke({}, {"configurable": {"thread_id": thread}})
    seen = []
    idle = agent.run_paper(work, entry, source_root, settings, lambda *x: seen.append(x))
    assert idle["status"] == "failed" and not seen
    resumed = agent.run_paper(work, entry, source_root, settings, lambda *x: seen.append(x), retry_legacy_review=True)
    assert resumed["status"] == "completed", resumed.get("reason")
    assert resumed["draft_round"] == 4 and not any(stage == "extract" for stage, _ in seen)
    assert resumed["reviews"][0]["issues"] == ["Recorded new layout issue"]

def test_height_progress_uses_recorded_units_including_groups_and_compares_each_box():
    state = {"draft": {"groups": [{"theorem": "current", "definitions": ["formula"]}]}}
    def measure(height, group, wide, narrow, text):
        target = {"draft_field": "groups.0.theorem", "current_text": text,
                  "diagnostics": [f"Block 2 (T) exceeds 12U (rendered height {height}U, limit 12U)",
                                  f"resultgroup_measurement_failed (rendered height {group}U, limit 22U)",
                                  f"Overfull \\hbox ({wide}pt too wide) in paragraph at lines 39--39",
                                  f"Overfull \\hbox ({narrow}pt too wide) in paragraph at lines 40--40"]}
        state.update(record_progress(state, "compile", [{"passed":False, "repair_targets":[target]}], ["layout"]))
        return state["repair_stalled"]
    assert not measure("12.604", "23.100", 10, 5, "old draft")
    assert not measure("12.400", "23.100", 10, 5, "shorter draft")
    assert not measure("12.400", "23.000", 10, 5, "improved group")
    assert not measure("12.400", "23.000", 10, 4, "improved second line")
    assert measure("12.400", "23.000", 10, 4, "different words, identical measurements")
    assert measure("12.600", "23.100", 12, 5, "different words, worse measurements")


def test_same_review_location_uses_actual_block_and_stops_rephrased_or_cyclic_defects():
    state = {"draft": {"history": ["A candidate historical statement."],
                       "groups": [{"theorem": "A current result."}]}}

    def check(field="history.0", source_id="p2-1", kind="incorrect_claim", quotation="A candidate", operation="replace"):
        issue = {"draft_field": field, "kind": kind, "evidence_id": source_id, "operation": operation,
                 "current_excerpt": quotation, "source_excerpt": "Changes with " + quotation}
        verdict = {"passed": False, "issues": [quotation], "defects": [issue]}
        state.update(record_progress(state, "review", [verdict], verdict["issues"]))
        return state["repair_stalled"]

    assert not check()
    assert check(quotation="historical statement", operation="remove")
    state["draft"]["history"][0] = "First revision."
    assert not check(quotation="First revision")
    state["draft"]["history"][0] = "A candidate historical statement."
    assert check(quotation="The opening adjective")
    assert not check(field="groups.0.theorem")
    assert not check(source_id="p3-1")
    assert not check(kind="incorrect_attribution_or_locator")


def test_identical_visual_page_cannot_gain_progress_from_a_new_description():
    state = {"draft": {"history": ["A historical block."], "groups": [{"theorem": "A current result."}]}}

    def check(sha="hash-1", page=1, field="history.0", issue="A dangling source citation."):
        verdict = {"passed": False, "page": page, "page_sha256": sha, "issues": [issue],
                   "locations": [{"draft_field": field, "issue": issue}]}
        state.update(record_progress(state, "visual", [verdict], verdict["issues"]))
        return state["repair_stalled"]

    assert not check()
    assert check(issue="The source note dangles by itself.")
    assert not check(field="groups.0.theorem")
    assert not check(page=2)
    assert not check(sha="hash-2")
    assert check()

def test_scoped_recheck_of_new_connecting_paragraph_and_reference_has_source_bindings(paper, tmp_path):
    models, run, _, _ = paper
    source = tmp_path / "input" / "source.tex"
    source.write_text(source.read_text(encoding="utf-8") +
        "\nConnectedness is the next assumption. Author. Test (2007).", encoding="utf-8")
    original = models.structured
    reviewed = []
    fixed = {"groups.0.paragraph": "Connectedness is the next assumption.",
             "references.0.text": "Author. Test (2007)."}
    def respond(schema, instructions, payload, image=None):
        if schema is DraftPatch:
            assert set(fixed) == {r["block_id"] for r in payload["repair_targets"]}
            return DraftPatch(replacements=[{"block_id": r["block_id"], "text": fixed[r["block_id"]],
                "evidence_ids": r["required_evidence_ids"]} for r in payload["repair_targets"]])
        if schema is AdmissionReview:
            if "changed_blocks" in payload:
                actual = {b["draft_field"]: b for b in payload["changed_blocks"]}
                assert {k: b["text"] for k,b in actual.items()} == fixed
                assert all(b["evidence_ids"] == ["t1-1"] for b in actual.values())
                assert any(row["draft_field"] == "groups.0.theorem" for row in payload["context_blocks"])
                assert "Author. Test (2007)." in payload["passages"][0]["text"]
                reviewed.append(actual)
                return AdmissionReview(route="pass", issues=[], evidence_ids=[])
            return AdmissionReview(route="draft", issues=[
                {"kind": "missing_required_content", "draft_field": "groups.0.paragraph", "current_excerpt": "",
                 "evidence_id": "t1-1", "source_excerpt": fixed["groups.0.paragraph"],
                 "required_change": "Insert the missing connecting assumption under v13 C1.", "operation": "insert",
                 "replacement_text": fixed["groups.0.paragraph"]},
                {"kind": "incorrect_attribution_or_locator", "draft_field": "references.0.text",
                 "current_excerpt": "Author. Test.", "evidence_id": "t1-1", "source_excerpt": fixed["references.0.text"],
                 "required_change": "Correct the publication year from the source under D7.", "operation": "replace",
                 "replacement_text": fixed["references.0.text"]}], evidence_ids=[])
        return original(schema, instructions, payload, image)
    models.structured = respond
    result = run()
    assert result["status"] == "completed", result.get("reason")
    assert result["draft_round"] == 2 and len(reviewed) == 1
    original_check = result["reviews"][0]
    for key in ("groups.0.paragraph", "references.0.text"):
        defect = next(d for d in original_check["defects"] if d["draft_field"] == key)
        same = {"passed": False, "issues": [defect["required_change"]], "defects": [defect]}
        state = {"draft": {"groups": [{"paragraph": ""}], "references": [{"text": "Author. Test."}]}}
        state.update(record_progress(state, "review", [same], same["issues"]))
        assert not state["repair_stalled"]
        state.update(record_progress(state, "review", [same], same["issues"]))
        assert state["repair_stalled"]

def test_stalled_layout_delivers_without_additional_model_calls(paper, monkeypatch):
    from paper_extraction.selection import WholePaperSelection
    models, run, _, events = paper
    old = models.structured
    calls, payloads = [], []
    def respond(schema, instructions, payload, image=None):
        calls.append(schema)
        if schema is DraftPatch:
            [t] = payload["repair_targets"]
            assert t["block_id"] == "paragraph" and "Overfull" in t["diagnostics"][0]
            assert any("source claim" in p["text"] for p in payload["passages"])
            assert payload["previous_draft"]["paragraph"] == t["current_text"]
            payloads.append(t["current_text"])
            return DraftPatch(replacements=[{"block_id":"paragraph","text":t["current_text"]+" Revised.",
                                              "evidence_ids":t["required_evidence_ids"]}])
        return old(schema, instructions, payload, image)
    def build(_, src, root, draft):
        finding = r"Overfull \hbox (5.50pt too wide) in paragraph at lines 7--7"
        raise agent.BuildFailure(finding, [{"draft_field":"paragraph", "current_text":draft["paragraph"], "diagnostics":[finding]}])
    models.structured = respond
    monkeypatch.setattr(agent, "build_case", build)
    first = run()
    assert first["status"] == "completed", first
    assert first["delivery"]["status"] == "delivered" and first["draft_round"] == 2
    assert first["reviews"][-1]["issues"] == first["reviews"][-3]["issues"]
    before = list(calls)
    assert run() == first and calls == before
    assert payloads == ["Scope."]
    assert calls.count(WholePaperSelection) == 1 and calls.count(Draft) == 1 and calls.count(DraftPatch) == 1
    assert sum(stage == "publish" for stage, _ in events) == 1


def test_old_selection_reuse_requires_same_round_content_pass_and_bound_compile_targets():
    from copy import deepcopy
    from paper_extraction.repair_progress import approved_legacy_layout_targets
    from test_saved_draft_review import old_terminal
    class Index:
        def catalog(self):
            return [{"id":"t1-1"}]
    state = old_terminal("layout")
    state["selection_version"] = "v13-whole-paper-2"
    state["reviews"].insert(-1, {"stage":"review", "passed":True, "issues":[], "attempt":3})
    index = Index()
    assert [t["block_id"] for t in approved_legacy_layout_targets(state, index, SELECTION_VERSION)] == ["paragraph"]
    for change in ("content_failed", "different_round", "no_selection", "unbound"):
        modified = deepcopy(state)
        if change == "content_failed":
            modified["reviews"][-2]["passed"] = False
        elif change == "different_round":
            modified["reviews"][-2]["attempt"] = 2
        elif change == "no_selection":
            modified["candidates"] = []
        else:
            modified["reviews"][-1]["repair_targets"][0]["draft_field"] = "unknown"
        assert approved_legacy_layout_targets(modified, index, SELECTION_VERSION) == []
    assert not agent._selection_ready({"selection_version":"", "retained_layout_selection":"v13-whole-paper-2"})
