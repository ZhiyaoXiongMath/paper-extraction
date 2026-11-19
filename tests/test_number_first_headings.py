"""Exercise formal PDF titles in their printed order through both admission paths."""

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from paper_extraction import agent
from paper_extraction.admission import (
    current_statement, damaged_heading_page, source_statement_options, with_page_reading,
)
from paper_extraction.lean_workflow import DirectResult, direct_admission
from paper_extraction.profile_inputs import new_batch_profile
from paper_extraction.segment_evidence import SegmentLeads
from paper_extraction.selection import select_contributions
from test_core_admission_calibration import _direct_decision
from test_whole_selection import finding, pick


def pdf_page(text):
    return {"id": "p7-1", "page": 7, "label": "p. 7", "file": "paper.pdf", "section": "", "text": text}


def test_old_image_line_binding_resolves_to_the_unique_original():
    page = with_page_reading(pdf_page("(5.20) Theorem. Every fiber is finite."), "Theorem 5.20. Every fiber is finite.")
    candidate = {"source_number": "(5.20) Theorem", "evidence_ids": ["p7-1"], "source_statement_id": "p7-1.l2"}
    assert current_statement(candidate, {"p7-1": page})["id"] == "p7-1.l1"


def test_damaged_prefixed_type_only_nominates_a_unique_page_for_lookup():
    page = pdf_page("(3.5) THÉORÈML. — Every fiber is finite.")
    assert not source_statement_options(page)
    row = {"source_number": "Théorème 3.5", "evidence_ids": ["p7-1"]}
    assert damaged_heading_page(row, {"p7-1": page}) == "p7-1"
    second = {**page, "id": "p8-1", "page": 8}
    assert not damaged_heading_page({**row, "evidence_ids": ["p7-1", "p8-1"]},
                                    {"p7-1": page, "p8-1": second})


def test_options_keep_the_exact_line_for_each_printed_heading():
    text = "\n".join([
        "See Theorem 7.2 for another proof.",
        "Theorem 2. Let the first set be finite.",
        "2. Theorem 4.1 implies the assertion above.",
        "(7.20) THEOREM. – The operator is compact.",
        "Proof of Theorem 7.20.",
        "8. COROLLARY (Finite fibers): Each fiber has finite length.",
        "4.1. Théorème. — Tout élément possède un inverse.",
        "(9.2) LEMMA. An auxiliary map exists.",
        "3. PROPOSITION. A uniform estimate holds.",
        "(7.21)  f(x) = 0.",
    ])
    record = pdf_page(text)
    assert [(s["id"], s["kind"], s["number"]) for s in source_statement_options(record)] == [
        ("p7-1.l2", "theorem", "2"),
        ("p7-1.l4", "theorem", "7.20"),
        ("p7-1.l6", "corollary", "8"),
        ("p7-1.l7", "theorem", "4.1"),
        ("p7-1.l8", "lemma", "9.2"),
        ("p7-1.l9", "proposition", "3"),
    ]
    assert record["text"] == text


def test_same_number_and_true_duplicate_titles_require_enough_evidence():
    page = pdf_page("7. Theorem. First case.\n7. Corollary. A consequence.\n7. Theorem. Second case.")
    records = {page["id"]: page}
    row = finding("current", "7", "Existence of a solution.", page["id"])
    assert current_statement(row, records) is None
    assert current_statement({**row, "source_number": "Theorem 7"}, records) is None
    assert current_statement({**row, "source_number": "Corollary 7"}, records)["id"] == "p7-1.l2"
    assert current_statement({**row, "source_statement_id": "p7-1.l3"}, records)["id"] == "p7-1.l3"


@pytest.mark.parametrize("path", ["whole", "segmented", "with_reading"])
@pytest.mark.parametrize("printed_kind", ["Theorem", "Lemma", "Proposition"])
def test_graph_checks_the_printed_kind_on_the_bound_page(tmp_path, monkeypatch, path, printed_kind):
    # Use a bare model locator so only the original heading proves the type.
    claim = "Every finite set has a minimal element under the given total order."
    source = pdf_page(f"An ordering problem.\n(7.20) {printed_kind}. — {claim}")
    result, _ = _direct_decision([(source["id"], "7.20", "current", claim, "Answers the ordering problem.")])
    reading = {"source_id": source["id"], "page": source["page"], "region": None,
               "transcription": f"(7.20) {printed_kind}. — {claim}"}

    class Index:
        citation_metadata = {}

        def catalog(self):
            return [source]

    class Offline:
        def structured(self, schema, instructions, packet):
            if schema is DirectResult:
                return result
            if schema is SegmentLeads:
                return schema(candidates=[{"kind": "current", "source_number": "7.20", "claim": claim,
                                           "gain": "Answers the ordering problem.", "evidence_ids": [source["id"]]}])
            return schema.model_validate({"selected": [pick("c1", source["id"])]}, context=packet)

    monkeypatch.setattr(agent, "Models", lambda _: Offline())
    settings = new_batch_profile()
    if path == "segmented":
        settings["max_input_tokens"] = 1024
    graph, _ = agent._make_graph(tmp_path, tmp_path / "paper.pdf", None, settings,
                                 lambda *args: None, InMemorySaver(), "", Index(), None, None, lambda: 0, True)
    if path == "with_reading":
        # Seeing the same title again in a saved image transcription must not
        # turn one real source location into two possible headings.
        state = {"direct_evidence_packet": {"source_images": [reading], "source_questions": []}}
        outcome = graph.nodes["direct_finish"].bound.invoke(state)
    else:
        config = {"configurable": {"thread_id": "offline"}}
        state = {"cursor": 0, "attempt": 0, "issues": [], "candidates": [], "uncertainties": [], "reviews": []}
        graph.invoke(state, config, interrupt_before=["draft", "accept_direct", "summary"])
        checkpoint = graph.get_state(config)
        next_node = "accept_direct" if path == "whole" else "draft"
        assert checkpoint.next == ((next_node if printed_kind == "Theorem" else "summary"),)
        outcome = checkpoint.values
    row, = outcome["candidates"]
    if printed_kind == "Theorem":
        assert row["decision"] == "admitted"
        assert row["source_statement_id"] == "p7-1.l2"
        assert "screening_note" not in outcome
        if path != "segmented":
            assert outcome["pending_direct"]["core_bindings"]["groups.0.theorem"] == "c1"
    else:
        assert row["decision"] == "excluded"
        assert outcome["screening_note"]["kind"] == "no_qualifying_result"
        assert outcome["screening_note"]["evidence_ids"] == [source["id"]]


@pytest.mark.parametrize("path", ["whole", "segmented"])
@pytest.mark.parametrize("text,reason_fragment", [
    ("(7.20) Theorem. Case one.\n7.20. Theorem. Case two.", "多个"),
    ("See Theorem 7.20 for details.", "没有找到"),
])
def test_unresolved_title_report_locates_candidate_on_checked_page(path, text, reason_fragment):
    page = pdf_page(text)
    claim = "A fixed point exists."
    if path == "whole":
        result, _ = _direct_decision([(page["id"], "Theorem 7.20", "current", claim, "Solves existence.")])
        _, draft, note = direct_admission(result, {page["id"]: page}, strict_types=True)
        assert draft is None
    else:
        class Selector:
            def structured(self, schema, instructions, packet):
                return schema.model_validate({"selected": [pick("c1", page["id"])]}, context=packet)

        candidate = finding("current", "Theorem 7.20", claim, page["id"])
        _, screening = select_contributions([page], [candidate], Selector(), current=True, lean=True,
                                            optimized=True, strict_types=True)
        note = screening.model_dump()
    assert note["kind"] == "insufficient_evidence"
    assert "7.20" in note["reason"] and page["label"] in note["reason"]
    assert reason_fragment in note["reason"]
