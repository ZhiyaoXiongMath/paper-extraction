"""Scope provenance distinguishes paper evidence, OCR text, and editorial paraphrase."""
import pytest

from paper_extraction.scope_reconciliation import (
    SCOPE_EVIDENCE_VERSION, ScopeReconciliation, _source_packet, reconcile_scope, scope_failure_reason,
)
from paper_extraction.selection import SELECTION_VERSION
from paper_extraction.store import stale_selection_failure
from test_whole_selection import finding


def sample():
    catalog, rows = [], []
    for i in range(1, 6):
        loc = f"p{i}"
        catalog.append({"id": loc, "page": i, "text": f"Original wording on issue {i}.", "transcribed_from_image": i == 1})
        rows.append({**finding("current", f"Theorem {i}", f"Statement {i}", loc), "candidate_id": f"c{i}",
                     "selection_evidence_ids": [loc], "local_evidence_ids": [loc], "independent_question": f"Question {i}"})
    return catalog, rows


def reply(key, outcome="excluded", **kwargs):
    loc = "p" + key[1:]
    return dict(candidate_id=key, outcome=outcome, source_id=loc, source_reason="The source describes this use.",
                **kwargs)


def test_model_paraphrases_and_ocr_mismatches_do_not_require_quote_repair():
    catalog, rows = sample()

    class Model:
        calls = 0

        def structured(self, typ, prompt, packet):
            self.calls += 1
            assert typ is ScopeReconciliation
            return typ(assessments=[reply("c1", "admitted", independent_question="Does a global solution exist?",
                                          source_excerpt="Original wording on issue 1."),
                                    reply("c2", source_excerpt="Words the author never wrote."),
                                    reply("c3", source_excerpt="Original  wording on issue 3."),
                                    reply("c4"), reply("c5")])

    model = Model()
    final = reconcile_scope(catalog, rows, model, "")
    assert model.calls == 1 and final[0]["decision"] == "admitted"
    assert "scope_excerpt" not in final[0] and "scope_excerpt" not in final[1]
    assert final[1]["scope_reason_type"] == "paraphrase" and final[1]["scope_evidence_id"] == "p2"
    assert final[2]["scope_excerpt"] == "Original wording on issue 3."
    assert not scope_failure_reason(final)


def test_unresolved_equivalence_without_any_central_admission_reports_b4():
    catalog, rows = sample()

    class Model:
        calls = []

        def structured(self, typ, prompt, packet):
            self.calls.append(packet["requested_ids"])
            return typ(assessments=[reply(k, "equivalent", retained_id="c2") if k == "c1" else reply(k)
                                    for k in packet["requested_ids"]], central_open_ids=["c1"])

    model = Model()
    final = reconcile_scope(catalog, rows, model, "")
    assert model.calls == [["c1", "c2", "c3", "c4", "c5"], ["c1"]]
    assert final[0]["decision"] == "unresolved" and "represented_by" not in final[0]
    assert "does not reference a distinct admitted row" in final[0]["decision_reason"]
    reason = scope_failure_reason(final)
    assert "B4" in reason and "Theorem 1 [p1]" in reason


def test_unknown_scope_evidence_remains_actionable_after_targeted_repair():
    catalog, rows = sample()

    class Model:
        def structured(self, typ, prompt, packet):
            return typ(assessments=[{**reply(k), "source_id": "unregistered" if k == "c3" else "p" + k[1:]}
                                    for k in packet["requested_ids"]])

    with pytest.raises(ValueError, match="c3: excluded: source ID 'unregistered' was not supplied"):
        reconcile_scope(catalog, rows, Model(), "")


def test_scope_receives_original_bindings_and_keeps_legacy_scope_provenance():
    catalog, rows = sample()
    rows[0]["source_number"] = "characterization.of.operator"
    catalog.append({"id": "later", "text": r"Use Proposition \ref{characterization.of.operator} to check the hypothesis of Theorem B."})
    packet = _source_packet(catalog, rows)
    assert "original_uses" not in packet
    assert all("decision_reason" not in c and "independent_question" not in c for c in packet["candidate_rows"])
    assert {r["id"] for r in packet["original_overview"] + packet["original_statements_and_context"]} >= {"p1", "p5"}
    for row in rows:
        row["scope_role"] = "independent"
    result = {"selection_version": SELECTION_VERSION, "status": "failed", "candidates": rows}
    assert stale_selection_failure(result)
    for row in rows:
        row["scope_evidence_version"] = SCOPE_EVIDENCE_VERSION
    assert not stale_selection_failure(result)

