"""Conditional source reconciliation for potential v13 scope exceptions."""
import pytest

from paper_extraction.scope_reconciliation import ScopeReconciliation, scope_failure_reason
from paper_extraction.selection import SELECTION_VERSION, WholePaperSelection, select_contributions
from paper_extraction.store import stale_selection_failure
from test_whole_selection import finding, pick


def test_proof_ingredients_do_not_trigger_a_paper_size_exception():
    catalogue = [
        {"id": "intro", "section": "Introduction",
         "text": "We settle existence for compact objects and obtain a new application to classification. "
                 "The paper is organized as follows. The approximation and inverse estimates are used to construct the solution."},
        {"id": "state", "section": "Main results",
         "text": "Theorem A. Every compact object admits a solution. Equivalently all compact objects have a resolving section."},
        {"id": "prior", "section": "Introduction", "text": "Earlier work resolved the finite case."},
        {"id": "app", "section": "Applications", "text": "The solution classifies all connected pairs."},
        {"id": "proof", "section": "Proof",
         "text": "These approximation and inverse estimates are used only in the proof of Theorem A. "
                 "For the construction let W be the auxiliary weighted norm."},
    ]
    candidates = [
        finding("historical", "Prior result", "Finite objects admit a solution.", "prior"),
        finding("current", "Theorem A", "Every compact object admits a solution.", "state"),
        finding("current", "Proposition B", "All compact objects have a resolving section.", "state"),
        finding("current", "Lemma C", "The approximation error is bounded.", "proof"),
        finding("current", "Theorem D", "An inverse exists for the auxiliary approximation.", "proof"),
        finding("application", "Corollary E", "Solutions classify connected pairs.", "app"),
        finding("definition", "Definition F", "W is the auxiliary norm.", "proof"),
        finding("definition", "Definition G", "A resolving section yields a solution.", "state"),
    ]
    class Model:
        schemas = []

        def structured(self, schema, prompt, packet):
            self.schemas.append(schema)
            if schema is WholePaperSelection:
                assert list(packet)[0] == "passages" and "positive source indication" in prompt
                return schema(selected=[pick("c1", "prior"), pick("c2", "state"), pick("c3", "state", definition_ids=["c8"]),
                                        pick("c4", "proof", definition_ids=["c7"]), pick("c5", "proof"), pick("c6", "app")])
            assert schema is ScopeReconciliation and "B6." in prompt
            assert {p["id"] for p in packet["original_overview"]}.issuperset({"intro", "app"})
            assert {x["candidate_id"] for x in packet["provisional_contributions"]} == {"c2", "c3", "c4", "c5", "c6"}
            assert all("independent_question" in x and "selection_evidence_ids" in x for x in packet["provisional_contributions"])
            assert {r["id"] for f in ("original_overview", "original_statements_and_context") for r in packet[f]} >= {"proof", "app"}
            return schema(assessments=[
                {"candidate_id": "c2", "outcome": "independent", "source_id": "intro",
                 "source_excerpt": "We settle existence for compact objects", "source_reason": "This answers the existence problem posed in the introduction.",
                 "independent_question": "Which compact objects admit a solution?"},
                {"candidate_id": "c3", "outcome": "equivalent", "source_id": "state",
                 "source_excerpt": "Equivalently all compact objects have a resolving section", "source_reason": "The source states this as an equivalent formulation.", "retained_id": "c2"},
                *[{"candidate_id": c, "outcome": "ingredient", "source_id": "proof", "source_excerpt": "used only in the proof of Theorem A",
                   "source_reason": "The text uses this assertion solely to construct the object in Theorem A.", "retained_id": "c2"} for c in ("c4", "c5")],
                {"candidate_id": "c6", "outcome": "independent", "source_id": "app", "source_excerpt": "classifies all connected pairs",
                 "source_reason": "It resolves the classification problem for pairs.", "independent_question": "Which connected pairs are equivalent?"},
            ])
    model = Model()
    rows = select_contributions(catalogue, candidates, model)
    assert model.schemas == [WholePaperSelection, ScopeReconciliation]
    assert not scope_failure_reason(rows)
    assert [r["candidate_id"] for r in rows if r["decision"] == "admitted"] == ["c1", "c2", "c6", "c8"]
    assert rows[2]["represented_by"] == "c2" and rows[3]["context_for"] == ["c2"]
    assert rows[7]["needed_by"] == ["c2"] and rows[6]["decision"] == "excluded"
    assert rows[1]["scope_excerpt"] == "We settle existence for compact objects"


@pytest.mark.parametrize("count", [4, 5])
def test_genuine_independent_results_have_no_arbitrary_cutoff(count):
    questions = ["existence", "uniqueness", "classification", "optimal rates", "a geometric application"][:count]
    catalogue = [{"id": "overview", "section": "Introduction", "text": "The paper resolves " + ", ".join(questions)}]
    for i, question in enumerate(questions):
        catalogue.append({"id": f"p{i}", "section": "Main results", "text": f"This independently settles {question} in the stated class."})
    candidates = [finding("current", f"Theorem {i}", question, f"p{i}") for i, question in enumerate(questions)]

    class Model:
        reconciled = 0

        def structured(self, schema, prompt, packet):
            if schema is WholePaperSelection:
                return schema(selected=[pick(f"c{i+1}", f"p{i}") for i in range(count)])
            self.reconciled += 1
            return schema(assessments=[{"candidate_id": f"c{i+1}", "outcome": "independent", "source_id": f"p{i}",
                                        "source_excerpt": f"independently settles {q}", "source_reason": f"The source gives a separate {q} conclusion.",
                                        "independent_question": f"Does the stated class satisfy {q}?"} for i, q in enumerate(questions)])

    model = Model()
    rows = select_contributions(catalogue, candidates, model)
    reason = scope_failure_reason(rows)
    assert model.reconciled == int(count > 4)
    assert len([r for r in rows if r["decision"] == "admitted"]) == count
    if count == 5:
        assert all(f"Theorem {i} [p{i}]" in reason and q in reason for i, q in enumerate(questions))
    else:
        assert reason == ""


def test_previous_scope_terminal_is_eligible_only_for_explicit_version_upgrade():
    previous = {"status": "failed", "selection_version": "v13-whole-paper-1",
                "reason": "scope exception", "candidates": [finding("current", "1", "Statement", "p1")]}
    assert stale_selection_failure(previous)
    assert not stale_selection_failure({**previous, "selection_version": SELECTION_VERSION})

