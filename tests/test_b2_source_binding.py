"""Source bindings and an independent existing review, exercised entirely offline."""
import json

from paper_extraction.models import AdmissionReview, Draft, Findings
from paper_extraction.selection import SelectionCorrection, WholePaperSelection, review_navigator, select_contributions
from test_agent import paper  # noqa: F401
from test_whole_selection import finding, pick


def test_assertion_location_and_b2_fact_location_remain_distinct_and_correctable():
    sources = [{"id": "intro", "text": "The application settles the compact-rank classification."},
               {"id": "body", "text": "A solution has odd rank exactly when its invariant vanishes."}]
    local = [finding("application", "Corollary A", sources[1]["text"], "body")]
    calls = []

    class Replies:
        def structured(self, schema, instructions, packet):
            calls.append(schema)
            if schema is SelectionCorrection:
                assert "b2_source_facts.source_id" in packet["conflict"]
                assert {r["id"] for r in packet["original_passages"]} == {"intro", "body"}
                assert packet["previous_bindings"][0]["b2_source_ids"] == ["unregistered"]
                assert "private prior interpretation" not in json.dumps(packet)
            fact = ({"source_id": "unregistered", "fact": "private prior interpretation"}
                    if schema is WholePaperSelection else {"source_id": "intro", "fact": sources[0]["text"]})
            return schema(selected=[pick("c1", "body", b2_source_facts=[fact])])

    row, = select_contributions(sources, local, Replies())
    assert calls == [WholePaperSelection, SelectionCorrection]
    assert row["decision"] == "admitted" and row["selection_evidence_ids"] == ["body"]
    assert row["b2_source_facts"] == [{"source_id": "intro", "fact": sources[0]["text"]}]
    assert review_navigator([row])[0]["evidence_ids"] == ["body", "intro"]
    old = {k: v for k, v in row.items() if k not in {"b2_source_facts", "local_evidence_ids", "selection_evidence_ids"}}
    assert review_navigator([old]) == [{"candidate_id": "c1", "source_number": "Corollary A", "evidence_ids": ["body", "intro"]}]


def test_empty_b2_facts_are_corrected_at_selection():
    source = {"id": "intro", "text": "The second proposition is the iteration step in the proof."}
    candidate = finding("current", "Proposition 2", "The iteration produces the next witness.", "intro")
    calls = []

    class Replies:
        def structured(self, schema, instructions, packet):
            calls.append(schema)
            if schema is WholePaperSelection:
                return schema(selected=[pick("c1", "intro", b2_source_facts=[])])
            assert "B2(iii)" in packet["conflict"] and packet["original_passages"] == [source]
            return schema(selected=[])

    row, = select_contributions([source], [candidate], Replies())
    assert calls == [WholePaperSelection, SelectionCorrection]
    assert row['decision'] != 'admitted'


def test_existing_review_can_remove_a_previously_admitted_proof_step_and_keep_an_application(paper, tmp_path):
    models, run, _, events = paper
    main = "Every compact object has a solution."
    auxiliary = "A sequence of witnesses reaches the next stage."
    application = "Rank-two objects admit a separate classification."
    context = "The witness sequence is the induction step in the proof of existence."
    source = "\n".join(["The source claim concerns compact objects.", main, context, auxiliary, application, "Prior theorem: a solution exists in rank one."])
    (tmp_path / "input" / "source.tex").write_text(source, encoding="utf-8")
    statements = [("historical", "Prior theorem", "A solution exists in rank one."),
                  ("current", "Theorem A", main), ("current", "Lemma B", auxiliary),
                  ("application", "Corollary C", application), ("current", "Lemma D", "An incidental bound.")]
    original = models.structured
    reviews = []

    def respond(schema, instructions, packet, image=None):
        if schema is Findings:
            return Findings(candidates=[finding(kind, label, claim, "t1-1") for kind, label, claim in statements],
                            uncertainties=[])
        if schema is WholePaperSelection:
            return schema(selected=[pick(f"c{i}", "t1-1", source_reason="PRIOR_APPROVAL_SENTINEL",
                                         b2_source_facts=[{"source_id": "t1-1", "fact": item[2]}])
                                    for i, item in enumerate(statements[:4], 1)],
                          exclusions=[{"candidate_ids": ["c5"], "source_reason": "PRIOR_EXCLUSION_SENTINEL"}])
        if schema is Draft:
            draft = original(schema, instructions, packet)
            claims = [main, application] if packet.get("repair_issues") else [main, auxiliary, application]
            data = draft.model_dump()
            data["groups"] = [{"definitions": [], "theorem": claim + r" \cite[p. 1]{source}."} for claim in claims]
            data["evidence_links"] = [dict(block_id=key, evidence_ids=["t1-1"])
                                      for key in ["paragraph", "history.0", *(f"groups.{i}.theorem" for i in range(len(claims)))]]
            return Draft.model_validate(data)
        if schema is AdmissionReview:
            assert all(set(c) == {"candidate_id", "source_number", "evidence_ids"} for c in packet["candidates"])
            assert len(packet["candidates"]) == len(statements)
            assert all(s not in json.dumps(packet) for s in ("PRIOR_APPROVAL_SENTINEL", "PRIOR_EXCLUSION_SENTINEL"))
            assert context in packet["passages"][0]["text"]
            groups = [g["theorem"] for g in packet["draft"]["groups"]]
            assert any(application in g for g in groups)
            reviews.append(groups)
            if any(auxiliary in g for g in groups):
                return AdmissionReview(route="draft", issues=[{
                    "kind": "improper_selection", "draft_field": "groups.1.theorem", "current_excerpt": auxiliary,
                    "evidence_id": "t1-1", "source_excerpt": context,
                    "required_change": "Remove this auxiliary induction step from the current results under B1 and B2.",
                    "operation": "remove",
                }], evidence_ids=[])
            return AdmissionReview(route="pass", issues=[], evidence_ids=[])
        return original(schema, instructions, packet, image)

    models.structured = respond
    result = run()
    assert result["status"] == "completed", result.get("reason")
    assert [len(group) for group in reviews] == [3, 2]
    assert [d["tool"] for _, d in events if d.get("tool") == "select_contributions"] == ["select_contributions"]

def test_legacy_b2_full_review_applies_once_to_failed_content_without_touching_layout_or_modern_results():
    from paper_extraction.agent import _needs_legacy_b2_review
    row = {"candidate_id":"c1", "decision":"admitted", "kind":"current"}
    failure = {"stage":"review", "passed":False}
    state = {"candidates":[row], "reviews":[failure]}
    assert _needs_legacy_b2_review(state)
    assert not _needs_legacy_b2_review({**state, "legacy_b2_reviewed":True})
    assert not _needs_legacy_b2_review({**state, "reviews":[{**failure, "legacy_b2_reviewed":True}, failure]})
    assert not _needs_legacy_b2_review({**state, "candidates":[{**row,"b2_source_facts":[]}]})
    for stage in ("compile", "visual"):
        assert not _needs_legacy_b2_review({**state,"reviews":[failure, {"stage":"review","passed":True},
                                                              {"stage":stage,"passed":False}]})
