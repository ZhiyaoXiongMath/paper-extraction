from copy import deepcopy

from paper_extraction.editorial import evidence_packet


def span(loc, text, section=""):
    return {"id": loc, "text": text, "section": section, "label": loc}


def candidate(kind, claim, ids, number=None, assumptions="A compact manifold"):
    return {"kind": kind, "claim": claim, "assumptions": assumptions, "decision": "admitted",
            "source_number": number, "evidence_ids": ids, "dependencies": [],
            "significance": "Manuscript attribution", "decision_reason": "Direct statement"}


def test_whole_draft_keeps_each_bound_span_even_beyond_packet_budget():
    catalog = [span("a", "Title and authors"), span("b", "Intermediate proof stage"),
               span("c", "Full hypotheses in a late section"), span("d", "The consequence with parameters.")]
    draft = {"evidence_links": [{"block_id": "paragraph", "evidence_ids": ["a"]},
             {"block_id": "groups.0.theorem", "evidence_ids": ["c", "d", "c", "missing"]}]}
    result = evidence_packet(catalog, [], draft, max_chars=1)
    assert [r["id"] for r in result["passages"]] == ["a", "c", "d"]
    assert result["passages"][-1]["text"] == catalog[-1]["text"]
    assert result["unregistered_ids"] == ["missing"] and result["draft"] == draft


def test_pick_relevant_original_text_independently_of_extracted_decisions():
    catalog = [span("front", "Paper title"), span("early", "A compactness question", "Introduction")]
    catalog += [span(str(i), "Repeated estimate in a stopping argument.", "Proof") for i in range(40)]
    catalog += [span("past", "It was known under strict positivity."),
                span("main", "Our main theorem establishes existence."),
                span("use", "As an application, convergence holds on the whole family.")]
    result = evidence_packet(catalog, [], max_chars=190)
    assert [row["id"] for row in result["passages"]] == ["front", "early", "past", "main", "use"]


def test_distinct_assertions_and_prior_works_survive_label_deduplication():
    rows = [
        candidate("current", "Existence of a metric.", ["c", "c"], "Thm. 1"),
        candidate("current", "Existence of a metric.", ["c", "d"], "Theorem 1"),
        candidate("current", "Existence of a unique metric.", ["e"], "Theorem 1", "A stronger hypothesis"),
        candidate("application", "A dimension consequence.", ["f"], "Theorem 1"),
        candidate("historical", "An earlier lower bound.", ["g"], "Theorem 1"),
        candidate("historical", "Another author's local result.", ["h"], "Theorem 1"),
    ]
    saved = deepcopy(rows)
    result = evidence_packet([span(x, x) for x in "cdefgh"], rows)
    assert rows == saved
    register = result["candidates"]
    assert len(register) == 4
    assert register[0]["evidence_ids"] == ["c", "d", "e"]
    variant, = register[0]["other_reported_clauses"]
    assert variant["claim"] == "Existence of a unique metric."
    assert variant["assumptions"] == "A stronger hypothesis" and variant["evidence_ids"] == ["e"]
    assert [r["kind"] for r in register[1:]] == ["application", "historical", "historical"]


def test_anonymous_oppositely_quantified_statements_are_never_flattened():
    rows = [candidate("current", "For every x, F is bounded.", ["a"]),
            candidate("current", "For some x, F is bounded.", ["a"])]
    assert len(evidence_packet([span("a", "The two assertions")], rows)["candidates"]) == 2

