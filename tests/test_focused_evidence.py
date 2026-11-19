import pytest

from paper_extraction.focused_evidence import projected_sources, repair_context
from paper_extraction.models import json_text
from paper_extraction.selection import LeanChoice, SourceSelection, select_contributions


@pytest.mark.parametrize(('extra_bytes', 'expected'), [(25, LeanChoice), (150_000, SourceSelection)])
def test_optional_full_selection_respects_increment_and_model_capacity(extra_bytes, expected):
    class Seen(Exception):
        pass

    class Reader:
        def structured(self, schema, instructions, packet):
            assert schema is expected
            assert ('b' in {row['id'] for row in packet['passages']}) == (schema is LeanChoice)
            raise Seen

    catalog = [{'id': 'a', 'text': 'Original theorem.'}, {'id': 'b', 'text': 'x' * extra_bytes}]
    candidate = dict(kind='current', claim='A result', source_number='Theorem 1', assumptions='',
                     dependencies=[], evidence_ids=['a', 'b'], decision='admitted', decision_reason='Main result.')
    with pytest.raises(Seen):
        select_contributions(catalog, [candidate], Reader(), current=True, lean=True, optimized=True,
                             strict_types=True, selection_settings={'context_window': 100_000})


def test_select_can_read_complete_proofs_without_truncating_adjacent_exceptions():
    first = r"Only for a general flag, some positive multiple is representable; except in cases (i) and (ii)."
    proof_start = "\\begin{proof}\n" + "verification " * 100
    proof_end = "\n\\end{proof}\n" + r"The application also covers every $n\ge 10$."
    source = [{"id": "a", "file": "main.tex", "text": first + proof_start},
              {"id": "b", "file": "main.tex", "text": "We settle the independent question." + proof_end}]
    projected = projected_sources(source, ["a", "b"], abridge=True)
    assert all(p["abridged"] for p in projected)
    assert projected[0]["text"].startswith(first)
    assert projected[1]["text"].endswith(r"The application also covers every $n\ge 10$.")
    assert len(json_text(projected)) < len(json_text(source))
    candidate = {"kind": "current", "claim": first, "source_number": "Theorem B", "assumptions": "a general flag",
                 "evidence_ids": ["a", "b"], "dependencies": [], "decision": "admitted", "decision_reason": "stated independently"}
    class Reader:
        def structured(self, schema, instructions, payload):
            if schema is SourceSelection:
                assert ("read_source_ids" in instructions and payload["passages"][0]["abridged"])
                return schema.model_validate({"read_source_ids": ["b"]}, context=payload)
            assert schema is LeanChoice
            assert next(r["text"] for r in payload["passages"] if r["id"] == "b") == source[1]["text"]
            return schema.model_validate({"selected": [{"candidate_id": "c1", "independent_question": "Which multiples?",
                     "b2_source_facts": [{"source_id": "b", "fact": "The independent question is settled"}],
                     "source_reason": "Extends applicability.", "evidence_ids": ["a", "b"]}]}, context=payload)
    selected, note = select_contributions(source, [candidate], Reader(), current=True, lean=True, optimized=True)
    assert len(selected) == 1 and selected[0]["decision"] == "admitted" and note is None
    assert "\\begin{proof}" in source[0]["text"]


def test_patch_sends_original_target_evidence_and_only_neighbour_text():
    sources = {k: {"id": k, "text": v} for k, v in (("p1", "A positive integer multiple."),
               ("p2", "The required exception."), ("p3", "An unrelated proof."))}
    targets = [{"block_id": "groups.1.theorem", "bound_evidence_ids": ["p1"], "required_evidence_ids": ["p2"]}]
    blocks = {"groups.1.theorem": "Fix this statement", "shared_definitions.0": "Definition of representation"}
    adjacent, evidence = repair_context(targets, blocks, sources)
    assert adjacent == {"shared_definitions.0": "Definition of representation"}
    assert [r["text"] for r in evidence] == ["A positive integer multiple.", "The required exception."]


def test_projection_keeps_ambiguous_or_nested_statements():
    rows = [{"id": "a", "file": "one.tex", "text": r"\begin{proof}\begin{lemma}A\end{lemma}\end{proof}"},
            {"id": "b", "file": "two.tex", "text": "\\begin{proof}\n% \\end{proof}\nA necessary assumption"},
            {"id": "p", "page": 2, "text": "An ambiguous exponent on the source page."}]
    assert [r["text"] for r in projected_sources(rows, ["a", "b", "p"], abridge=True)] == [r["text"] for r in rows]
