"""Exercise the production graph with fixed provider replies and the local TeX toolchain."""
from pathlib import Path
import shutil

import pytest

from paper_extraction import agent
from paper_extraction.lean_workflow import AddedCores, DirectResult, EvidenceReview, fits_direct
from paper_extraction.models import AdmissionReview, AuthoritativeDraft, DraftPatch, Review
from paper_extraction.profile_inputs import new_batch_profile
from paper_extraction.selection import LeanChoice
from paper_extraction.segment_evidence import SegmentLeads
from paper_extraction.writer_patch import apply_patch, repair_targets
from test_calabi_routes import APPLICATION, MAIN
from test_current_workflow import claim


def field(text, candidate, loc):
    return {"text": text, "candidate_id": candidate, "loc": loc}


def manuscript(rows):
    return {"title": "Calabi flow", "authors": ["Fixture Author"],
            "paragraph": "Consider the Calabi flow.", "history": [], "shared_definitions": [],
            "groups": [{"theorem": row["text"], "definitions": []} for row in rows],
            "references": [{"key": "source", "text": "Fixture Author, Calabi flow."}],
            "evidence_links": [{"block_id": "paragraph", "evidence_ids": [rows[0]["loc"]]}] + [
                {"block_id": f"groups.{i}.theorem", "evidence_ids": [row["loc"]]} for i, row in enumerate(rows)],
            "formal_sources": {f"groups.{i}.theorem": {"source": r"\cite[Theorem~1.1]{source}" if i == 0
                                                       else r"\cite[Corollary~1.2]{source}"} for i in range(len(rows))},
            "core_bindings": {f"groups.{i}.theorem": row["candidate_id"] for i, row in enumerate(rows)}}


class Replay:
    def __init__(self, mode):
        self.mode = mode
        self.calls = []
        self.review_count = 0
        self.main = self.application = ""

    def structured(self, schema, instructions, packet, image=None):
        self.calls.append(schema)
        if schema is DirectResult:
            self.main = next(r["id"] for r in packet["passages"] if MAIN in r["text"])
            self.application = next(r["id"] for r in packet["passages"] if APPLICATION in r["text"])
            if self.mode == "summary":
                return schema(cores=[], summary={"question": "研究什么范围？", "reason": "此测试快照未给出独立的主要成果。",
                                                 "kind": "no_qualifying_result", "evidence_ids": [self.main]})
            rows = [field(MAIN, "c1", self.main)]
            if self.mode != "omitted":
                rows.append(field(APPLICATION.replace(r"n\ge10", "n>10") if self.mode == "quantifier" else APPLICATION,
                                  "c2", self.application))
            cores = [{"candidate_id": r["candidate_id"], "kind": "current" if i == 0 else "application",
                      "claim": r["text"], "source_number": "Theorem 1.1" if i == 0 else "Corollary 1.2",
                      "independent_question": "Which initial metrics develop singularities?",
                      "source_reason": "A finite-time singularity in the specified dimension is established.",
                      "evidence_ids": [r["loc"]]} for i, r in enumerate(rows)]
            return schema(cores=cores, draft=manuscript(rows))
        if schema is SegmentLeads:
            assert self.mode == "omitted" and self.application in {r["id"] for r in packet["passages"]}
            return schema(candidates=[{**claim(self.application), "kind": "application", "claim": APPLICATION,
                                       "gain": "An independent application in each dimension.",
                                       "source_number": "Corollary 1.2"}], uncertainties=[])
        if schema is LeanChoice:
            assert {c["candidate_id"] for c in packet["retained_cores"]} == {"c1"}
            return schema(selected=[{"candidate_id": "c2", "role": "application",
                                     "independent_question": "Which dimensions?", "source_reason": "An independent application",
                                     "b2_source_facts": [{"source_id": self.application, "fact": "Includes dimension ten and higher."}],
                                     "evidence_ids": [self.application]}])
        if schema is AuthoritativeDraft:
            assert set(packet["approved_cores"]) == {"c1", "c2"}
            return schema.model_validate(manuscript([field(MAIN, "c1", self.main), field(APPLICATION, "c2", self.application)]),
                                         context=packet)
        if schema is AddedCores:
            assert set(packet["requested_cores"]) == {"c2"} and "previous_draft" not in packet
            return schema(cores=[{"candidate_id": "c2", "text": APPLICATION, "evidence_ids": [self.application],
                                  "formal_source": {"source": r"\cite[Corollary~1.2]{source}"}}])
        if schema is EvidenceReview:
            self.review_count += 1
            if self.review_count == 1 and self.mode in {"omitted", "quantifier"}:
                missing = self.mode == "omitted"
                issue = {"kind": "missing_required_content" if missing else "incorrect_claim",
                         "draft_field": "groups" if missing else "groups.1.theorem",
                         "current_excerpt": "" if missing else "n>10", "source_excerpt": APPLICATION,
                         "evidence_id": self.application, "required_change": "Retain the application in all stated dimensions.",
                         "operation": "insert" if missing else "replace",
                         "replacement_text": APPLICATION if missing else r"n\ge10"}
                # Even if the reviewer wrongly proposes a draft insertion, admission must handle it.
                return schema(verdict={"route": "draft", "issues": [issue], "evidence_ids": []})
            return schema(verdict={"route": "pass", "issues": [], "evidence_ids": []})
        if schema is DraftPatch:
            target, = packet["repair_targets"]
            assert target["block_id"] == "groups.1.theorem" and "previous_draft" not in packet
            return schema(replacements=[{"block_id": target["block_id"], "text": APPLICATION,
                                         "evidence_ids": [self.application]}])
        assert schema is Review and image.startswith(b"\x89PNG")
        return schema(passed=True, issues=[])


@pytest.mark.parametrize("mode", ["first", "summary", "omitted", "quantifier"])
def test_lean_short_source(mode, tmp_path, monkeypatch):
    if mode != "summary" and not shutil.which("pdflatex"):
        pytest.skip("formal delivery requires pdfLaTeX")
    source = tmp_path / "source" / "352.tex"
    source.parent.mkdir()
    source.write_text(r"\documentclass{article}\begin{document}" + "\n" + r"\section{Theorem 1.1}" + "\n" +
                      MAIN + "\n" + r"\section{Corollary 1.2}" + "\n" + APPLICATION + "\n" + r"\end{document}")
    replay = Replay(mode)
    monkeypatch.setattr(agent, "Models", lambda _: replay)
    events = []
    result = agent.run_paper(tmp_path / "case", source, source.parent, new_batch_profile(),
                             lambda stage, data: events.append((stage, data)))
    assert result["status"] == "completed", result.get("reason", result)
    assert replay.calls[0] is DirectResult
    if mode == "summary":
        assert replay.calls == [DirectResult] and result["draft_round"] == 0
        assert not any(stage in {"draft", "compile", "review", "visual"} for stage, _ in events)
        assert [p.name for p in Path(result["result_dir"]).iterdir()] == ["summary.md"]
        return
    assert result["delivery"]["status"] == "reviewed"
    assert result["draft_round"] == (1 if mode == "first" else 2)
    assert result["draft"]["core_bindings"] == {"groups.0.theorem": "c1", "groups.1.theorem": "c2"}
    assert APPLICATION in result["draft"]["groups"][1]["theorem"]
    assert (Path(result["result_dir"]) / "introduction.pdf").read_bytes().startswith(b"%PDF")
    assert (LeanChoice in replay.calls) == (mode == "omitted")
    assert (DraftPatch in replay.calls) == (mode == "quantifier")


def test_direct_never_truncates_or_discards_scans():
    small = [{"id": "t1", "text": "a"}]
    assert fits_direct(small, "rules", {"passages": small}, {})
    medium = [{"id": "t1", "text": "a" * 68000}]
    assert fits_direct(medium, "rules", {"passages": medium}, {"context_window": 131072})
    assert not fits_direct(medium, "rules", {"passages": medium}, {"context_window": 32768})
    for source in ([{"id": "p1", "text": "", "page": 1}], [{"id": "t1", "text": "a" * 65536}],
                   [{"id": "p1", "text": "(cid:32)", "page": 1}]):
        assert not fits_direct(source, "rules", {"passages": source}, {})


def test_local_deletions_can_be_mixed_with_content_replacements():
    draft = manuscript([field(MAIN, "c1", "s1")])
    draft["groups"][0]["paragraph"] = "An unnecessary restatement."
    draft["evidence_links"].append({"block_id": "groups.0.paragraph", "evidence_ids": ["s1"]})
    issues = [{"kind": "improper_selection", "draft_field": "groups.0.paragraph", "current_excerpt": "An unnecessary restatement.",
               "operation": "remove", "replacement_text": "", "required_change": "Delete repetition.",
               "source_excerpt": MAIN, "evidence_id": "s1"},
              {"kind": "incorrect_claim", "draft_field": "groups.0.theorem", "current_excerpt": "singularity",
               "operation": "replace", "replacement_text": "singularity in finite time", "required_change": "Clarify the time.",
               "source_excerpt": MAIN, "evidence_id": "s1"}]
    review = AdmissionReview(route="draft", issues=issues, evidence_ids=[])
    targets = repair_targets(draft, {"passed": False, "issues": review.repair_issues(), "defects": issues,
                                     "route": "draft", "stage": "review"}, {"s1"})
    assert len(targets) == 2 and targets[0]["remove"]
    patch = DraftPatch(replacements=[{"block_id": "groups.0.paragraph", "text": "", "evidence_ids": ["s1"]},
                                     {"block_id": "groups.0.theorem", "text": MAIN + " The time is finite.",
                                      "evidence_ids": ["s1"]}])
    source = {"id": "s1", "text": MAIN}
    out = apply_patch(draft, patch, {"repair_targets": targets, "passages": [source]}, {"s1": source})
    assert not out["groups"][0]["paragraph"] and len(out["evidence_links"]) == 2
    assert out["core_bindings"] == draft["core_bindings"] and out["formal_sources"] == draft["formal_sources"]


def test_projection_rebinds_previous_headings_without_touching_source_or_math():
    from paper_extraction.lean_workflow import project_cores
    old = manuscript([field(MAIN, "c1", "s1"), field(APPLICATION, "c2", "s2")])
    old["history"] = ["Historic comparison.", "Another comparison."]
    old["core_bindings"].update({"history.0": "c3", "history.1": "c4"})
    old["formal_sources"].update({"history.0": {"source": r"\cite[Lemma~2.1]{source}"},
                                  "history.1": {"source": r"\cite{source}"}})
    old["evidence_links"].extend([{"block_id": "history.0", "evidence_ids": ["h1"]},
                                  {"block_id": "history.1", "evidence_ids": ["h2"]},
                                  {"block_id": "groups.1.paragraph", "evidence_ids": ["s2"]}])
    old["groups"][1]["paragraph"] = r"Theorem B extends Theorem A. Compare Theorem 2 and source Theorem 2.1, \cite[Theorem 2]{source}."
    out = project_cores(old, {"c2": {"kind": "current"}, "c4": {"kind": "historical"}})
    assert out["groups"][0]["paragraph"] == r"Theorem A extends the source paper’s Lemma~2.1. Compare Theorem 1 and source Theorem 2.1, \cite[Theorem 2]{source}."
    assert old["groups"][1]["paragraph"].startswith("Theorem B")
    assert next(r for r in out["evidence_links"] if r["block_id"] == "groups.0.paragraph")["evidence_ids"] == ["s2", "h1"]


def test_nonenglish_paper_can_explicitly_announce_a_principal_lemma():
    from paper_extraction.lean_workflow import IndependentRole, independent_role_supported
    quote = "Dans cet article, nous établissons le Lemme 2.3 sur la représentation spectrale."
    records = {"s1": {"text": quote, "section": "Introduction"}, "s2": {"text": "Lemme 2.3. Les mesures ..."}}
    basis = IndependentRole(role="principal_result", quote=quote, source_id="s1")
    assert independent_role_supported("current", "2.3", "Représentation spectrale des mesures.", ["s2"], basis, records)


def test_central_only_recovery_preserves_other_cores_and_passes_precise_source_reading():
    from paper_extraction.selection import select_contributions
    records = [{"id": "p1", "page": 1, "text": "An application is Corollary 3.4. We first prove Theorem 2.1."},
               {"id": "p2", "page": 2, "text": "COROLLARY 3.4. - Suppose T is ... [unreadable formula]"}]
    candidates = [dict(claim("p1"), source_number="Theorem 2.1"),
                  dict(claim("p2"), source_number="3.4", claim="Integration by parts.", evidence_ids=["p1", "p2"], kind="application")]
    class Selector:
        count = 0
        def structured(self, schema, instructions, packet):
            self.count += 1
            key = "c1" if self.count == 1 else "c2"
            loc = "p1" if self.count == 1 else "p2"
            choice = {"candidate_id": key, "independent_question": "Which identity holds?", "evidence_ids": [loc],
                      "source_reason": "The introduction highlights this application.",
                      "b2_source_facts": [{"source_id": loc, "fact": "Extension to the given boundary data."}]}
            if self.count == 1:
                return schema(selected=[choice], unresolved=[{"candidate_id": "c2","source_reason": "The exact formula is missing."}],
                              central_open_ids=["c2"])
            assert [c["candidate_id"] for c in packet["candidate_register"]] == ["c2"]
            assert [c["candidate_id"] for c in packet["retained_cores"]] == ["c1"]
            assert packet["checked_source_images"][0]["transcription"] == "Source-specific integral identity."
            return schema(selected=[choice])
    def resolve(checks, origins):
        assert list(origins) == ["p2"] and len(checks) == 1 and checks[0].read_image
        return {"source_images": [{"source_id": "p2", "page": 2, "region": None, "transcription": "Source-specific integral identity."}]}
    selector = Selector()
    rows, note = select_contributions(records, candidates, selector, current=True, lean=True, resolve_central=resolve)
    assert selector.count == 2 and note is None
    assert [row["candidate_id"] for row in rows if row["decision"] == "admitted"] == ["c1","c2"]
    assert rows[1]["source_readings"][0]["source_id"] == "p2"
