"""Replay fixed editorial decisions against the same source and draft contracts as genuine runs.

333/352 use short paraphrases of the Introduction_v13 calibration manuscripts:
333 Theorem 1.1 and Proposition 5.1; 352 Theorem 1.1, Corollary 1.2 and §6.5.
0158 uses Cegrell, Ann. Inst. Fourier 54 (2004), pp. 159, 173–174, 177.
These are admission/identity regressions, not a measurement of live LLM judgment.
"""
from copy import deepcopy

import pytest
from pydantic import ValidationError

from paper_extraction.lean_workflow import DirectResult, approved_cores, direct_admission, selection_change, verify_bindings
from paper_extraction.profile import render_tex
from paper_extraction.profile_variants import CURRENT_PROFILE


def _direct_decision(results, supporting=(), roles=None):
    """The offline model's approved subset plus the full source, including supporting lemmas."""
    cores, groups, history, links, bindings, sources, passages = [], [], [], [], {}, {}, []
    for i, (location, label, kind, claim, increment) in enumerate(results):
        candidate_id = f"c{i + 1}"
        if kind == "historical":
            path = f"history.{len(history)}"
            history.append(claim)
        else:
            path = f"groups.{len(groups)}.theorem"
            groups.append({"theorem": claim, "definitions": []})
        cores.append({"candidate_id": candidate_id, "kind": kind, "source_number": label,
                      "claim": claim, "independent_question": increment, "source_reason": increment,
                      "independent_role": (roles or {}).get(candidate_id), "evidence_ids": [location]})
        links.append({"block_id": path, "evidence_ids": [location]})
        bindings[path] = candidate_id
        sources[path] = {"source": r"\cite[" + label + "]{source}"}
        passages.append({"id": location, "text": label + ". " + claim})
    first = results[0][0]
    links.append({"block_id": "paragraph", "evidence_ids": [first]})
    passages.extend({"id": location, "text": text} for location, text in supporting)
    draft = {"title": "Calibration", "authors": ["Source author"], "paragraph": "We consider the following problem.",
             "history": history, "shared_definitions": [], "groups": groups, "evidence_links": links,
             "core_bindings": bindings, "formal_sources": sources,
             "references": [{"key": "source", "text": "Supplied source."}]}
    return DirectResult.model_validate({"cores": cores, "draft": draft}, context={"passages": passages}), passages


def test_333_construction_is_available_as_evidence_but_cannot_be_published_as_a_core():
    result, passages = _direct_decision([
        ("p2-1", "Theorem 1.1", "current",
         r"Every closed smooth Riemannian surface admits a smooth isometric immersion in $\mathbb R^4$.",
         "The previously known smooth global immersion target had dimension five."),
    ], [("p15-1", "Proposition 5.1 corrects an approximate immersion within the proof of the existence theorem.")])
    approved = approved_cores([c.register_row() for c in result.cores])
    draft = result.draft.model_dump()
    assert "Proposition 5.1" not in render_tex({**draft, "editorial_profile": CURRENT_PROFILE})
    source_ids = {p["id"] for p in passages}
    verify_bindings(draft, approved, source_ids)
    # Even a genuine source location does not make this extra formal result admitted.
    draft["core_bindings"]["groups.1.theorem"] = "c2"
    draft["evidence_links"].append({"block_id": "groups.1.theorem", "evidence_ids": ["p15-1"]})
    with pytest.raises(ValueError, match="approved core"):
        verify_bindings(draft, approved, source_ids)


def test_352_distinct_dimension_application_survives_without_its_construction():
    result, _ = _direct_decision([
        ("p1-1", "Theorem 1.1", "current",
         r"A smooth $U(10)$-invariant form in the Fubini--Study class on $\mathbb{CP}^{10}$ "
         "has a Calabi flow with a finite-time singularity.",
         "Exhibits a finite smooth maximal time in this Kähler class."),
        ("p2-1", "Corollary 1.2", "application",
         r"For every integer $n\ge10$, $\mathbb{CP}^{10}\times\mathbb{CP}^{n-10}$ admits a Calabi flow "
         r"with a finite-time singularity in a class containing a K\"ahler--Einstein metric.",
         "Answers a separate existence question in every dimension at least ten."),
    ], [("p-proof", "The proof in section 6.5 constructs a particular rescaled representative.")])
    tex = render_tex({**result.draft.model_dump(), "editorial_profile": CURRENT_PROFILE})
    assert tex.count(r"\begin{theorem}") == 2 and r"n\ge10" in tex
    assert "Corollary 1.2" in tex and "rescaled representative" not in tex


def test_0158_reviewer_must_nominate_unselected_historical_and_current_results():
    result, passages = _direct_decision([
        ("p2-1", "Theorem 4.5", "current",
         "The energy class gives the largest domain for the operator under the stated continuity and maximum conditions.",
         "Identifies the domain of the extension beyond locally bounded potentials."),
    ], [("p4-1", "Theorem 3.2 concerns a product of currents under integrability hypotheses."),
        ("p16-1", "The paper reports the Bedford--Taylor comparison Theorem 5.12 and proves Lemma 5.14."),
        ("p20-1", "Lemma 5.14 is used to handle the non-pluripolar component in the proof of Theorem 6.2.")])
    draft = result.draft.model_dump()
    registered = {row["id"] for row in passages}
    approved = approved_cores([core.register_row() for core in result.cores])
    for path, label, location in (("groups.1.theorem", "Theorem 3.2", "p4-1"),
                                  ("history.0", "Theorem 5.12", "p16-1"),
                                  ("groups.1.theorem", "Lemma 5.14", "p16-1")):
        assert selection_change({"draft_field": path, "operation": "insert", "current_excerpt": ""}, draft)
        unauthorized = deepcopy(draft)
        unauthorized["core_bindings"][path] = "c2"
        unauthorized["evidence_links"].append({"block_id": path, "evidence_ids": [location]})
        with pytest.raises(ValueError, match="approved core"):
            verify_bindings(unauthorized, approved, registered)
        assert label not in render_tex({**draft, "editorial_profile": CURRENT_PROFILE})


def test_five_independently_approved_cores_are_not_truncated_or_reused():
    # A synthetic collection of unrelated standard statements tests capacity only.
    statements = [
        ("Every positive integer is a sum of four squares.", "Represents integers by a quadratic form."),
        ("Every planar graph admits a proper coloring with four colors.", "Bounds the planar chromatic number."),
        ("Connected closed orientable surfaces are classified up to homeomorphism by genus.", "Gives a surface classification."),
        ("Every continuous self-map of the closed ball has a fixed point.", "Gives a topological existence theorem."),
        ("There are infinitely many primes.", "Establishes the infinitude of primes."),
    ]
    result, passages = _direct_decision([(f"p{i}-1", f"Theorem {i}", "current", text, reason)
        for i, (text, reason) in enumerate(statements, 1)])
    draft = result.draft.model_dump()
    assert render_tex({**draft, "editorial_profile": CURRENT_PROFILE}).count(r"\begin{theorem}") == 5
    draft["core_bindings"]["groups.4.theorem"] = draft["core_bindings"]["groups.0.theorem"]
    with pytest.raises(ValidationError, match="distinct approved core"):
        DirectResult.model_validate({"cores": result.cores, "draft": draft}, context={"passages": passages})


@pytest.mark.parametrize("quote", [
    "By Lemma 5.14, it is enough to show that for the unique solution",
    "It follows from Lemma 5.14 and Lemma 6.1 that for each",
])
def test_0158_proof_invocations_do_not_certify_independent_principal_status(quote):
    results = [
        ("p16-1", "Lemma 5.14", "current", "Certain finite measures admit a unique potential.",
         "Gives an ingredient in the proof."),
        ("p20-1", "Theorem 6.2", "current", "The extension solves the stated measure problem.",
         "Solves the prescribed measure problem."),
    ]
    note = {"role": "principal_result", "source_id": "p17-1", "quote": quote}
    result, passages = _direct_decision(results, [("p17-1", quote)], {"c1": note})
    candidates, accepted, summary = direct_admission(result, {p["id"]: p for p in passages})
    assert not summary and [c["decision"] for c in candidates] == ["excluded", "admitted"]
    assert accepted.core_bindings == {"groups.0.theorem": "c2"}
    assert accepted.formal_sources["groups.0.theorem"].source == r"\cite[Theorem 6.2]{source}"
    assert "5.14" not in render_tex({**accepted.model_dump(), "editorial_profile": CURRENT_PROFILE})


def test_central_lemma_and_relevant_historical_comparison_are_admissible():
    # Synthetic positive control: explicit, attributable author descriptions of role.
    principal = "Our main result is Lemma 2 on simultaneous polynomial representation."
    comparison = "Our representation extends Anderson's compact approximation Theorem 1 beyond bounded domains."
    results = [
        ("intro", "Lemma 2", "current", "Each family admits simultaneous polynomial representation.",
         "Resolves simultaneous representation."),
        ("baseline", "Theorem 1", "historical", "Anderson's compact approximation holds on bounded domains.",
         "Supplies the historical baseline for the extension."),
        ("different", "Theorem 17", "historical", "Brown proved the equivalence of Gaussian moment laws.",
         "Determines moment laws for the Gaussian class."),
    ]
    roles = {"c1": {"role": "principal_result", "source_id": "goal", "quote": principal},
             "c2": {"role": "explicit_comparison", "source_id": "comparison", "quote": comparison},
             # This is a genuine quote; it describes Anderson's work, not Brown's.
             "c3": {"role": "explicit_comparison", "source_id": "comparison", "quote": comparison}}
    result, passages = _direct_decision(results, [("goal", principal), ("comparison", comparison)], roles)
    candidates, accepted, _ = direct_admission(result, {p["id"]: p for p in passages})
    assert [c["decision"] for c in candidates] == ["admitted", "admitted", "excluded"]
    assert accepted.core_bindings == {"groups.0.theorem": "c1", "history.0": "c2"}
