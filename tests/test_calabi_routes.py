"""Replay paper 352's two calibrated editorial defects with fixed offline model replies."""
import shutil
from pathlib import Path

import pytest

from paper_extraction import agent
from paper_extraction.profile_variants import PREVIOUS_PROFILE
from paper_extraction.models import AdmissionReview, CurrentDraft, DraftPatch, TextFindings
from paper_extraction.profile_inputs import new_batch_profile
from test_current_workflow import LocalModel, claim


# Self-contained paraphrases of the selected parts of Theorem 1.1 and Corollary
# 1.2 from the 352 calibration. The TeX fixture is a routing test, not the paper.
MAIN = (r"A smooth $U(10)$-invariant initial metric in the Fubini--Study class on "
        r"$\mathbb{CP}^{10}$ develops a finite-time Calabi flow singularity.")
APPLICATION = (r"For every integer $n\ge10$, the product $\mathbb{CP}^{10}\times\mathbb{CP}^{n-10}$ "
               r"admits an initial metric whose Calabi flow develops a finite-time singularity "
               r"in a class containing a K\"ahler--Einstein metric.")


class CalabiReplay(LocalModel):
    def __init__(self, defect):
        super().__init__()
        self.defect = defect
        self.main_id = None
        self.application_id = None
        self.extractions = []
        self.reviews = 0

    def structured(self, schema, instructions, packet, image=None):
        if schema is TextFindings:
            rows = packet["passages"]
            self.extractions.append([row["id"] for row in rows])
            items = []
            for row in rows:
                if MAIN in row["text"]:
                    self.main_id = row["id"]
                    items.append({**claim(self.main_id), "claim": MAIN, "source_number": "Theorem 1.1"})
                if APPLICATION in row["text"]:
                    self.application_id = row["id"]
                    if self.defect != "omitted" or packet["targeted_issue"]:
                        items.append({**claim(self.application_id), "kind": "application", "claim": APPLICATION,
                                      "assumptions": "n is an integer at least 10", "source_number": "Corollary 1.2"})
            return schema(candidates=items, uncertainties=[])
        if schema is CurrentDraft:
            groups = [{"theorem": MAIN + r" \cite[Theorem~1.1]{source}", "definitions": []}]
            links = [("paragraph", self.main_id), ("groups.0.theorem", self.main_id)]
            if any(c["kind"] == "application" for c in packet["candidates"]):
                text = APPLICATION.replace(r"n\ge10", "n>10") if self.defect == "quantifier" else APPLICATION
                groups.append({"theorem": text + r" \cite[Corollary~1.2]{source}", "definitions": []})
                links.append(("groups.1.theorem", self.application_id))
            return schema(title="Calabi flow calibration", authors=["Fixture Author"], paragraph="Consider Calabi flow.",
                          history=[], groups=groups, shared_definitions=[],
                          references=[{"key": "source", "text": "Calabi flow calibration, source excerpts."}],
                          evidence_links=[{"block_id": field, "evidence_ids": [loc]} for field, loc in links])
        if schema is AdmissionReview:
            self.reviews += 1
            if self.reviews == 1:
                omitted = self.defect == "omitted"
                return schema(route="extract" if omitted else "draft", evidence_ids=[self.application_id] if omitted else [],
                              issues=[{"draft_field": "groups" if omitted else "groups.1.theorem",
                                       "kind": "missing_required_content" if omitted else "incorrect_claim",
                                       "current_excerpt": "" if omitted else "n>10", "source_excerpt": APPLICATION,
                                       "evidence_id": self.application_id, "replacement_text": "" if omitted else r"n\ge10",
                                       "operation": "recover" if omitted else "replace",
                                       "required_change": "Recover the independent dimension consequence." if omitted
                                       else "Include the endpoint dimension 10 in Corollary 1.2."}])
            return schema(route="pass", issues=[], evidence_ids=[])
        if schema is DraftPatch:
            target, = packet["repair_targets"]
            assert target["block_id"] == "groups.1.theorem"
            assert self.application_id in target["required_evidence_ids"]
            return schema(replacements=[{"block_id": target["block_id"], "evidence_ids": [self.application_id],
                                         "text": target["current_text"].replace("n>10", r"n\ge10")}])
        return super().structured(schema, instructions, packet, image)


@pytest.mark.skipif(not shutil.which("pdflatex"), reason="end-to-end delivery needs pdfLaTeX")
@pytest.mark.parametrize("defect", ["omitted", "quantifier"])
def test_352_independent_application_is_recovered_or_patched(tmp_path, monkeypatch, defect):
    source = tmp_path / "source" / "352.tex"
    source.parent.mkdir()
    source.write_text(r"\documentclass{article}\title{Calabi flow calibration}\author{Fixture Author}" + "\n"
                      + r"\begin{document}\maketitle" + "\n" + r"\section{Theorem 1.1}" + "\n" + MAIN + "\n"
                      + r"\section{Corollary 1.2}" + "\n" + APPLICATION + "\n" + r"\end{document}", encoding="utf-8")
    replay = CalabiReplay(defect)
    monkeypatch.setattr(agent, "Models", lambda _: replay)
    events = []
    result = agent.run_paper(tmp_path / "case", source, source.parent, new_batch_profile(PREVIOUS_PROFILE),
                             lambda stage, data: events.append((stage, data)))
    assert result["status"] == "completed", result.get("reason")
    assert result["delivery"]["status"] == "reviewed" and result["draft_round"] == 2
    extract_events = [data for stage, data in events if stage == "extract"]
    primary, = [data for data in extract_events if not data["targeted"]]
    assert {replay.main_id, replay.application_id} <= set(primary["locations"])
    assert replay.main_id != replay.application_id
    targeted = [[replay.application_id]] if defect == "omitted" else []
    assert [data["locations"] for data in extract_events if data["targeted"]] == targeted
    assert replay.extractions == [primary["locations"], *targeted]
    assert result["reviews"][0]["route"] == ("extract" if defect == "omitted" else "draft")
    assert [d["mode"] for s, d in events if s == "draft"] == ["full", "full" if defect == "omitted" else "patch"]
    assert len(result["draft"]["groups"]) == 2
    assert APPLICATION in result["draft"]["groups"][1]["theorem"]
    assert {"block_id": "groups.1.theorem", "evidence_ids": [replay.application_id]} in result["evidence_links"]
    tex = (Path(result["result_dir"]) / "introduction.tex").read_text(encoding="utf-8")
    assert r"n\ge10" in tex and "n>10" not in tex
