import json
import os
from pathlib import Path
import shutil
import unittest
from tempfile import TemporaryDirectory

from paper_extraction import pdfops, profile
from paper_extraction.sources import SourceIndex
from paper_extraction.profile_assets.tools import v13_check


def test_grave_accent_in_paper_title():
    draft = synthetic_draft()
    draft["title"] = r"Monge-Amp\`ere measures"
    assert r"\paperheader{Monge-Amp\`ere measures}" in profile.render_tex(draft)


def synthetic_draft():
    return {
        "title": "Positive real numbers", "authors": ["Fixture Author"],
        "paragraph": "We consider positive real numbers.",
        "history": [r"A positive real number has a positive square. \cite[p. 1]{source}"],
        "shared_definitions": [],
        "groups": [{"definitions": [], "theorem": r"If $x>0$, then $x+x>0$. \cite[Theorem 2]{source}",
                    "paragraph": ""}],
        "references": [{"key": "source", "text": "Fixture Author, Positive real numbers."}],
    }


class ProfileTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "nt" and shutil.which("pdflatex"), "requires Windows and pdfLaTeX")
    def test_build_and_recovery_with_nested_windows_job_path_and_long_source_title(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            source = base / (("An-equation-for-large-scale-mathematical-geometry-" * 3) + ".pdf")
            source.write_bytes(b"Synthetic source snapshot.")
            runs = base / ("r" * max(1, 141 - len(str(base))))
            draft = {**synthetic_draft(), "editorial_profile": "reader"}

            case = profile.build_case(runs, source, None, draft)
            binding = json.loads((case / "input-manifest.json").read_text("utf-8"))
            status = json.loads((case / "build-status.json").read_text("utf-8"))
            artifact = case / status["diagnostic_artifacts"]["directory"] / "introduction.pdf"
            self.assertEqual(binding["case_id"], source.stem)
            self.assertTrue(case.name.endswith("--" + binding["snapshot"]["sha256"][:12]))
            self.assertLess(len(case.name), 100)
            self.assertTrue(artifact.is_file())
            self.assertEqual(profile.recover_draft_case(runs, source, None, draft), case)

            bundle = profile.profile_bundle(profile.ASSETS, runs, v13_check.EXPANDED_RELEASE)
            another = v13_check.allocate(bundle, source, runs)
            self.assertTrue(another.name.endswith("--run-002"))
            self.assertEqual(profile.recover_draft_case(runs, source, None, draft), case)

    @unittest.skipUnless(shutil.which("pdflatex"), "requires pdfLaTeX")
    def test_math_spacing_with_connective_passes_and_stacked_spacing_is_located(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source.md"
            source.write_text("Synthetic source.", "utf-8")
            draft = synthetic_draft()
            draft["groups"][0]["theorem"] = (
                r"For a real number $x$, \[ x>0\quad\Longrightarrow\quad x+x>0. \] \cite[Theorem 2]{source}"
            )
            case = profile.build_case(base / "runs", source, None, draft)
            self.assertTrue((case / "introduction.pdf").is_file())

            theorem = draft["groups"][0]["theorem"]
            draft["groups"][0]["theorem"] = theorem.replace(r"\quad\Longrightarrow", r"\quad\qquad\Longrightarrow")
            with self.assertRaises(profile.BuildFailure) as failed:
                profile.build_case(base / "runs", source, None, draft)
            self.assertFalse(failed.exception.technical)
            self.assertIn("theorem_1_manual_spacing", str(failed.exception))
            self.assertEqual([row["draft_field"] for row in failed.exception.repair_targets], ["groups.0.theorem"])

    def test_missing_source_snapshot_is_a_technical_build_failure(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            with self.assertRaises(profile.BuildFailure) as failed:
                profile.build_case(base / "runs", base / "missing.md", None, synthetic_draft())
            self.assertTrue(failed.exception.technical)
            self.assertEqual(failed.exception.repair_targets, [])

    def test_printed_numbers_candidate_bindings_and_local_definition_scope(self):
        draft = synthetic_draft()
        draft["shared_definitions"] = ["First concept.", "Second concept."]
        draft["history"] += ["Another comparison."]
        draft["groups"] = [{"definitions": [], "theorem": "One."},
                           {"definitions": ["Third concept."], "theorem": "Two."}]
        candidates = [{"candidate_id": "c2", "kind": "definition", "needed_by": ["c5"],
                       "evidence_ids": ["p2-1"], "decision_reason": "Previous model's judgment"},
                      {"candidate_id": "c5", "source_number": "Theorem 6"}]
        data = profile.renderer_context(draft, candidates)
        self.assertEqual(data["visible_headings"], {
            "shared_definitions.0": "Definition 1", "shared_definitions.1": "Definition 2",
            "history.0": "Theorem A", "history.1": "Theorem B", "groups.0.theorem": "Theorem 1",
            "groups.1.definitions.0": "Definition 3", "groups.1.theorem": "Theorem 2"})
        self.assertEqual(data["result_groups"][0]["definitions_printed_in_this_group"], [])
        self.assertEqual(data["result_groups"][1]["definitions_printed_in_this_group"], ["groups.1.definitions.0"])
        self.assertEqual(data["source_candidate_dependencies"][0]["needed_by_candidates"],
                         [{"candidate_id": "c5", "source_number": "Theorem 6"}])
        self.assertNotIn("decision_reason", str(data))
        self.assertEqual(profile.renderer_context(None)["visible_headings"], {})

    def test_standalone_leading_locator_moves_once_without_rewriting_prose_attribution(self):
        draft = synthetic_draft()
        citation = r"\cite[Theorem 3]{source}"
        for separator in (" ", ". "):
            draft["groups"][0]["theorem"] = citation + separator + "Let the hypotheses hold."
            tex = profile.render_tex(draft)
            self.assertIn(r"\begin{theorem}[{" + citation + "}]\nLet the hypotheses hold.", tex)
            self.assertEqual(tex.count(citation), 1)
            self.assertEqual(profile.render_tex(draft, _legacy_citations=True).count(citation), 2)
        draft["groups"][0]["theorem"] = citation + " proves a statement about positive reals."
        self.assertIn("\n" + draft["groups"][0]["theorem"], profile.render_tex(draft))

    @unittest.skipUnless(shutil.which("pdflatex"), "requires the v13 TeX toolchain")
    def test_recovery_recognizes_the_original_pre_deduplication_tex(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source.md"
            source.write_text("Synthetic source.", "utf8")
            draft = synthetic_draft()
            draft["groups"][0]["theorem"] = r"\cite[Theorem 3]{source} Let the hypotheses hold."
            runs = base / "runs"
            case = v13_check.allocate(profile.ASSETS, source, runs)
            old_text = profile.render_tex(draft, _legacy_citations=True)
            (case / "introduction.tex").write_text(old_text, "utf8")
            self.assertTrue(v13_check.build(profile.ASSETS, case)["engineering_accepted"])
            self.assertNotEqual(old_text, profile.render_tex(draft))
            self.assertEqual(profile.recover_draft_case(runs, source, None, draft), case)
            self.assertEqual((case / "introduction.tex").read_text("utf8"), old_text)

    def test_inputs_bound_to_v13_and_current_checker(self):
        record = v13_check.verify(profile.ASSETS)
        self.assertEqual(record["sha256"]["preamble-v13.tex"],
                         "c363b47a067f7206c1c7ac56e77a913ff39f20f206d77760cf8ab02fad902590")
        self.assertNotEqual(record["checker_sha256"],
                            "6588488b32c6e491df64605502344af575137f2d0f18509115bca013e5568c96")

    def test_renderer_accepts_math_and_citations_in_source_notation(self):
        draft = synthetic_draft()
        draft["groups"][0]["theorem"] = (
            r"For $s,t\in\mathbb{R}$ assume \begin{conditions}&s>0,\\&t>0."
            r"\end{conditions} Then $s+t>0$. \cite[\S2]{source}")
        tex = profile.render_tex(draft)
        self.assertIn(r"\begin{conditions}&s>0,\\&t>0.\end{conditions}", tex)
        self.assertIn(r"\begin{theorem}[{\cite[\S2]{source}}]", tex)

    def test_renderer_rejects_untrusted_file_operations_and_structure_changes(self):
        for payload in [r"\input{private}", r"\write18{cmd}", r"\end{theorem}",
                        r"\csname input\endcsname", r"^^5cinput{private}", r"} outside {"]:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                draft = synthetic_draft()
                draft["groups"][0]["theorem"] += " " + payload
                profile.render_tex(draft)

    def test_exact_template_wrapper_keeps_its_caption_and_checks_its_body(self):
        draft = synthetic_draft()
        draft["history"] = [r"\begin{historical}[Prior author, {\cite[p.~1]{source}}]"
                            r"The square of a positive real is positive.\end{historical}"]
        tex = profile.render_tex(draft)
        self.assertEqual(tex.count(r"\begin{historical}"), 1)
        self.assertIn(r"Prior author, {\cite[p.~1]{source}}", tex)
        for payload in [r"\begin{definition}Wrong field.\cite{source}\end{definition}",
                        r"\begin{historical}One.\end{historical} outside "
                        r"\begin{historical}Two.\cite{source}\end{historical}",
                        r"\begin{historical}[\input{secret}]\cite{source}\end{historical}"]:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                draft["history"] = [payload]
                profile.render_tex(draft)

    def test_missing_source_locator_and_unknown_key(self):
        for value in ["The historical result.", r"The historical result. \cite[p. 1]{unlisted}"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                draft = synthetic_draft()
                draft["history"] = [value]
                profile.render_tex(draft)

    def test_evidence_locator_preserves_attribution_and_compiles_real_source_paths(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "paper"
            chapter = root / "defs" / "a_[1]&b.tex"
            chapter.parent.mkdir(parents=True)
            chapter.write_text("A positive real number is greater than zero.\n", "utf-8")
            entry = root / "main.tex"
            entry.write_text(r"\input{defs/a_[1]&b.tex}", "utf-8")
            catalog = SourceIndex(entry, root, Path(tmp) / "cache").catalog()
            records = {row["id"]: row for row in catalog}
            ids = [row["id"] for row in catalog]
            self.assertEqual(len(ids), 2)
            draft = synthetic_draft()
            draft["history"] = [r"\begin{historical}[As reported by X]A positive number has a positive square.\end{historical}"]
            draft["shared_definitions"] = ["Positive means greater than zero."]
            draft["groups"][0]["definitions"] = ["A square is the product of a number with itself."]
            draft["references"].append({"key": "prior", "text": "X, Reported result."})
            draft["groups"][0]["theorem"] += r" \cite[discussion]{prior}"
            draft["evidence_links"] = [{"block_id": block, "evidence_ids": ids} for block in
                                       ("history.0", "shared_definitions.0", "groups.0.definitions.0", "groups.0.theorem")]
            normalized = profile.bind_formal_citations(draft, records)
            self.assertEqual(draft["shared_definitions"], ["Positive means greater than zero."])
            self.assertEqual(normalized, profile.bind_formal_citations(normalized, records))
            self.assertEqual(normalized["groups"][0]["theorem"], draft["groups"][0]["theorem"])
            tex = profile.render_tex(normalized)
            self.assertIn(r"\begin{historical}[{As reported by X \cite[", tex)
            self.assertIn(r"\texttt{main.tex} L~1; \texttt{defs/a\_\symbol{91}1\symbol{93}\&b.tex} L~1", tex)
            if shutil.which("pdflatex"):
                case = profile.build_case(Path(tmp) / "cases", entry, root, normalized)
                self.assertTrue((case / "introduction.pdf").is_file())

    def test_pdf_physical_pages_and_partial_or_missing_evidence(self):
        draft = synthetic_draft()
        draft["history"] = ["A positive real has a positive square."]
        draft["evidence_links"] = [{"block_id": "history.0", "evidence_ids": ["a", "b", "c", "d"]}]
        records = {key: {"id": key, "page": page} for key, page in zip("abcd", (2, 2, 3, 5))}
        normalized = profile.bind_formal_citations(draft, records)
        self.assertIn(r"\cite[PDF physical pp.~2--3, 5]{source}", normalized["history"][0])
        unicode_file = {"file": "ch/例_1.tex", "line": 14, "end_line": 19}
        tex = profile.render_tex(profile.bind_formal_citations(draft, dict.fromkeys("abcd", unicode_file)))
        self.assertIn(r"URI path \texttt{ch/\%E4\%BE\%8B\_1.tex} L~14--19", tex)
        del records["c"]
        self.assertEqual(draft, profile.bind_formal_citations(draft, records))
        for bad in ({}, {"line": 0, "file": "paper.tex", "end_line": 1}):
            self.assertEqual(draft, profile.bind_formal_citations(draft, {**records, "c": bad}))
        draft["references"][0]["key"] = "earlier"
        self.assertEqual(draft, profile.bind_formal_citations(draft, {**records, "c": {"page": 4}}))

    def test_retry_targets_use_measured_block_order_and_rendered_lines(self):
        draft = synthetic_draft()
        definition = r"For $u$ set $v=u+u$. \cite[Definition 2]{source}"
        draft["shared_definitions"] = [definition]
        draft["groups"][0]["definitions"] = [definition + "\nMore text."]
        blocks = []
        tex = profile.render_tex(draft, block_map=blocks)
        boxed = next(row for row in blocks if row["draft_field"] == "groups.0.definitions.0")
        line = boxed["last"]
        self.assertEqual(tex.splitlines()[line-1], r"\end{definition}")
        detail = (r"Engineering checks failed: no_blocking_diagnostics, v13_reader_format_contract | "
                  rf"introparagraph_1_word_limit:54>45 | Overfull \hbox (75.58548pt too wide) in paragraph at lines {line}--{line}")
        targets = profile.saved_build_targets(draft, [detail])
        self.assertEqual([t["draft_field"] for t in targets], ["paragraph", boxed["draft_field"]])
        self.assertIn("54 prose words", targets[0]["diagnostics"][0])
        self.assertEqual(targets[1]["current_text"], draft["groups"][0]["definitions"][0])
        self.assertIn("75.58548pt", targets[1]["diagnostics"][0])
        self.assertEqual(profile.saved_build_targets(draft, [detail + " | three_page_limit"]), [])
        self.assertEqual(profile.saved_build_targets(draft, [detail + " | theorem_99_word_limit:101>100"]), [])
        log = "INTRO-BLOCK|4|D|800000|700000|1\n"
        measured = profile._build_targets(blocks, [f"./introduction.tex:{line}: Package intro-v13 Error: Block 4 (D) exceeds 7U."], log)
        self.assertEqual([t["draft_field"] for t in measured], [boxed["draft_field"]])
        self.assertIn("8.000U, limit 7U", measured[0]["diagnostics"][0])

    @unittest.skipUnless(shutil.which("pdflatex"), "requires the v13 TeX toolchain")
    def test_build_artifacts_errors_and_repeated_runs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Synthetic source fixture.", encoding="utf-8")
            first_draft = synthetic_draft()
            first_draft["history"] = [r"A positive real has positive square. "
                                      r"\cite[see [8, Theorem~2.1] and [6]]{source}"]
            first = profile.build_case(root / "runs", source, None, first_draft)
            state = json.loads((first / "build-status.json").read_text(encoding="utf-8"))
            self.assertTrue(state["input_binding_checked"])
            self.assertEqual(state["content_check"], "not_assessed_by_this_tool")
            self.assertEqual(state["pdf_pages"], 2)
            pdf = first / "introduction.pdf"
            self.assertEqual(len(pdfops.read_pages(pdf)), 2)
            self.assertEqual(pdfops.render_page(pdf, 1)[:8], b"\x89PNG\r\n\x1a\n")
            failed = synthetic_draft()
            failed["paragraph"] = " ".join(["long"] * 48)
            with self.assertRaisesRegex(profile.BuildFailure, "word_limit") as error:
                profile.build_case(root / "runs", source, None, failed)
            self.assertEqual([t["draft_field"] for t in error.exception.repair_targets], ["paragraph"])
            self.assertTrue(pdf.exists())
            self.assertEqual(len(list((root / "runs").glob("*/introduction.pdf"))), 1)
            case = error.exception.case_dir
            blocked = json.loads((case / "build-status.json").read_text("utf8"))
            archive = blocked["diagnostic_artifacts"]
            saved = case / archive["directory"]
            self.assertFalse(blocked["engineering_accepted"])
            self.assertFalse(blocked["pdf_available"])
            self.assertEqual(archive["pages"], 2)
            for name, expected in archive["files"].items():
                self.assertEqual(v13_check.digest(saved / name), expected)
            self.assertEqual(archive, json.loads((saved / "receipt.json").read_text("utf8")))
            self.assertNotEqual(archive["run_id"], state["diagnostic_artifacts"]["run_id"])

            # Recovery is explicitly labelled, idempotent and isolated from the recorded round.
            self.assertEqual(profile.recover_draft_case(root / "runs", source, None, failed), case)
            self.assertEqual(profile.recover_draft_case(root / "runs", source, None, first_draft), first)
            previous_ok = dict(state)
            del previous_ok["diagnostic_artifacts"]
            (first / "build-status.json").write_text(json.dumps(previous_ok), "utf8")
            self.assertEqual(profile.recover_draft_case(root / "runs", source, None, first_draft), first)
            legacy = dict(blocked)
            del legacy["diagnostic_artifacts"]
            (case / "build-status.json").write_text(json.dumps(legacy), "utf8")
            replay = profile.recover_draft_case(root / "runs", source, None, failed)
            self.assertIsNotNone(replay)
            self.assertEqual(replay, profile.recover_draft_case(root / "runs", source, None, failed))
            provenance = json.loads((replay / "draft-recovery.json").read_text("utf8"))
            self.assertEqual(provenance["original_run_id"], blocked["run_id"])
            self.assertEqual(provenance["original_tex_sha256"], v13_check.digest(replay / "introduction.tex"))
            self.assertIsNone(profile.recover_draft_case(root / "runs", source, None, {**failed, "title": "Another draft"}))

            # A new failed compiler invocation must never inherit or remove the preceding PDF.
            (case / "introduction.tex").write_text(profile.render_tex(failed).replace(
                r"\begin{document}", r"\begin{document}\unknowncompilercommand"), "utf8")
            broken = v13_check.build(profile.ASSETS, case)
            fresh = broken["diagnostic_artifacts"]
            self.assertFalse(broken["engineering_accepted"])
            self.assertEqual(fresh["pages"], 0)
            self.assertNotIn("introduction.pdf", fresh["files"])
            self.assertNotEqual(fresh["directory"], archive["directory"])
            self.assertEqual(v13_check.digest(saved / "introduction.pdf"), archive["files"]["introduction.pdf"])


if __name__ == "__main__":
    unittest.main()

