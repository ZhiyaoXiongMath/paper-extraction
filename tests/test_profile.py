import json
from pathlib import Path
import shutil
import unittest
from tempfile import TemporaryDirectory

from paper_extraction import pdfops, profile
from paper_extraction.sources import SourceIndex
from paper_extraction.profile_assets.tools import v13_check


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


if __name__ == "__main__":
    unittest.main()

