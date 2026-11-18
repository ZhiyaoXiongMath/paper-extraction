import json
from pathlib import Path
import shutil
import unittest
from tempfile import TemporaryDirectory

from paper_extraction import pdfops, profile
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

    def test_missing_source_locator_and_unknown_key(self):
        for value in ["The historical result.", r"The historical result. \cite[p. 1]{unlisted}"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                draft = synthetic_draft()
                draft["history"] = [value]
                profile.render_tex(draft)

    @unittest.skipUnless(shutil.which("pdflatex"), "requires the v13 TeX toolchain")
    def test_build_artifacts_errors_and_repeated_runs(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.md"
            source.write_text("Synthetic source fixture.", encoding="utf-8")
            first = profile.build_case(root / "runs", source, None, synthetic_draft())
            state = json.loads((first / "build-status.json").read_text(encoding="utf-8"))
            self.assertTrue(state["input_binding_checked"])
            self.assertEqual(state["content_check"], "not_assessed_by_this_tool")
            self.assertEqual(state["pdf_pages"], 2)
            pdf = first / "introduction.pdf"
            self.assertEqual(len(pdfops.read_pages(pdf)), 2)
            self.assertEqual(pdfops.render_page(pdf, 1)[:8], b"\x89PNG\r\n\x1a\n")
            failed = synthetic_draft()
            failed["paragraph"] = " ".join(["long"] * 48)
            with self.assertRaisesRegex(ValueError, "word_limit"):
                profile.build_case(root / "runs", source, None, failed)
            self.assertTrue(pdf.exists())
            self.assertEqual(len(list((root / "runs").glob("*/introduction.pdf"))), 1)


if __name__ == "__main__":
    unittest.main()

