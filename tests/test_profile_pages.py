"""Exercise bibliography geometry, including content nested in a PDF form."""
import shutil
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from paper_extraction.profile_assets.tools import v13_check


@unittest.skipUnless(shutil.which("pdflatex"), "requires pdfLaTeX")
class PageGeometryTests(unittest.TestCase):
    def test_graphics_text_and_nested_graphics_on_reference_page(self):
        heading = r"\noindent References\par"
        aux = r"\newlabel{intro:references-start}{{}{2}}\newlabel{intro:references-end}{{}{2}}"
        bib = r"\begin{thebibliography}{9}\bibitem{a} A.\end{thebibliography}\end{document}"
        nested = (r"\setbox0=\hbox{\includegraphics[width=2cm]{mark.png}}\pdfxform0 "
                  r"\setbox0=\hbox{\pdfrefxform\pdflastxform}\pdfxform0 "
                  r"\noindent\pdfrefxform\pdflastxform\par")
        graphic = r"\noindent\rule{4cm}{0.3cm}\par"
        image = r"\noindent\includegraphics[width=2cm]{mark.png}\par"
        mixed_text = r"\noindent Part of the body.\par"
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (20, 10), "blue").save(root / "mark.png")
            for before, after, reason in [
                ("", graphic, "separate"), ("", nested, "separate"),
                (mixed_text, "", "text"), (graphic, "", "graphics"),
                (image, "", "graphics"), (nested, "", "graphics"),
            ]:
                with self.subTest(before=before, after=after):
                    tex = (r"\documentclass[a4paper]{article}\usepackage{graphicx}\begin{document}"
                           r"Body.\newpage " + before + heading + after + r"\end{document}")
                    (root / "case.tex").write_text(tex, encoding="utf-8")
                    subprocess.run(
                        ["pdflatex", "-halt-on-error", "-no-shell-escape", "-interaction=batchmode", "case.tex"],
                        cwd=root, check=True, capture_output=True, timeout=45)
                    report = v13_check.inspect_pdf(root / "case.pdf", aux, bib)
                    self.assertTrue(report["bibliography_located"])
                    self.assertEqual(report["references_separate"], reason == "separate", report)
                    self.assertEqual(report["main_text_pages"], 1 if reason == "separate" else 2)
                    if reason == "text":
                        self.assertIn("Part of the body.", report["body_before_references"])
                    if reason == "graphics":
                        self.assertTrue(report["graphics_before_references"])

