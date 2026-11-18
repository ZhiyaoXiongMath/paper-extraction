import hashlib
import io
import json
import shutil
import subprocess
from zipfile import ZipFile

import pytest

from paper_extraction import pdfops, profile
from paper_extraction.exporting import batch_archive_filename, batch_export_directory, extraction_basename, standalone_tex
from paper_extraction.store import Store
from test_profile import synthetic_draft


def test_extraction_name_preserves_source_stem_with_windows_and_unicode_paths():
    assert extraction_basename(r"D:\papers\椭圆方程 v2.pdf") == "椭圆方程 v2-extraction"
    assert extraction_basename("/papers/0705.0766v3.tex") == "0705.0766v3-extraction"
    assert extraction_basename("/papers/CON.note?.pdf") == "_CON.note_-extraction"
    long = extraction_basename("/papers/" + "😀" * 260 + ".pdf")
    assert len(long.encode("utf-8")) <= 240 and long.endswith("-extraction")
    assert batch_export_directory(1, "../结果") == "001-.._结果"


def test_batch_archive_names_keep_one_zip_extension_and_avoid_reserved_windows_names():
    assert batch_archive_filename("fixture", "卷积算子.zip") == "卷积算子.zip"
    assert batch_archive_filename("fixture", "CON .") == "_CON.zip"


def test_already_standalone_tex_never_reads_external_preamble():
    tex = b"% \\input{preamble-v13.tex}\n" b"\\documentclass{article}\n\\begin{document}Hello\\end{document}\n"

    def missing():
        raise AssertionError("standalone TeX should not read a companion")

    assert standalone_tex(tex, missing) == tex


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="requires pdfLaTeX")
def test_archive_reuses_exact_pdf_and_exports_tex_with_the_same_type_and_body(tmp_path):
    source = tmp_path / "test paper.2026.pdf"
    source.write_bytes(b"Fixed original snapshot.")
    store = Store(tmp_path / "home")
    store.create("batch", {"provider": "openai"}, [
        {"id": "paper", "name": "Display title", "kind": "pdf", "source": str(source), "root": None},
    ])
    paper = store.paper("paper")
    case = profile.build_case(store.work(paper), source, None, synthetic_draft())
    store.finish(paper, {"status": "completed", "result_dir": str(case)})
    raw_tex = (case / "introduction.tex").read_bytes()
    raw_pdf = (case / "introduction.pdf").read_bytes()
    raw_preamble = (case / "preamble-v13.tex").read_bytes()
    name = "test paper.2026-extraction"
    output = tmp_path / "extracted"
    with ZipFile(io.BytesIO(store.archive("batch", "paper"))) as archive:
        assert set(archive.namelist()) == {name + ".tex", name + ".pdf"}
        assert archive.read(name + ".pdf") == raw_pdf
        tex = archive.read(name + ".tex")
        document_start = b"\\begin{document}"
        assert tex.split(document_start, 1)[1] == raw_tex.split(document_start, 1)[1]
        assert raw_preamble.rstrip() in tex and b"\\input{preamble-v13.tex}" not in tex
        archive.extractall(output)
    with ZipFile(io.BytesIO(store.archive("batch"))) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        sha256 = manifest["papers"][0]["sha256"]
        assert set(sha256) == {name + ".tex", name + ".pdf"}
        assert all(hashlib.sha256(archive.read(f"001-Display title/{file}")).hexdigest() == digest
                   for file, digest in sha256.items())
    assert (case / "introduction.tex").read_bytes() == raw_tex
    runner = [shutil.which("pdflatex"), "-interaction=nonstopmode", "-halt-on-error",
              "-no-shell-escape", name + ".tex"]
    for _ in range(2):
        attempt = subprocess.run(runner, cwd=output, capture_output=True, timeout=90)
        assert attempt.returncode == 0, attempt.stdout.decode(errors="replace")
    assert pdfops.read_pages(output / (name + ".pdf")) == pdfops.read_pages(case / "introduction.pdf")
