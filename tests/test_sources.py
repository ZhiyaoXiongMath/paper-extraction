"""Import real-world mixed folders without mixing evidence from neighboring manuscripts."""
import io
from pathlib import Path
from types import SimpleNamespace

from paper_extraction.importing import prepare_sources, stage
from paper_extraction.sources import SourceIndex


def import_files(home, contents):
    uploads = [SimpleNamespace(file=io.BytesIO(text.encode("utf-8"))) for text in contents.values()]
    return stage(home, uploads, list(contents))


def snapshot(home, imported, entries):
    selected = [SimpleNamespace(id=p["id"], entry=entry)
                for entry in entries for p in imported["papers"] if entry in p["entries"]]
    return prepare_sources(home, imported["id"], selected, "testbatch")


def test_independent_sibling_papers_preserve_own_sources_and_figures(tmp_path):
    imported = import_files(tmp_path, {
        "batch/a.tex": r"\documentclass{article}\title{A}\input{sec/a}\bibliography{refsA}",
        "batch/sec/a.tex": r"\graphicspath{{figs/}}\includegraphics[width=1cm]{chart}",
        "batch/figs/chart.pdf": "figure bytes",
        "batch/a.pdf": "printed version of A",
        "batch/b.tex": r"\documentclass{article}\title{B}\input{sec/b}\addbibresource{refsB.bib}",
        "batch/sec/b.tex": r"B's main statement",
        "batch/refsA.bib": "Bibliography for A",
        "batch/refsB.bib": "Bibliography for B",
        "batch/unrelated.bib": "Unrelated bibliography",
        "batch/c.pdf": "independent publication",
    })
    entries = {p["entry"] for p in imported["papers"]}
    assert entries == {"batch/a.tex", "batch/b.tex", "batch/c.pdf"}
    first, second = snapshot(tmp_path, imported, ["batch/a.tex", "batch/b.tex"])
    files_a = {p.relative_to(first["root"]).as_posix() for p in Path(first["root"]).rglob("*") if p.is_file()}
    assert files_a == {"a.tex", "refsA.bib", "sec/a.tex", "figs/chart.pdf"}
    index = SourceIndex(Path(second["source"]), Path(second["root"]), tmp_path / "index")
    assert {r["file"] for r in index.catalog()} == {"b.tex", "sec/b.tex", "refsB.bib"}
    assert not index.search("Bibliography for A")[0]["text"] == "Bibliography for A"


def test_choose_one_tex_package_root_snapshots_only_that_version(tmp_path):
    imported = import_files(tmp_path, {
        "volume/paper/main.tex": r"\documentclass{article}\input{chapter}\bibliography{old}",
        "volume/paper/revised.tex": r"\documentclass{article}\input{chapter}\bibliography{new}",
        "volume/paper/chapter.tex": r"\input{../shared/math}",
        "volume/shared/math.tex": r"shared mathematical definition",
        "volume/paper/old.bib": "original bibliography",
        "volume/paper/new.bib": "revised bibliography",
    })
    [option] = imported["papers"]
    assert not option["entry"] and len(option["entries"]) == 2
    [chosen] = snapshot(tmp_path, imported, ["volume/paper/revised.tex"])
    entry = Path(chosen["source"])
    assert entry.is_file()
    index = SourceIndex(entry, Path(chosen["root"]), tmp_path / "index")
    assert {r["file"] for r in index.catalog()} == {"paper/revised.tex", "paper/chapter.tex",
                                                    "shared/math.tex", "paper/new.bib"}
    assert not list(Path(chosen["root"]).rglob("old.bib"))


def test_loose_standalone_sources_share_macros_without_becoming_alternates(tmp_path):
    imported = import_files(tmp_path, {
        "corpus/a.tex": r"\documentclass{article}\input{macros}",
        "corpus/b.tex": r"\documentclass{article}\input{macros}",
        "corpus/macros.tex": r"\newcommand{\A}{A}",
    })
    assert {p["entry"] for p in imported["papers"]} == {"corpus/a.tex", "corpus/b.tex"}


def test_plain_tex_inputs_in_same_directory_remain_separate(tmp_path):
    imported = import_files(tmp_path, {"papers/one.tex": "one manuscript", "papers/two.tex": "two manuscripts"})
    assert {p["entry"] for p in imported["papers"]} == {"papers/one.tex", "papers/two.tex"}

