"""Portable file names and standalone TeX for downloaded extractions."""
import re
from collections.abc import Callable
from pathlib import PurePosixPath


_INVALID_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
_RESERVED_FILENAME = re.compile(r"^(?:CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\.|$)", re.I)
_PREAMBLE_INPUT = re.compile(
    rb"^[ \t]*\\input[ \t]*\{preamble-v13(?:\.tex)?\}[ \t]*(?:%[^\r\n]*)?(\r?\n|\Z)", re.M
)


def _safe_component(value: str, suffix: str = "") -> str:
    value = _INVALID_FILENAME.sub("_", value)
    if _RESERVED_FILENAME.match(value + suffix):
        value = "_" + value
    # The UTF-8 byte limit also fits the UTF-16 component limit on Windows.
    limit = 240 - len(suffix)
    value = value.encode("utf-8")[:limit].decode("utf-8", errors="ignore")
    if not suffix:
        value = value.rstrip(" .")
    return (value or "paper") + suffix


def extraction_basename(source: str) -> str:
    stem = PurePosixPath(source.replace("\\", "/")).stem
    return _safe_component(stem, "-extraction")


def batch_export_directory(ordinal: int, title: str) -> str:
    return _safe_component(f"{ordinal:03d}-{title}")


def batch_archive_filename(batch_id: str, name: str | None) -> str:
    stem = (name or "").rstrip(" .")
    if stem.lower().endswith(".zip"):
        stem = stem[:-4].rstrip(" .")
    return _safe_component(stem or f"papers-{batch_id[:10]}", ".zip")


def standalone_tex(tex: bytes, load_preamble: Callable[[], bytes]) -> bytes:
    """Expand the template's local preamble input without changing any body bytes."""
    match = _PREAMBLE_INPUT.search(tex)
    if match is None:
        return tex
    newline = match[1] or b"\n"
    embedded = newline.join((
        b"% Start of document preamble",
        load_preamble().rstrip(b"\r\n"),
        b"% End of document preamble",
        b"",
    ))
    return tex[:match.start()] + embedded + tex[match.end():]
