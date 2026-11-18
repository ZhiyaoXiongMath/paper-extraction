"""Reuse transcriptions of identical source images within one paper's work directory."""
import hashlib
import json
import logging
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

from filelock import FileLock

from .settings import thinking_enabled
from .usage import record_cache_hit


_LOG = logging.getLogger(__name__)
_VISION_SETTINGS = ("provider", "base_url", "vision_model", "vision_reasoning_effort")


def _store_text(path: Path, value: str) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="tmp-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(value.encode("utf-8"))
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def cached_transcription(work: Path, settings: Mapping[str, object], png: bytes, page: int,
                         transcribe: Callable[..., str], *, question: str | None = None) -> tuple[str, bool]:
    """Return a transcription and whether it was reused, storing only successful nonblank text."""
    # The page number is part of the vision prompt. Hash the public settings with
    # it and the exact rendered bytes, so changing crop, model or effort misses.
    request = ["source-transcription-v1", page, *(settings.get(name) for name in _VISION_SETTINGS)]
    if question is not None:
        request.extend(("targeted-reading-v1", question))
    if thinking_enabled(str(settings.get("provider", "")), str(settings.get("vision_model", ""))):
        request.append("thinking-enabled")
    fingerprint = hashlib.sha256(json.dumps(request, ensure_ascii=False).encode("utf-8"))
    fingerprint.update(png)
    directory = Path(work) / "img"
    path = directory / f"{fingerprint.hexdigest()}.txt"
    lock = FileLock(path.with_suffix(".lock"))
    args = (png, page, question) if question is not None else (png, page)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        lock.acquire()
    except OSError as exc:
        _LOG.debug("Source image cache unavailable: %s", type(exc).__name__)
        return transcribe(*args), False

    try:
        try:
            cached = path.read_bytes().decode("utf-8")
        except (OSError, UnicodeError):
            cached = ""
        if cached.strip():
            record_cache_hit("source_region_reading" if question else "source_transcription", "vision")
            return cached, True

        result = transcribe(*args)
        if result.strip() and result.strip() != "[]":
            try:
                _store_text(path, result)
            except (OSError, UnicodeError) as exc:
                _LOG.debug("Source image cache write failed: %s", type(exc).__name__)
        return result, False
    finally:
        lock.release()
