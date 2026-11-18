from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from paper_extraction.source_image_cache import cached_transcription


SETTINGS = dict(provider="openai", base_url="https://api.openai.com/v1", vision_model="vision-model",
                vision_reasoning_effort="medium", api_key="private-key")


def test_reuses_exact_transcription_across_restarts_and_key_rotation(tmp_path):
    calls = []

    def transcribe(png, page):
        calls.append((png, page))
        return "α = 1\r\n\r\nTheorem 2.\r\n"

    expected = "α = 1\r\n\r\nTheorem 2.\r\n"
    assert cached_transcription(tmp_path, SETTINGS, b"full page", 1, transcribe) == (expected, False)
    assert cached_transcription(tmp_path, {**SETTINGS, "api_key": "new-key"}, b"full page", 1, transcribe) == (expected, True)
    assert cached_transcription(tmp_path, SETTINGS, b"crop", 1, transcribe)[1] is False
    assert cached_transcription(tmp_path, SETTINGS, b"full page", 2, transcribe)[1] is False
    for field in ("provider", "base_url", "vision_model", "vision_reasoning_effort"):
        assert cached_transcription(tmp_path, {**SETTINGS, field: "changed"}, b"full page", 1, transcribe)[1] is False
    assert len(calls) == 7
    assert all("private-key" not in path.read_text("utf-8") for path in (tmp_path / "img").glob("*.txt"))


def test_failing_or_blank_transcriptions_are_not_cached(tmp_path):
    def fail(png, page):
        raise RuntimeError("vision service failed")

    with pytest.raises(RuntimeError, match="vision service failed"):
        cached_transcription(tmp_path, SETTINGS, b"page", 1, fail)
    for text in ("", "   ", "[]"):
        assert cached_transcription(tmp_path, SETTINGS, b"page", 1, lambda image, page: text) == (text, False)
    assert not list((tmp_path / "img").glob("*.txt"))
    assert cached_transcription(tmp_path, SETTINGS, b"page", 1, lambda image, page: "x = 1") == ("x = 1", False)


def test_simultaneous_reads_transcribe_once(tmp_path):
    ready = Barrier(2)
    calls = []

    def transcribe(png, page):
        calls.append((png, page))
        return "x = 1"

    def read():
        ready.wait(timeout=5)
        return cached_transcription(tmp_path, SETTINGS, b"page", 1, transcribe)

    with ThreadPoolExecutor(max_workers=2) as pool:
        requests = [pool.submit(read), pool.submit(read)]
        assert sorted(request.result(timeout=5) for request in requests) == [("x = 1", False), ("x = 1", True)]
    assert calls == [(b"page", 1)]
def test_two_questions_on_one_crop_keep_distinct_answers(tmp_path):
    from paper_extraction.source_image_cache import cached_transcription

    asked = []
    def read(png, page, question):
        asked.append((png, page, question))
        return "observed " + question

    work = tmp_path / "work"
    opts = {"provider": "openai", "vision_model": "test"}
    for question in ("outer exponent?", "scope of bar?", "outer exponent?"):
        text, hit = cached_transcription(work, opts, b"same crop", 10, read, question=question)
        assert text == "observed " + question
    assert hit and len(asked) == 2
