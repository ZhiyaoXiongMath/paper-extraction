"""Decide whether the complete structured request can be sent as a single read."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit

from .models import structured_request_text


# OpenAI API model card, verified 2026-11-19. Only valid for the exact
# provider/model pair, not a gateway which happens to use the same alias.
# https://developers.openai.com/api/docs/models/gpt-6.2-sol
_OPENAI_LIMITS = {
    "gpt-6.2-sol": {"context_window": 1_050_000, "max_input_tokens": 922_000, "max_output_tokens": 128_000},
}
_FRAMING_RESERVE = 1024
_OUTPUT_RESERVE = 8192
_DEFAULT_WINDOW = 65536

# tiktoken verifies the same hashes when reading its cache. Only open an
# encoding if its vocabulary is already available; offline imports and routing
# must never implicitly fetch the vocabulary from the network.
_VOCAB_HASHES = {
    "o200k_base": "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
    "cl100k_base": "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
}


@lru_cache(maxsize=32)
def _local_encoding(model):
    try:
        import tiktoken
        from tiktoken.registry import ENCODINGS
        name = tiktoken.encoding_name_for_model(model)
        if name in ENCODINGS:
            return ENCODINGS[name]
        if name not in _VOCAB_HASHES:
            return None
        cache = os.environ.get("TIKTOKEN_CACHE_DIR", os.environ.get(
            "DATA_GYM_CACHE_DIR", os.path.join(tempfile.gettempdir(), "data-gym-cache")))
        if not cache:
            return None
        url = f"https://openaipublic.blob.core.windows.net/encodings/{name}.tiktoken"
        entry = Path(cache) / hashlib.sha1(url.encode(), usedforsecurity=False).hexdigest()
        if hashlib.sha256(entry.read_bytes()).hexdigest() == _VOCAB_HASHES[name]:
            return tiktoken.get_encoding(name)
    except (ImportError, KeyError, OSError, ValueError):
        pass
    return None


def whole_source_budget(catalog, schema, instructions, packet, settings, model_profile=None):
    """Count the same message content the adapter will send; keep every source ID."""
    model = settings.get("text_model", "")
    endpoint = urlsplit(settings.get("base_url") or "")
    official_openai = (settings.get("provider") == "openai" and endpoint.scheme == "https" and
                       endpoint.netloc.lower() in {"api.openai.com", "api.openai.com:443"} and
                       endpoint.path.rstrip("/") in {"", "/v1"} and not endpoint.query and not endpoint.fragment)
    known = _OPENAI_LIMITS.get(model, {}) if official_openai else {}
    # ChatOpenAI may infer its profile from a model name even on a custom host.
    # An OpenAI-compatible proxy's alias cannot establish its actual limits.
    adapter = (model_profile or {}) if official_openai else {}
    limits, origins = {}, {}
    for key in ("context_window", "max_input_tokens", "max_output_tokens"):
        for source, values in (("batch", settings), ("adapter", adapter), ("model_card", known)):
            value = values.get(key)
            # A deployment may set a smaller limit. An accidental larger
            # override cannot enlarge a provider's verified window.
            if type(value) is int and value > 0 and (key not in limits or value < limits[key]):
                limits[key], origins[key] = value, source
    capacity_known = "max_input_tokens" in limits or "context_window" in limits
    if not capacity_known:
        limits["context_window"], origins["context_window"] = _DEFAULT_WINDOW, "fallback"
    system, user = structured_request_text(schema, instructions, packet)
    raw_size = len(system.encode("utf-8")) + len(user.encode("utf-8"))
    encoding = _local_encoding(model) if official_openai and capacity_known else None
    count = (len(encoding.encode_ordinary(system)) + len(encoding.encode_ordinary(user))
             if encoding else raw_size)
    input_required = count + _FRAMING_RESERVE
    reason = "fits"
    if any(not str(row.get("text") or "").strip() for row in catalog):
        reason = "empty_span"
    elif any(row.get("transcribed_from_image") for row in catalog):
        reason = "image_transcription"
    elif any(row.get("page") and re.search(r"\ufffd|\(cid:\d+\)", row["text"]) for row in catalog):
        reason = "damaged_text"
    elif "max_input_tokens" in limits and input_required > limits["max_input_tokens"]:
        reason = "input_limit"
    elif "context_window" in limits and input_required + _OUTPUT_RESERVE > limits["context_window"]:
        reason = "context_limit"
    elif "max_output_tokens" in limits and _OUTPUT_RESERVE > limits["max_output_tokens"]:
        reason = "output_limit"
    return {"route": "direct" if reason == "fits" else "extract", "reason": reason,
            "input_estimate": count, "input_utf8_bytes": raw_size,
            "count_method": encoding.name if encoding else "utf8_byte_bound",
            "framing_reserve": _FRAMING_RESERVE, "output_reserve": _OUTPUT_RESERVE,
            "limits": limits, "limit_sources": origins}


def context_limit_error(exc):
    """Only switch routes for an actual provider rejection of this input length."""
    if getattr(exc, "status_code", None) not in {400, 413, 422}:
        return False
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        body = body.get("error", body)
    code = getattr(exc, "code", None) or (body.get("code") if isinstance(body, dict) else None)
    if code in ("context_length_exceeded", "context_window_exceeded", "input_token_limit_exceeded"):
        return True
    if code not in (None, "", "invalid_request_error"):
        return False
    message = body.get("message", "") if isinstance(body, dict) else str(exc)
    return isinstance(message, str) and bool(re.search(
        r"maximum context length is\b|exceeds? (?:the |this model.s )?(?:maximum )?context (?:window|length)", message, re.I))
