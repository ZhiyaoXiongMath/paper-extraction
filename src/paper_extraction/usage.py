"""Request-level usage, local source coordinates and measured input sizes."""
import json
import logging
import re
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


_session = ContextVar("paper_usage_session", default=None)
_stage = ContextVar("paper_usage_stage", default=("unknown", 0))
_wire = ContextVar("paper_usage_http", default=None)
_lock = Lock()
_FILE = "model-usage.jsonl"
_LOG = logging.getLogger(__name__)


@contextmanager
def usage_session(work, paper_id, attempt, settings):
    # Use an allowlist so neither credentials nor paper text can enter the ledger.
    config = {key: settings.get(key) for key in (
        "provider", "text_model", "vision_model", "text_reasoning_effort", "vision_reasoning_effort"
    )}
    token = _session.set((Path(work) / _FILE, paper_id, attempt, config))
    try:
        yield
    finally:
        _session.reset(token)


@contextmanager
def usage_stage(stage, round_no=None):
    token = _stage.set((stage, _stage.get()[1] if round_no is None else round_no))
    try:
        yield
    finally:
        _stage.reset(token)


def _count_http(request):
    counter = _wire.get()
    if counter is not None:
        counter["requests"] += 1


def observe_http(model):
    """Count actual sync HTTP sends, including SDK retries, on the existing provider transport."""
    if _session.get() is None:
        return
    root = getattr(model, "root_client", None)
    client = getattr(root, "_client", None)
    hooks = getattr(client, "event_hooks", {}).get("request")
    if hooks is not None:
        with _lock:
            if _count_http not in hooks:
                hooks.append(_count_http)


def _record(operation, channel, **fields):
    binding = _session.get()
    if binding is None:
        return
    path, paper_id, attempt, settings = binding
    stage, round_no = _stage.get()
    row = dict(time=datetime.now(timezone.utc).isoformat(), paper_id=paper_id, attempt=attempt,
               draft_round=round_no, stage=stage, operation=operation, channel=channel,
               provider=settings["provider"], model=settings[channel + "_model"],
               reasoning_effort=settings[channel + "_reasoning_effort"], **fields)
    try:
        with _lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    except OSError as exc:
        _LOG.warning("Could not save model usage (%s)", type(exc).__name__)


def record_cache_hit(operation, channel):
    _record(operation, channel, status="cache_hit", http_requests=0)


def _number(mapping, *keys):
    for key in keys:
        value = mapping.get(key)
        if type(value) is int and value >= 0:
            return value
    return None


def _tokens(usage):
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump(exclude_none=True)
    usage = usage if isinstance(usage, dict) else {}
    inputs = usage.get("input_token_details") or usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    outputs = usage.get("output_token_details") or usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    cache_read = _number(inputs, "cache_read", "cached_tokens", "priority_cache_read", "flex_cache_read")
    return dict(input_tokens=_number(usage, "input_tokens", "prompt_tokens"),
                output_tokens=_number(usage, "output_tokens", "completion_tokens"),
                cache_read_tokens=_number(usage, "prompt_cache_hit_tokens") if cache_read is None else cache_read,
                reasoning_tokens=_number(outputs, "reasoning", "reasoning_tokens", "priority_reasoning", "flex_reasoning"))


def invoke_tracked(runner, messages, *, operation, channel, schema_attempt=1, passages=(), previous_schema_failure=None):
    if _session.get() is None:
        return runner.invoke(messages)
    # The SDK may perform several HTTP requests for a single model invocation.
    # LangChain propagates a copy of the current context to its synchronous runners.
    counter = Counter()
    token = _wire.set(counter)
    usage = None
    status = "error"
    try:
        result = runner.invoke(messages)
        raw = result.get("raw") if isinstance(result, dict) else result
        metadata = getattr(raw, "response_metadata", {}) or {}
        # Some adapters fill absent quantities with 0 when normalizing metadata.
        # Prefer the original response values when available.
        usage = (metadata.get("token_usage") or metadata.get("usage")
                 or getattr(raw, "usage_metadata", None))
        status = "response"
        return result
    except Exception as exc:
        # Some OpenAI SDK methods raise a truncation error carrying a billable completion.
        usage = getattr(getattr(exc, "completion", None), "usage", None)
        raise
    finally:
        _wire.reset(token)
        # Source coordinates let a replay measure repeated context without
        # copying any manuscript text, filename or model response into the ledger.
        locations = [row for row in (passages or ()) if isinstance(row, dict)
                     and re.fullmatch(r"[pt][0-9]+-[0-9]+", str(row.get("id", "")))
                     and isinstance(row.get("text"), str)]
        shape = {}
        if locations:
            shape = {"source_ids": [row["id"] for row in locations],
                     "source_text_bytes": sum(len(row["text"].encode("utf-8")) for row in locations)}
        if previous_schema_failure:
            shape["previous_schema_failure"] = previous_schema_failure
        _record(operation, channel, status=status, schema_attempt=schema_attempt,
                http_requests=counter.get("requests"), **shape, **_tokens(usage))


def usage_summary(paper_root):
    """Summarize every business attempt, including retries sharing a checkpoint directory."""
    totals = Counter()
    groups = {}
    for path in sorted(Path(paper_root).glob("attempt-*/" + _FILE)):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # A worker can still be appending the final line.
                key = tuple(row.get(k) for k in ("attempt", "draft_round", "stage", "operation",
                                                "channel", "provider", "model", "reasoning_effort"))
                counts = Counter()
                if row["status"] == "cache_hit":
                    counts["cache_hits"] = 1
                else:
                    counts["model_calls"] = 1
                    counts["error_calls"] = int(row["status"] == "error")
                    counts["schema_retry_calls"] = int(row.get("schema_attempt", 1) > 1)
                    for name in ("http_requests", "input_tokens", "output_tokens", "cache_read_tokens", "reasoning_tokens"):
                        if row.get(name) is None:
                            counts[name + "_unknown_calls"] = 1
                        else:
                            counts[name] = row[name]
                groups.setdefault(key, Counter()).update(counts)
                totals.update(counts)
    labels = ("attempt", "draft_round", "stage", "operation", "channel", "provider", "model", "reasoning_effort")
    return {"recorded": bool(groups), "totals": dict(totals),
            "groups": [dict(zip(labels, key), **counts) for key, counts in groups.items()]}
