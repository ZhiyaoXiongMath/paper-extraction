from types import SimpleNamespace

import pytest

from paper_extraction import agent, request_budget
from paper_extraction.lean_workflow import DirectResult, ScreeningNote
from paper_extraction.models import structured_request_text
from paper_extraction.profile_inputs import new_batch_profile
from paper_extraction.segment_evidence import SegmentLeads


OPENAI = {"provider": "openai", "base_url": "https://api.openai.com/v1", "text_model": "gpt-6.2-sol"}


def measure(rows, config, metadata=None):
    return request_budget.whole_source_budget(rows, DirectResult, "rules", {"passages": rows}, config, metadata)


def test_verified_limits_use_exact_serialized_content_and_unknown_gateway_is_conservative(monkeypatch):
    monkeypatch.setattr(request_budget, "_local_encoding", lambda _: None)
    rows = [{"id": "t1", "text": "Ω" * 50000}]
    direct = measure(rows, OPENAI)
    sent = structured_request_text(DirectResult, "rules", {"passages": rows})
    assert direct["input_estimate"] == sum(len(s.encode("utf8")) for s in sent)
    assert direct["route"] == "direct" and direct["limits"]["max_input_tokens"] == 922000
    assert measure(rows, {**OPENAI, "base_url": "https://API.OPENAI.COM:443/v1/"})["limits"] == direct["limits"]
    assert measure(rows, {**OPENAI, "max_input_tokens": 2000000})["limits"]["max_input_tokens"] == 922000
    assert measure(rows, OPENAI, {"max_input_tokens": 20000})["reason"] == "input_limit"
    assert measure(rows, {**OPENAI, "text_model": "custom-alias"})["limit_sources"] == {"context_window": "fallback"}
    gateway = {**OPENAI, "base_url": "https://example.test/v1"}
    unknown = measure(rows, gateway, {"max_input_tokens": 922000})
    assert unknown["route"] == "extract" and unknown["limit_sources"] == {"context_window": "fallback"}
    assert measure(rows, {**gateway, "max_input_tokens": 500000})["route"] == "direct"
    assert measure(rows, {**OPENAI, "provider": "zhipu"})["reason"] == "context_limit"
    assert measure(rows, {**OPENAI, "context_window": 65536})["reason"] == "context_limit"


def test_token_count_and_separate_input_output_and_context_limits(monkeypatch):
    encoding = SimpleNamespace(name="known_fixture", encode_ordinary=lambda text: text.split())
    monkeypatch.setattr(request_budget, "_local_encoding", lambda _: encoding)
    rows = [{"id": "t", "text": "A definition and a theorem."}]
    result = measure(rows, OPENAI)
    assert result["count_method"] == encoding.name
    n = result["input_estimate"] + result["framing_reserve"]
    # A pure input limit already excludes output. The total window includes it.
    assert measure(rows, {**OPENAI, "max_input_tokens": n})["route"] == "direct"
    assert measure(rows, {**OPENAI, "max_input_tokens": n - 1})["reason"] == "input_limit"
    assert measure(rows, {**OPENAI, "context_window": n + result["output_reserve"] - 1})["reason"] == "context_limit"
    assert measure(rows, {**OPENAI, "max_output_tokens": 4096})["reason"] == "output_limit"
    assert measure([{"id": "p", "page": 1, "text": "(cid:73)"}], OPENAI)["reason"] == "damaged_text"
    assert measure([{"id": "p", "text": "legible", "transcribed_from_image": True}], OPENAI)["reason"] == "image_transcription"
    assert measure([{"id": "p", "text": " "}], OPENAI)["reason"] == "empty_span"


def test_grounded_request_preserves_verbatim_math_and_matches_budget(monkeypatch):
    import json
    from paper_extraction.profile_inputs import (
        TARGETED_WORKFLOW, new_batch_profile, uses_grounded_revision, uses_targeted_revision,
    )

    monkeypatch.setattr(request_budget, "_local_encoding", lambda _: None)
    latest = {**OPENAI, **new_batch_profile()}
    prior = {**latest, "workflow_revision": TARGETED_WORKFLOW}
    assert uses_grounded_revision(latest) and not uses_grounded_revision(prior)
    assert uses_targeted_revision(prior)
    text = "Données  et hypothèses\n" + r"\forall x\in K,\quad \Phi(x)\leq 0."
    rows = [{"id": "section-7", "text": text}]
    packet = {"passages": rows}
    old = structured_request_text(DirectResult, "rules", packet)
    new = structured_request_text(DirectResult, "rules", packet, compact=True)
    assert json.loads(old[1]) == json.loads(new[1]) == packet
    assert json.loads(old[0].split("\n")[-1]) == json.loads(new[0].split("\n")[-1])
    assert measure(rows, latest)["input_estimate"] == sum(len(s.encode("utf8")) for s in new)
    assert measure(rows, prior)["input_estimate"] == sum(len(s.encode("utf8")) for s in old)
    assert len(new[0]) + len(new[1]) < len(old[0]) + len(old[1])


class ProviderError(Exception):
    def __init__(self, status, code):
        super().__init__(code)
        self.status_code = status
        self.body = {"error": {"code": code}}


@pytest.mark.parametrize(("status", "code", "fallback"), [
    (400, "context_length_exceeded", True),
    (401, "invalid_api_key", False),
    (429, "insufficient_quota", False),
    (400, "content_filter", False),
])
def test_provider_limit_falls_back_once_without_losing_sources(tmp_path, monkeypatch, status, code, fallback):
    source = tmp_path / "original" / "note.tex"
    source.parent.mkdir()
    source.write_text("An open question. Could every example have property P?")
    calls, original, events = [], [], []

    class Responses:
        def structured(self, schema, instructions, payload):
            calls.append(schema)
            if schema is DirectResult:
                original.extend(r["id"] for r in payload["passages"])
                raise ProviderError(status, code)
            assert schema is SegmentLeads and original == [r["id"] for r in payload["passages"]]
            return SegmentLeads(candidates=[])

    def no_result(*args, **kwargs):
        return [], ScreeningNote(kind="no_qualifying_result", question="开放问题。",
                                 reason="只有问题，未陈述结论。", evidence_ids=original)

    monkeypatch.setattr(agent, "Models", lambda _: Responses())
    monkeypatch.setattr(agent, "select_contributions", no_result)
    result = agent.run_paper(tmp_path / "work", source, source.parent, new_batch_profile(),
                             lambda name, info: events.append(info))
    assert calls == ([DirectResult, SegmentLeads] if fallback else [DirectResult]), result.get("reason")
    assert (result["status"] == "completed") is fallback
    reasons = [e["request_budget"]["reason"] for e in events if "request_budget" in e]
    assert reasons == (["fits", "provider_context_limit"] if fallback else ["fits"])
