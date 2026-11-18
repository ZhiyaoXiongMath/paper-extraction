"""Typed contracts and adapters for the workbench's model providers."""
from __future__ import annotations

import base64
import json
import re
from typing import Literal, TypeVar
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import LengthFinishReasonError
from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, model_validator

from .editorial import says_no_edit
from .settings import DEFAULT_REASONING_EFFORT, chat_options

Model = TypeVar("Model", bound=BaseModel)


class Candidate(BaseModel):
    kind: Literal["historical", "current", "definition", "application", "setting"]
    claim: str = Field(min_length=1)
    assumptions: str
    significance: str
    source_number: str | None = Field(description="Original theorem/definition label, null if unnumbered")
    dependencies: list[str] = Field(description="Definition names and meanings for this entry")
    decision: Literal["admitted", "excluded", "unresolved"]
    decision_reason: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class Findings(BaseModel):
    candidates: list[Candidate]
    uncertainties: list[str]


class Group(BaseModel):
    definitions: list[str] = Field(max_length=2)
    theorem: str = Field(min_length=1)
    paragraph: str = ""
    paragraph_position: Literal["before", "after"] = Field(default="after", description="Place the connecting paragraph before the group's definitions and theorem, or after the theorem")


class Reference(BaseModel):
    key: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
    text: str = Field(min_length=1, description="Bibliographic entry content only; the renderer supplies the TeX item and environment")


class EvidenceLink(BaseModel):
    block_id: str
    evidence_ids: list[str] = Field(min_length=1)


class BlockReplacement(EvidenceLink):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, description="Complete revised text of this existing block, including its citations")
    paragraph_position: Literal["before", "after"] | None = Field(default=None, description="Optional new location for a group paragraph")


class DraftPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    replacements: list[BlockReplacement] = Field(min_length=1)

    @model_validator(mode="after")
    def match_requested_edits(self, info: ValidationInfo):
        ids = [row.block_id for row in self.replacements]
        if len(ids) != len(set(ids)):
            raise ValueError("Return one complete replacement per target, combining all its corrections")
        if info.context is None:
            return self
        targets = {row["block_id"]: row for row in info.context["repair_targets"]}
        if set(ids) != set(targets):
            raise ValueError(f"Replace exactly these existing blocks: {list(targets)}")
        registered = {row["id"] for row in info.context["passages"]}
        for row in self.replacements:
            target = targets[row.block_id]
            revised = " ".join(row.text.split())
            if row.paragraph_position is not None and "paragraph_position" not in target:
                raise ValueError("Only a result group's connecting paragraph can change position")
            moved = row.paragraph_position is not None and row.paragraph_position != target.get("paragraph_position")
            if not revised or (not moved and revised == " ".join(target["current_text"].split())):
                raise ValueError(f"{row.block_id}: make the requested edit to the existing text")
            locations = set(row.evidence_ids)
            if not locations <= registered or not set(target["required_evidence_ids"]) <= locations:
                raise ValueError(f"{row.block_id}: cite only supplied original passages and include "
                                 f"all required locations {target['required_evidence_ids']}")
            if len(locations) != len(row.evidence_ids):
                raise ValueError(f"{row.block_id}: list each source ID once")
        return self


class Draft(BaseModel):
    title: str = Field(min_length=1)
    authors: list[str] = Field(min_length=1)
    paragraph: str = Field(min_length=1)
    history: list[str] = Field(min_length=1, max_length=3)
    shared_definitions: list[str] = Field(max_length=2)
    groups: list[Group] = Field(min_length=1)
    references: list[Reference] = Field(min_length=1)
    evidence_links: list[EvidenceLink]

    @model_validator(mode="after")
    def check_shape(self):
        blocks = {"paragraph"}
        blocks.update(f"history.{i}" for i in range(len(self.history)))
        blocks.update(f"shared_definitions.{i}" for i in range(len(self.shared_definitions)))
        for i, group in enumerate(self.groups):
            blocks.add(f"groups.{i}.theorem")
            if group.paragraph:
                blocks.add(f"groups.{i}.paragraph")
            blocks.update(f"groups.{i}.definitions.{j}" for j in range(len(group.definitions)))
        linked = [link.block_id for link in self.evidence_links]
        if set(linked) != blocks or len(linked) != len(blocks):
            raise ValueError(f"Provide one evidence link for each block: {sorted(blocks)}")
        if len(self.shared_definitions) + sum(len(g.definitions) for g in self.groups) > 6:
            raise ValueError("The Introduction has more than six definitions")
        keys = [ref.key for ref in self.references]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate bibliography keys")
        return self


class VisualLocation(BaseModel):
    draft_field: str = Field(min_length=1)
    issue: str = Field(min_length=1)


class Review(BaseModel):
    passed: bool
    issues: list[str]
    locations: list[VisualLocation] = Field(default_factory=list)
    @model_validator(mode="after")
    def require_consistency(self):
        if self.passed == bool(self.issues):
            raise ValueError("A passing review has no issues. A failing review identifies the defects.")
        if any(says_no_edit(issue) for issue in self.issues):
            raise ValueError("A no-issue entry is inconsistent with passed=false")
        return self


class EditorialDefect(BaseModel):
    kind: Literal["incorrect_claim", "missing_required_content", "incorrect_attribution_or_locator", "improper_selection"]
    draft_field: str = Field(min_length=1, description="Affected Draft JSON path, e.g. groups.0.theorem or history for a missing block")
    current_excerpt: str = Field(description="Corresponding text in draft_field, or an empty string for absent text")
    evidence_id: str = Field(min_length=1, description="Registered original source location")
    source_excerpt: str = Field(min_length=1, description="Original passage at evidence_id")
    required_change: str = Field(min_length=1, description="Requested edit and applicable v13 clause")
    operation: Literal["replace", "insert", "remove", "recover"] = Field(
        description="Edit operation. recover requests the extract route.")
    replacement_text: str = Field(default="", description=(
        "Exact revised wording for replace, or exact new wording for insert. Empty for remove or recover."))

    @model_validator(mode="after")
    def require_current_excerpt(self):
        if self.kind != "missing_required_content" and not self.current_excerpt.strip():
            raise ValueError("Quote the affected current draft passage for this defect")
        if says_no_edit(self.required_change):
            raise ValueError("A required_change saying no edit is needed is inconsistent with an edit operation")
        new = " ".join(self.replacement_text.split())
        if self.operation in {"insert", "replace"}:
            if not new or new.casefold() in {"unchanged", "none", "n/a", "no change"}:
                raise ValueError("Supply the actual new wording for this edit")
            if new == " ".join(self.current_excerpt.split()):
                raise ValueError("The proposed wording is identical to the existing passage")
        elif new:
            raise ValueError("remove or recover must leave replacement_text empty")
        if self.operation in {"remove", "replace"} and not self.current_excerpt.strip():
            raise ValueError("Quote the existing text to remove or replace")
        if self.operation == "recover" and self.kind != "missing_required_content":
            raise ValueError("recover requires kind=missing_required_content")
        return self


class AdmissionReview(BaseModel):
    route: Literal["pass", "extract", "draft"]
    issues: list[EditorialDefect] = Field(description="Edit records; empty on the pass route")
    evidence_ids: list[str] = Field(description="Original locations to reread on the extract route; otherwise use []")

    def repair_issues(self) -> list[str]:
        return list(dict.fromkeys(
            f'{issue.draft_field}: {issue.required_change} '
            + (f'[{issue.operation}: {issue.replacement_text}] ' if issue.replacement_text else "")
            + f'(draft: {issue.current_excerpt or "[missing]"}; original {issue.evidence_id}: {issue.source_excerpt})'
            for issue in self.issues
        ))

    @model_validator(mode="after")
    def validate_route(self):
        if (self.route == "pass") == bool(self.issues):
            raise ValueError("A pass has no issues; a repair identifies a source-specific issue")
        if self.route == "extract" and not self.evidence_ids:
            raise ValueError("Target missing claims by their original source IDs")
        if any(item.operation == "recover" for item in self.issues) and self.route != "extract":
            raise ValueError("Route evidence recovery to extract")
        return self


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def png_message(request: str, image: bytes) -> HumanMessage:
    url = "data:image/png;base64," + base64.b64encode(image).decode("ascii")
    return HumanMessage(content=[
        {"type": "text", "text": request},
        {"type": "image_url", "image_url": {"url": url}},
    ])


def _correctable_candidates(document, errors):
    """Only repair invalid rows; keep every other original candidate and its evidence."""
    if not isinstance(document, dict) or not isinstance(document.get("candidates"), list):
        return []
    indexes = set()
    for error in errors:
        loc = error["loc"]
        if len(loc) < 2 or loc[0] != "candidates" or not isinstance(loc[1], int):
            return []
        indexes.add(loc[1])
    return sorted(indexes)


def _field_errors(exc, *, explain=True):
    # Pydantic's ordinary repr includes input values. Only expose field names and
    # schema expectations, never paper passages or a raw provider response.
    errors = exc.errors(include_url=False, include_input=False, include_context=False)
    details = []
    for error in errors[:16]:
        location = ".".join(map(str, error["loc"])) or "document"
        message = f"{location}: {error['type']}"
        details.append(f"{message} ({error['msg']})" if explain else message)
    return errors, "; ".join(details)



class ReasoningChatOpenAI(ChatOpenAI):
    """Keep provider reasoning only for replay in tool conversations."""

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        raw = response if isinstance(response, dict) else response.model_dump(warnings=False)
        for generation, choice in zip(result.generations, raw["choices"]):
            reasoning = choice["message"].get("reasoning_content")
            if isinstance(reasoning, str):
                generation.message.additional_kwargs["reasoning_content"] = reasoning
        return result

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if payload.get("tools") and "messages" in payload:
            for message, wire in zip(self._convert_input(input_).to_messages(), payload["messages"]):
                if wire["role"] == "assistant" and "reasoning_content" in message.additional_kwargs:
                    wire["reasoning_content"] = message.additional_kwargs["reasoning_content"]
        return payload


class Models:
    def __init__(self, settings: dict):
        if not settings.get("api_key"):
            raise ValueError("The selected provider requires an API key in the local configuration")
        timeout = settings.get("timeout", 120)
        self.provider = settings["provider"].lower()
        text_options = chat_options(settings, "text", timeout=timeout, max_retries=2)
        text_client = ReasoningChatOpenAI if "extra_body" in text_options else ChatOpenAI
        self.text = text_client(**text_options)
        self.vision = ChatOpenAI(**chat_options(settings, "vision", timeout=timeout, max_retries=2))
        self.tool_model = self.text
        if (self.provider == "openai" and urlparse(settings["base_url"]).hostname == "api.openai.com"
                and re.match(r"^gpt-6\.2-(?:sol|astra)(?:-|$)", settings["text_model"])):
            # Official GPT-6.2 requires Responses for function calls. Replay encrypted reasoning
            # with the tool output instead of requiring server-side conversation state.
            self.tool_model = self.text.model_copy(update={
                "use_responses_api": True, "store": False, "include": ["reasoning.encrypted_content"],
                "reasoning_effort": settings.get("text_reasoning_effort", DEFAULT_REASONING_EFFORT),
            })

    def structured(self, schema: type[Model], instructions: str, payload,
                   image: bytes | None = None) -> Model:
        model = self.vision if image is not None else self.text
        contract = schema.model_json_schema()
        system = SystemMessage(content=instructions + "\nReturn one JSON object matching the supplied "
                               "schema; escape TeX backslashes in JSON strings.\n" + json_text(contract))
        original = png_message(json_text(payload), image) if image is not None else HumanMessage(content=json_text(payload))
        next_messages = [system, original]
        original_findings, repair_indexes = None, []
        keep_previous = schema in (Findings, Draft, DraftPatch, AdmissionReview, Review)
        history = []
        for attempt in range(3):
            runner = model.with_structured_output(schema, method="json_mode", include_raw=True)
            try:
                response = runner.invoke(next_messages)["raw"]
                finish = response.response_metadata.get("finish_reason") or "unknown"
                out = (response.usage_metadata or {}).get("output_tokens", 0)
                text = response.content if isinstance(response.content, str) else ""
            except LengthFinishReasonError as exc:
                # The SDK's parse endpoint can raise before LangChain returns the raw
                # completion, even when the caller explicitly requests include_raw=True.
                finish, text = "length", ""
                out = getattr(exc.completion.usage, "completion_tokens", 0) or 0

            details = ""
            next_messages = [system, original]
            if finish == "length":
                # ChatOpenAI aliases max_tokens to max_completion_tokens. These providers
                # accept max_tokens in the JSON wire body, so supply the provider field.
                budget = min(65536, max(16384, 2 * out))
                if self.provider == "openai":
                    model = model.model_copy(update={"max_tokens": budget})
                else:
                    model = model.model_copy(update={"extra_body": {**(model.extra_body or {}),
                                                                   "max_tokens": budget}})
                details = "Truncated output. Regenerate a complete JSON object from the original evidence."
            elif not text.strip():
                details = "Blank JSON output. Produce a complete JSON object using the schema."
            else:
                try:
                    document = json.loads(text)
                except json.JSONDecodeError as exc:
                    details = f"Malformed JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}."
                else:
                    if repair_indexes:
                        corrections = document.get("candidates") if isinstance(document, dict) else None
                        if not isinstance(corrections, list) or len(corrections) != len(repair_indexes):
                            details = f"Return exactly the {len(repair_indexes)} requested corrected rows in order."
                        else:
                            merged = list(original_findings["candidates"])
                            for idx, corrected in zip(repair_indexes, corrections):
                                merged[idx] = corrected
                            document = {**original_findings, "candidates": merged}
                    if not details:
                        try:
                            return schema.model_validate(document, context=payload if schema is DraftPatch else None)
                        except ValidationError as exc:
                            errors, details = _field_errors(exc, explain=keep_previous)
                            if schema is Findings:
                                indexes = _correctable_candidates(document, errors)
                                if indexes:
                                    original_findings, repair_indexes = document, indexes
                    if not repair_indexes and details and keep_previous:
                        next_messages.append(HumanMessage(content=json_text({
                            "previous_complete_json": document, "correction": details,
                            "return": "Correct the reported fields and return JSON matching the schema in the system message."
                        })))
            history.append(f"{attempt+1}: finish={finish}, output_tokens={out}, {details}")
            if repair_indexes:
                next_messages.append(HumanMessage(content=json_text({
                    "correction": details,
                    "rows_to_fix_in_order": [{"original_index": i, "value": original_findings["candidates"][i]}
                                             for i in repair_indexes],
                    "return": "Use the Findings schema. Return only these corrected rows in their original "
                              "order. Preserve valid field values. Use [] for uncertainties."
                })))
            elif len(next_messages) == 2:
                next_messages.append(HumanMessage(content=details + " Return JSON matching the schema in the system message."))
        raise ValueError(f"Invalid {schema.__name__} response after three attempts: " + " | ".join(history))


    def transcribe(self, image: bytes, page: int) -> str:
        response = self.vision.invoke([SystemMessage(content=(
            "Read the mathematical source page. Transcribe visible definitions, statements, exact "
            "symbols, qualifiers and source labels. Flag any unreadable formula; do not infer it."
        )), png_message(f"Inspect page {page}.", image)])
        return str(response.content)

