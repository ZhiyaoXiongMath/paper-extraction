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
from .citation_types import FormalSource
from .direct_response import (EvidenceCoverageError, direct_retry_packet, direct_source_action, normalize_direct_shape,
                              prepare_evidence_patch, provisional_direct_answer, safe_direct_errors, unfence,
                              with_evidence_links)
from .profile_inputs import (is_optimized_workflow, is_reliable_workflow, uses_block_workflow,
                             uses_evidence_link_patch, uses_grounded_revision, uses_scope_revision)
from .settings import DEFAULT_REASONING_EFFORT, chat_options
from .usage import invoke_tracked, observe_http

Model = TypeVar("Model", bound=BaseModel)


class StructuredResponseError(ValueError):
    """The provider replied, but no attempt supplied a usable typed response."""
    def __init__(self, schema, history):
        self.schema = schema
        attempts = {1: "one attempt", 2: "two attempts", 3: "three attempts"}[len(history)]
        super().__init__(f"Invalid {schema.__name__} response after {attempts}: " + " | ".join(history))


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


class SourceCheck(BaseModel):
    """One concrete question raised while reading a particular source passage."""
    source_id: str = Field(min_length=1, description="ID of the supplied passage raising the question")
    question: str = Field(min_length=1)
    target_id: str | None = Field(default=None, description="Known ID of the passage to consult")
    query: str | None = Field(default=None, description="Exact label, symbol or phrase to find in the paper")
    read_image: bool = Field(default=False, description="Read the PDF page of target_id or source_id for this question")
    region: tuple[float, float, float, float] | None = Field(default=None, description="Optional normalized left,top,right,bottom")

    @model_validator(mode="after")
    def check_location(self):
        if self.region is not None:
            left, top, right, bottom = self.region
            if not self.read_image or not (0 <= left < right <= 1 and 0 <= top < bottom <= 1):
                raise ValueError("Use a normalized rectangle only for a requested PDF image")
        if not (self.target_id or (self.query and self.query.strip()) or self.read_image):
            raise ValueError("Supply the target passage, a concrete search query or an image request")
        return self


class TextFindings(Findings):
    model_config = ConfigDict(title="Findings")
    source_checks: list[SourceCheck] = Field(default_factory=list)


MAX_DEFINITIONS = 6
MAX_GROUP_DEFINITIONS = 2


class Group(BaseModel):
    definitions: list[str] = Field(max_length=MAX_GROUP_DEFINITIONS)
    theorem: str = Field(min_length=1)
    paragraph: str = ""
    paragraph_position: Literal["before", "after"] = Field(default="after", description="Place the connecting paragraph before the group's definitions and theorem, or after the theorem")


class Reference(BaseModel):
    key: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
    text: str = Field(min_length=1, description="Bibliographic entry content only; the renderer supplies the TeX item and environment")


class EvidenceLink(BaseModel):
    block_id: str
    evidence_ids: list[str] = Field(min_length=1)


class EvidenceLinkPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_links: list[EvidenceLink]


class BlockReplacement(EvidenceLink):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(description="Complete revised block; empty only for a requested deletion")
    formal_source: FormalSource | None = Field(default=None, description="Supply only when this formal attribution must change")
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
            if target.get("requires_formal_source") and row.formal_source is None:
                raise ValueError(f"{row.block_id}: supply the formal source for the new definition heading")
            if requested_attrs := target.get("attribution_fields"):
                if row.formal_source is None:
                    raise ValueError(f"{row.block_id}: revise the specified formal_source fields {list(requested_attrs)}")
                updated = row.formal_source.model_dump()
                current = target["formal_source"]
                if any(updated[key] == value for key, value in requested_attrs.items()):
                    raise ValueError(f"{row.block_id}: the requested heading attribution still has its old value")
                if any(updated[key] != value for key, value in current.items() if key not in requested_attrs):
                    raise ValueError(f"{row.block_id}: retain the unrequested heading attribution fields")
                if (not target.get("diagnostics") and all(defect["draft_field"] != row.block_id for defect in target["defects"])
                        and row.text != target["current_text"]):
                    raise ValueError(f"{row.block_id}: preserve the statement body for a heading-only repair")
            revised = " ".join(row.text.split())
            if target.get("remove"):
                if revised or row.formal_source or row.paragraph_position is not None:
                    raise ValueError(f"{row.block_id}: return empty text for a requested deletion")
                if not set(row.evidence_ids) <= registered:
                    raise ValueError("Removal cites unregistered evidence")
                continue
            if row.paragraph_position is not None and "paragraph_position" not in target:
                raise ValueError("Only a result group's connecting paragraph can change position")
            moved = row.paragraph_position is not None and row.paragraph_position != target.get("paragraph_position")
            new_attribution = row.formal_source is not None and row.formal_source.model_dump() != target.get("formal_source")
            fixes_reference = any(ref.key in target.get("missing_reference_keys", ())
                                  for ref in getattr(self, "additional_references", ()))
            if not revised or (not moved and not new_attribution and not fixes_reference and revised == " ".join(target["current_text"].split())):
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
            raise EvidenceCoverageError(blocks)
        if len(self.shared_definitions) + sum(len(g.definitions) for g in self.groups) > MAX_DEFINITIONS:
            raise ValueError("The Introduction has more than six definitions")
        keys = [ref.key for ref in self.references]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate bibliography keys")
        return self


class CurrentDraft(Draft):
    model_config = ConfigDict(title="Draft")
    history: list[str] = Field(max_length=3)


class AuthoritativeDraft(CurrentDraft):
    """Every printed result is an admitted core; citations are owned by formal blocks."""
    core_bindings: dict[str, str] = Field(description="Map every history.i and groups.i.theorem path to its unique admitted candidate_id")
    formal_sources: dict[str, FormalSource] = Field(description="One heading attribution for every historical theorem, current theorem and definition, keyed by its draft path")

    @model_validator(mode="after")
    def bound_to_register(self, info: ValidationInfo):
        histories = {f"history.{i}" for i in range(len(self.history))}
        results = {f"groups.{i}.theorem" for i in range(len(self.groups))}
        formal = histories | results
        if set(self.core_bindings) != formal or len(set(self.core_bindings.values())) != len(formal):
            raise ValueError("Bind each printed historical/current theorem to exactly one distinct approved core")
        formal.update(f"shared_definitions.{i}" for i in range(len(self.shared_definitions)))
        formal.update(f"groups.{i}.definitions.{j}" for i, group in enumerate(self.groups)
                      for j in range(len(group.definitions)))
        if set(self.formal_sources) != formal:
            raise ValueError(f"Provide heading attributions for exactly these blocks: {sorted(formal)}")
        context = info.context or {}
        approved = context.get("approved_cores")
        if approved is not None:
            if set(self.core_bindings.values()) != set(approved):
                raise ValueError("The draft must present precisely the admitted cores in approved_cores")
            links = {link.block_id: set(link.evidence_ids) for link in self.evidence_links}
            for field, key in self.core_bindings.items():
                core = approved[key]
                if (field in histories) != (core.get("kind") == "historical"):
                    raise ValueError(f"{key}: preserve its approved historical/current role")
                if not links[field].intersection(core["evidence_ids"]):
                    raise ValueError(f"{field}: include original evidence for {key}")
        if "passages" in context:
            registered = {row["id"] for row in context["passages"]}
            if any(set(link.evidence_ids) - registered for link in self.evidence_links):
                raise ValueError("Copy evidence_ids from the supplied original passages")
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
    required_change: str = Field(min_length=1, description="Requested edit and applicable rule clause")
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

    @model_validator(mode="before")
    @classmethod
    def unwrap_final_evidence_verdict(cls, value):
        # After a requested source read the wire contract is the verdict itself.
        # A model may retain the first call's wrapper around an otherwise complete verdict.
        if isinstance(value, dict) and value.get("source_checks"):
            raise ValueError("The requested evidence was supplied. Return the final route, issues and evidence_ids.")
        if isinstance(value, dict) and "route" not in value and isinstance(value.get("verdict"), dict):
            return value["verdict"]
        return value

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


def structured_request_text(schema, instructions, payload, *, compact=False):
    # Whitespace in source text, formulae and schema descriptions must survive intact.
    separators = (",", ":") if compact else None
    system = (instructions + "\nReturn one JSON object matching the supplied "
              "schema; escape TeX backslashes in JSON strings.\n"
              + json.dumps(schema.model_json_schema(), ensure_ascii=False, separators=separators))
    return system, json.dumps(payload, ensure_ascii=False, separators=separators)


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
        self.optimized_workflow = is_optimized_workflow(settings)
        self.reliable_workflow = is_reliable_workflow(settings)
        self.patch_evidence_links = uses_evidence_link_patch(settings)
        self.block_workflow = uses_block_workflow(settings)
        from .profile_inputs import uses_targeted_revision
        self.targeted_revision = uses_targeted_revision(settings)
        self.grounded_revision = uses_grounded_revision(settings)
        self.scope_revision = uses_scope_revision(settings)
        self.on_direct_format_repair = None
        text_options = chat_options(settings, "text", timeout=timeout, max_retries=2)
        text_client = ReasoningChatOpenAI if "extra_body" in text_options else ChatOpenAI
        self.text = text_client(**text_options)
        self.vision = ChatOpenAI(**chat_options(settings, "vision", timeout=timeout, max_retries=2))
        observe_http(self.text)
        observe_http(self.vision)
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
        wire_schema = schema
        local_blocks = getattr(self, "block_workflow", False) and schema.__name__ == "DirectResult"
        if local_blocks:
            from .block_response import BlockDirectResult, expand_direct_blocks, place_paragraph_positions
            wire_schema = BlockDirectResult
        targeted = getattr(self, "targeted_revision", False)
        review_form = targeted and schema.__name__ in {"EvidenceReview", "AdmissionReview"}
        if review_form:
            from .review_response import EvidenceReviewForm, ReviewVerdictForm, editorial_response
            wire_schema = EvidenceReviewForm if schema.__name__ == "EvidenceReview" else ReviewVerdictForm
            instructions += ("\nUse nominations on the extract route for missing results. Identify the original heading, "
                             "passage and omission loss. Keep independent edits to existing text in issues.")
        compact = getattr(self, "grounded_revision", False)
        system_text, user_text = structured_request_text(wire_schema, instructions, payload, compact=compact)
        system = SystemMessage(content=system_text)
        original = png_message(user_text, image) if image is not None else HumanMessage(content=user_text)
        next_messages = [system, original]
        original_findings, repair_indexes = None, []
        keep_previous = (issubclass(schema, (Findings, Draft, DraftPatch, AdmissionReview, Review))
                         or (isinstance(payload, dict) and "definition_budget" in payload)
                         or getattr(schema, "retry_with_previous", False))
        history = []
        direct = getattr(self, "optimized_workflow", False) and schema.__name__ == "DirectResult"
        reliable_direct = direct and getattr(self, "reliable_workflow", False)
        retry_diagnostics = []
        request_payload = payload
        row_repairs = getattr(self, "block_workflow", False) and schema.__name__ in {
            "ReadingAnswers", "EvidenceReview", "AdmissionReview"}
        patch = None
        for attempt in range(2 if reliable_direct else 3):
            current_patch, patch = patch, None
            response_schema = current_patch.schema if current_patch else wire_schema
            runner = model.with_structured_output(response_schema, method="json_mode", include_raw=True)
            try:
                response = invoke_tracked(runner, next_messages, operation=response_schema.__name__,
                                          channel="vision" if image is not None else "text",
                                          passages=request_payload.get("passages", ()) if isinstance(request_payload, dict) else (),
                                          schema_attempt=attempt + 1,
                                          **({"previous_schema_failure": retry_diagnostics} if retry_diagnostics else {}))["raw"]
                finish = response.response_metadata.get("finish_reason") or "unknown"
                out = (response.usage_metadata or {}).get("output_tokens", 0)
                text = response.content if isinstance(response.content, str) else ""
            except LengthFinishReasonError as exc:
                # The SDK's parse endpoint can raise before LangChain returns the raw
                # completion, even when the caller explicitly requests include_raw=True.
                finish, text = "length", ""
                out = getattr(exc.completion.usage, "completion_tokens", 0) or 0

            details = ""
            document = None
            retry_diagnostics = []
            next_messages = [system, original]
            request_payload = payload
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
                if direct or row_repairs:
                    retry_diagnostics = [{"field": "document", "code": "output_length"}]
            elif not text.strip():
                details = "Blank JSON output. Produce a complete JSON object using the schema."
                if direct or row_repairs:
                    retry_diagnostics = [{"field": "document", "code": "empty_response"}]
            else:
                try:
                    document = json.loads(unfence(text) if direct else text)
                except json.JSONDecodeError as exc:
                    details = f"Malformed JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}."
                    if direct or row_repairs:
                        retry_diagnostics = [{"field": "document", "code": "invalid_json"}]
                else:
                    if current_patch:
                        try:
                            document = current_patch.merge(document)
                        except (ValidationError, ValueError) as exc:
                            # Use the last existing retry slot for the original complete task.
                            document = None
                            details = "The partial correction was insufficient. Return the complete original response."
                            retry_diagnostics = [{"field": "partial_response", "code":
                                "invalid_partial_rows" if isinstance(exc, ValidationError) else str(exc)}]
                    if reliable_direct:
                        checks, invalid = direct_source_action(document, payload, SourceCheck)
                        if checks:
                            # Validate the action by itself. A simultaneous draft is private,
                            # provisional context for the next read, never a published decision.
                            action = schema.model_validate({"source_checks": checks})
                            action._provisional = provisional_direct_answer(document)
                            action._invalid_checks = invalid
                            return action
                        if invalid:
                            retry_diagnostics = invalid[:16]
                            details = "; ".join(f"{d['field']}: {d['code']}" for d in retry_diagnostics)
                    if local_blocks and not details:
                        moved = []
                        if getattr(self, "scope_revision", False):
                            document, moved = place_paragraph_positions(document)
                        try:
                            document = expand_direct_blocks(document)
                        except ValidationError as exc:
                            retry_diagnostics = safe_direct_errors(wire_schema, exc.errors(include_input=False))
                            for error in retry_diagnostics:
                                error["field"] = "draft." + error["field"]
                            details = "; ".join(f"{d['field']}: {d['code']}" for d in retry_diagnostics)
                        if moved:
                            # Count a saved resend only after the complete canonical draft,
                            # including every source binding, passes on the first attempt.
                            verified = None
                            if not details:
                                try:
                                    verified = schema.model_validate(document, context=payload)
                                except ValidationError:
                                    pass  # The usual correction path below handles the remaining defect.
                            observer = getattr(self, "on_direct_format_repair", None)
                            if observer:
                                observer({"fields": moved, "schema_attempt": attempt + 1,
                                          "validated": verified is not None,
                                          "avoided_retry": verified is not None and attempt == 0})
                            if verified is not None:
                                return verified
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
                            if review_form:
                                return editorial_response(wire_schema, document, payload, schema)
                            return schema.model_validate(document, context=payload)
                        except ValidationError as exc:
                            if direct:
                                # Only lossless representation changes are local. Revalidate against
                                # the same complete source and require the same semantic invariants.
                                normalized = normalize_direct_shape(document)
                                if normalized is not document:
                                    try:
                                        return schema.model_validate(normalized, context=payload)
                                    except ValidationError as remaining:
                                        exc = remaining
                                errors = exc.errors(include_input=False)
                                coverage = errors[0].get("ctx", {}).get("error")
                                if (not attempt and reliable_direct and getattr(self, "patch_evidence_links", False)
                                        and len(errors) == 1 and isinstance(coverage, EvidenceCoverageError)):
                                    kept, targets, packet = prepare_evidence_patch(normalized, coverage.blocks, payload)
                                    if targets:
                                        return self._correct_direct_links(model, schema, normalized, payload,
                                                                          kept, targets, packet)
                                    document = with_evidence_links(normalized, kept)
                                    try:
                                        return schema.model_validate(document, context=payload)
                                    except ValidationError as remaining:
                                        # Removing an orphan row cost no request. Any other failing
                                        # field still gets the normal single correction attempt.
                                        errors = remaining.errors(include_input=False)
                                retry_diagnostics = safe_direct_errors(schema, errors)
                                details = "; ".join(f"{d['field']}: {d['code']}" for d in retry_diagnostics)
                            else:
                                errors, details = _field_errors(exc, explain=keep_previous)
                                if attempt == 0 and hasattr(payload, "budget_correction"):
                                    patch = payload.budget_correction(document)
                                    if patch:
                                        patch.payload["validation_errors"] = details
                                        patch_system, patch_user = structured_request_text(
                                            patch.schema, instructions + "\nRevise only the listed over-budget cores, "
                                            "retaining all source-supported assumptions and conclusions.", patch.payload)
                                        request_payload = patch.payload
                                        next_messages = [SystemMessage(content=patch_system), HumanMessage(content=patch_user)]
                                        history.append(f"{attempt + 1}: definition budget correction, {details}")
                                        continue
                                if row_repairs:
                                    from .partial_response import plan_reply_repair, retry_diagnostics as row_diagnostics
                                    retry_diagnostics = row_diagnostics(errors)
                                    if attempt == 0 or (review_form and attempt == 1 and not current_patch):
                                        patch = plan_reply_repair(schema, document, payload, errors,
                                                                  targeted=review_form,
                                                                  source_index=getattr(self, "review_source_index", None))
                                        if patch:
                                            patch.payload["validation_errors"] = details
                                            patch_system, patch_user = structured_request_text(
                                                patch.schema, instructions + "\nFor this correction use the supplied "
                                                "patch schema and return only its requested rows.", patch.payload, compact=compact)
                                            request_payload = patch.payload
                                            next_messages = [SystemMessage(content=patch_system), HumanMessage(content=patch_user)]
                                            history.append(f"{attempt + 1}: row-level correction, {details}")
                                            continue
                            if issubclass(schema, Findings):
                                indexes = _correctable_candidates(document, errors)
                                if indexes:
                                    original_findings, repair_indexes = document, indexes
                    if not repair_indexes and details and keep_previous and document is not None:
                        next_messages.append(HumanMessage(content=json_text({
                            "previous_complete_json": document, "correction": details,
                            "return": "Correct the reported fields and return JSON matching the schema in the system message."
                        })))
            history.append(f"{attempt+1}: finish={finish}, output_tokens={out}, {details}")
            if reliable_direct:
                if attempt:
                    break
                request_payload = direct_retry_packet(payload, document, retry_diagnostics)
                if request_payload is None:
                    break
                # Initiate a focused correction only when the provider supplied
                # a response to repair. An empty/truncated reply goes back to the
                # graph so another full-source paid call cannot repeat the failure.
                next_messages = [system, HumanMessage(content=json_text(request_payload))]
                continue
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
        raise StructuredResponseError(schema, history)

    def _correct_direct_links(self, model, schema, document, payload, kept, targets, packet):
        """Use the existing correction slot to fix ONLY the inconsistent evidence rows."""
        history = ["evidence_link_coverage", "evidence-link patch"]
        instructions = ("Choose the supplied source passages that actually support each requested block. "
                        "Return exactly one evidence link per block_id. If the supplied sources are "
                        "insufficient, return an empty list. Preserve mathematical claims and source IDs.")
        system, user = structured_request_text(EvidenceLinkPatch, instructions, packet,
                                               compact=getattr(self, "grounded_revision", False))
        runner = model.with_structured_output(EvidenceLinkPatch, method="json_mode", include_raw=True)
        diagnostics = [{"field": "draft.evidence_links", "code": "evidence_link_coverage"}]
        try:
            raw = invoke_tracked(runner, [SystemMessage(content=system), HumanMessage(content=user)],
                                 operation=EvidenceLinkPatch.__name__, channel="text", schema_attempt=2,
                                 passages=packet["passages"], previous_schema_failure=diagnostics)["raw"]
        except LengthFinishReasonError as exc:
            raise StructuredResponseError(schema, [history[0], "incomplete evidence-link patch"]) from exc
        try:
            if raw.response_metadata.get("finish_reason") == "length" or not isinstance(raw.content, str):
                raise ValueError("Incomplete evidence patch")
            patch = EvidenceLinkPatch.model_validate_json(unfence(raw.content))
            if sorted(row.block_id for row in patch.evidence_links) != targets:
                raise ValueError("Incorrect evidence patch targets")
            shown = {row["id"] for row in packet["passages"]}
            if any(not set(row.evidence_ids) <= shown for row in patch.evidence_links):
                raise ValueError("Evidence patch cites an unseen passage")
            links = [row.model_dump() for row in patch.evidence_links]
            # Only evidence links can change. Re-run every semantic validator,
            # including core identity and source membership, on the entire paper.
            return schema.model_validate(with_evidence_links(document, [*kept, *links]), context=payload)
        except ValidationError as exc:
            codes = safe_direct_errors(schema, exc.errors(include_input=False))
            history[-1] += " / " + ", ".join(row["code"] for row in codes)
            raise StructuredResponseError(schema, history) from exc
        except ValueError as exc:
            raise StructuredResponseError(schema, [history[0], str(exc)]) from exc


    def transcribe(self, image: bytes, page: int) -> str:
        response = invoke_tracked(self.vision, [SystemMessage(content=(
            "Read the mathematical source page. Transcribe visible definitions, statements, exact "
            "symbols, qualifiers and source labels. Flag any unreadable formula; do not infer it."
        )), png_message(f"Inspect page {page}.", image)], operation="source_transcription", channel="vision")
        return str(response.content)

    def read_source_region(self, image: bytes, page: int, question: str) -> str:
        """Answer the actual reading question from this crop, including uncertainty."""
        reply = invoke_tracked(self.vision, [
            SystemMessage(content="Read this source image to resolve the stated mathematical question. "
                          "Quote the relevant expression and conditions. Say which marks remain unreadable."),
            png_message(json_text({"page": page, "question": question}), image),
        ], operation="source_region_reading", channel="vision")
        return str(reply.content)

