"""Typed contracts for local models and the two OpenAI-compatible providers."""
from __future__ import annotations

import base64
import json
from typing import Literal, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import LengthFinishReasonError
from pydantic import BaseModel, Field, ValidationError, model_validator

Model = TypeVar("Model", bound=BaseModel)


class Candidate(BaseModel):
    kind: Literal["historical", "current", "definition", "application", "setting"]
    claim: str = Field(min_length=1)
    assumptions: str
    significance: str
    source_number: str | None = Field(description="Original theorem/definition label, null if unnumbered")
    dependencies: list[str] = Field(description="Names and meanings of definitions needed by the claim")
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


class Reference(BaseModel):
    key: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")
    text: str = Field(min_length=1)


class EvidenceLink(BaseModel):
    block_id: str
    evidence_ids: list[str] = Field(min_length=1)


class Draft(BaseModel):
    title: str = Field(min_length=1)
    authors: list[str] = Field(min_length=1)
    paragraph: str = Field(min_length=1)
    history: list[str] = Field(min_length=1, max_length=3)
    shared_definitions: list[str] = Field(max_length=2)
    groups: list[Group] = Field(min_length=1, max_length=4)
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


class Review(BaseModel):
    passed: bool
    issues: list[str]
    @model_validator(mode="after")
    def require_consistency(self):
        if self.passed == bool(self.issues):
            raise ValueError("A passing review has no issues. A failing review identifies the defects.")
        return self


class AdmissionReview(BaseModel):
    route: Literal["pass", "extract", "draft"]
    issues: list[str]
    evidence_ids: list[str]

    @model_validator(mode="after")
    def validate_route(self):
        if (self.route == "pass") == bool(self.issues):
            raise ValueError("A pass has no issues; a repair identifies a source-specific issue")
        if self.route == "extract" and not self.evidence_ids:
            raise ValueError("Target missing claims by their original source IDs")
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


def _field_errors(exc):
    # Pydantic's ordinary repr includes input values. Only expose field names and
    # schema expectations, never paper passages or a raw provider response.
    errors = exc.errors(include_url=False, include_input=False, include_context=False)
    details = []
    for error in errors[:16]:
        location = ".".join(map(str, error["loc"])) or "document"
        details.append(f"{location}: {error['type']} ({error['msg']})")
    return errors, "; ".join(details)



class Models:
    def __init__(self, settings: dict):
        if not settings.get("api_key"):
            raise ValueError("The selected provider requires an API key in the local configuration")
        opts = {
            "api_key": settings["api_key"], "base_url": settings["base_url"],
            "timeout": settings.get("timeout", 120), "max_retries": 2,
        }
        if settings["provider"].lower() in {"deepseek", "zhipu", "glm"}:
            opts["extra_body"] = {"thinking": {"type": "disabled"}}
        self.text = ChatOpenAI(model=settings["text_model"], **opts)
        self.vision = ChatOpenAI(model=settings["vision_model"], **opts)

    def structured(self, schema: type[Model], instructions: str, payload,
                   image: bytes | None = None) -> Model:
        model = self.vision if image is not None else self.text
        contract = schema.model_json_schema()
        system = SystemMessage(content=(
            instructions + "\nProduce a single complete JSON object. Escape all TeX backslashes "
            r'in JSON strings (write "\\alpha" for the TeX command). Use the exact field names, '
            "enum values and data types. An unnumbered source label is null; a numbered label is "
            "a string. Use [] for empty arrays.\nReturn JSON matching this schema:\n" + json_text(contract)
        ))
        original = png_message(json_text(payload), image) if image is not None else HumanMessage(content=json_text(payload))
        next_messages = [system, original]
        original_findings, repair_indexes = None, []
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
                            return schema.model_validate(document)
                        except ValidationError as exc:
                            errors, details = _field_errors(exc)
                            if schema is Findings:
                                indexes = _correctable_candidates(document, errors)
                                if indexes:
                                    original_findings, repair_indexes = document, indexes
                    if not repair_indexes and details:
                        next_messages.append(HumanMessage(content=json_text({
                            "previous_complete_json": document, "correction": details,
                            "preserve": "Keep every valid candidate, claim, and original evidence ID."
                        })))
            history.append(f"{attempt+1}: finish={finish}, output_tokens={out}, {details}")
            if repair_indexes:
                next_messages.append(HumanMessage(content=json_text({
                    "correction": details,
                    "rows_to_fix_in_order": [{"original_index": i, "value": original_findings["candidates"][i]}
                                             for i in repair_indexes],
                    "return": "Use the Findings schema. Return only these corrected candidates in the "
                              "same order. Keep the original claim and evidence for each. Use [] for uncertainties."
                })))
            elif len(next_messages) == 2:
                next_messages.append(HumanMessage(content=details + " Use the JSON schema in the system message."))
        raise ValueError(f"Invalid {schema.__name__} response after three attempts: " + " | ".join(history))


    def transcribe(self, image: bytes, page: int) -> str:
        response = self.vision.invoke([SystemMessage(content=(
            "Read the mathematical source page. Transcribe visible definitions, statements, exact "
            "symbols, qualifiers and source labels. Flag any unreadable formula; do not infer it."
        )), png_message(f"Inspect page {page}.", image)])
        return str(response.content)

