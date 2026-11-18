"""Typed contracts for local models and the two OpenAI-compatible providers."""
from __future__ import annotations

import base64
import json
from typing import Literal, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, model_validator

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
        schema_text = json_text(schema.model_json_schema())
        system = SystemMessage(content=f"{instructions}\nReturn JSON matching this schema:\n{schema_text}")
        user = png_message(json_text(payload), image) if image is not None else HumanMessage(content=json_text(payload))
        runner = model.with_structured_output(schema, method="json_mode", include_raw=True)
        messages = [system, user]
        for attempt in range(2):
            result = runner.invoke(messages)
            if result["parsed"] is not None:
                return schema.model_validate(result["parsed"])
            if attempt == 0:
                messages.extend([result["raw"], HumanMessage(content=(
                    "Correct the JSON shape and all field validation errors. "
                    + str(result.get("parsing_error", ""))[:1800]
                ))])
        raise ValueError(f"Model did not return valid {schema.__name__} JSON on two attempts")

    def transcribe(self, image: bytes, page: int) -> str:
        response = self.vision.invoke([SystemMessage(content=(
            "Read the mathematical source page. Transcribe visible definitions, statements, exact "
            "symbols, qualifiers and source labels. Flag any unreadable formula; do not infer it."
        )), png_message(f"Inspect page {page}.", image)])
        return str(response.content)

