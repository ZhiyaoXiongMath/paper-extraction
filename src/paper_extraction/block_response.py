"""Model-facing whole-paper draft. Storage and all mathematical checks use AuthoritativeDraft."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .citation_types import FormalSource
from .lean_workflow import DirectCore, ScreeningNote
from .models import Reference, SourceCheck


class BodyBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(description="Registered passages supporting this block; include at least one.")

    @model_validator(mode="before")
    @classmethod
    def keep_text_for_link_repair(cls, value):
        # Keep an otherwise valid body so the evidence-only correction can recover
        # a missing row. The canonical draft rejects every unsupported block.
        if isinstance(value, dict) and "evidence_ids" not in value:
            return {**value, "evidence_ids": []}
        return value


class DefinitionBlock(BodyBlock):
    formal_source: FormalSource


class TheoremBlock(DefinitionBlock):
    candidate_id: str = Field(description="ID of this result in cores.")


class ResultGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")
    definitions: list[DefinitionBlock] = Field(max_length=2)
    theorem: TheoremBlock
    paragraph: BodyBlock | None = None
    paragraph_position: Literal["before", "after"] = "after"

    @field_validator("paragraph", mode="before")
    @classmethod
    def empty_paragraph(cls, value):
        return None if value == "" else value


class BlockDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    authors: list[str]
    paragraph: BodyBlock
    history: list[TheoremBlock] = Field(max_length=3)
    shared_definitions: list[DefinitionBlock] = Field(max_length=2)
    groups: list[ResultGroup] = Field(min_length=1)
    references: list[Reference] = Field(min_length=1)

    def linked_draft(self):
        links, attributions, bindings = [], {}, {}

        def body(path, block):
            if block.evidence_ids:
                links.append({"block_id": path, "evidence_ids": block.evidence_ids})
            if isinstance(block, DefinitionBlock):
                attributions[path] = block.formal_source.model_dump()
            if isinstance(block, TheoremBlock):
                bindings[path] = block.candidate_id
            return block.text

        opening = body("paragraph", self.paragraph)
        history = [body(f"history.{i}", item) for i, item in enumerate(self.history)]
        shared = [body(f"shared_definitions.{i}", item) for i, item in enumerate(self.shared_definitions)]
        groups = []
        for i, group in enumerate(self.groups):
            path = f"groups.{i}"
            groups.append({
                "definitions": [body(f"{path}.definitions.{j}", item) for j, item in enumerate(group.definitions)],
                "theorem": body(f"{path}.theorem", group.theorem),
                "paragraph": body(f"{path}.paragraph", group.paragraph) if group.paragraph else "",
                "paragraph_position": group.paragraph_position,
            })
        return {"title": self.title, "authors": self.authors, "paragraph": opening, "history": history,
                "shared_definitions": shared, "groups": groups,
                "references": [ref.model_dump() for ref in self.references],
                "evidence_links": links, "core_bindings": bindings, "formal_sources": attributions}


class BlockDirectResult(BaseModel):
    """Choose a source check, a complete supported draft, or a screening note."""
    cores: list[DirectCore] = Field(default_factory=list)
    draft: BlockDraft | None = None
    summary: ScreeningNote | None = None
    source_checks: list[SourceCheck] = Field(default_factory=list)


def expand_direct_blocks(document):
    draft = document.get("draft") if isinstance(document, dict) else None
    if isinstance(draft, dict) and isinstance(draft.get("paragraph"), dict):
        return {**document, "draft": BlockDraft.model_validate(draft).linked_draft()}
    # Includes a legacy reply or provisional draft from an existing checkpoint.
    return document
