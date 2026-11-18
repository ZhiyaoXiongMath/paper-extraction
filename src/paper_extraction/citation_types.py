"""Types shared by the writing, repair and reading paths for formal attributions."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FormalSource(BaseModel):
    """The source of one particular theorem/definition, distinct from a comparison."""
    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1, description=r"Exact source citation, e.g. \cite[Theorem~5.15]{source}.")
    prior: str = Field(default="", description=r"Historical author and original reference if available, e.g. Bedford and Taylor \cite{bt}.")
    reporting: Literal["as reported in", "as stated in", "as restated in"] = "as reported in"
    comparison: str = Field(default="", description="Separate source citation for this result group's comparison.")

    @field_validator("source", "comparison")
    @classmethod
    def require_citation(cls, text: str) -> str:
        text = text.strip()
        if text and r"\cite" not in text:
            raise ValueError("Identify the source with a bibliography citation")
        return text

    @field_validator("source")
    @classmethod
    def require_primary(cls, text: str) -> str:
        if not text:
            raise ValueError("The formal statement needs its own source")
        return text
