"""Provider-facing review forms. Stored verdicts retain the checkpoint's edit contract."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from .lean_workflow import EvidenceReview
from .models import AdmissionReview, EditorialDefect


class ResultNomination(BaseModel):
    """A potentially missing result for the selector to assess against the original."""
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    scope: Literal["current", "historical"]
    source_heading: str = Field(min_length=1, description="Actual heading or attribution identifying the result.")
    evidence_id: str = Field(min_length=1)
    source_excerpt: str = Field(min_length=1)
    omission_loss: str = Field(min_length=1, description="Distinct central contribution, application, or comparison omitted.")
    candidate_id: str | None = Field(default=None, description="Existing candidate ID when available.")

    @model_validator(mode="after")
    def registered_location(self, info: ValidationInfo):
        packet = info.context or {}
        registered = {row["id"] for key in ("passages", "source_locations") for row in packet.get(key, ())}
        if registered and self.evidence_id not in registered:
            raise ValueError("A nomination must use a registered evidence ID")
        return self

    def edit(self):
        reference = self.source_heading
        if self.candidate_id:
            reference += f" ({self.candidate_id})"
        return EditorialDefect(kind="missing_required_content", operation="recover",
                               draft_field="history" if self.scope == "historical" else "groups",
                               current_excerpt="", evidence_id=self.evidence_id, source_excerpt=self.source_excerpt,
                               required_change=f"Assess {reference} for selection. {self.omission_loss}")


class ReviewVerdictForm(BaseModel):
    route: Literal["pass", "extract", "draft"]
    issues: list[EditorialDefect] = Field(description="Actions on manuscript material. Use [] if none.")
    nominations: list[ResultNomination] = Field(default_factory=list, description="Missing results only, on the extract route.")
    evidence_ids: list[str] = Field(default_factory=list, description="Additional locations to revisit on extract.")

    @model_validator(mode="before")
    @classmethod
    def unwrap_final_response(cls, value):
        return AdmissionReview.unwrap_final_evidence_verdict(value)

    @model_validator(mode="after")
    def check_nomination_route(self):
        if self.nominations and self.route != "extract":
            raise ValueError("A result nomination requires the extract route")
        return self

    def editorial(self):
        # The selector owns admission. The ordinary graph receives one recovery
        # issue per nomination and keeps any unrelated edits from the same reply.
        locations = [*self.evidence_ids, *(row.evidence_id for row in self.nominations)]
        return {"route": self.route, "evidence_ids": list(dict.fromkeys(locations)),
                "issues": [*(row.model_dump() for row in self.issues),
                           *(row.edit().model_dump() for row in self.nominations)]}


class EvidenceReviewForm(EvidenceReview):
    verdict: ReviewVerdictForm | None = None

    @model_validator(mode="before")
    @classmethod
    def accept_recognizable_envelope(cls, value):
        if isinstance(value, dict) and not value.get("source_checks") and value.get("verdict") is None and "route" in value:
            return {"verdict": value}
        return EvidenceReview.accept_recognizable_envelope(value)


def editorial_response(form, document, packet, schema):
    parsed = form.model_validate(document, context=packet)
    if isinstance(parsed, EvidenceReviewForm):
        if parsed.source_checks:
            return schema.model_validate({"source_checks": [q.model_dump() for q in parsed.source_checks]}, context=packet)
        proposal = parsed.verdict
        result = schema.model_validate({"verdict": proposal.editorial()}, context=packet)
        verdict = result.verdict
    else:
        proposal = parsed
        result = verdict = schema.model_validate(proposal.editorial(), context=packet)
    # Keep the typed wire data on the local verdict. Older checkpoints and
    # the schema used by legacy model calls keep their original edit contract.
    verdict._source_nominations = [row.model_dump() for row in proposal.nominations]
    return result
