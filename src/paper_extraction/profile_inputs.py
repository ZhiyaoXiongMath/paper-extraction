"""Model-facing parts of the versioned extraction profile."""

import re
from contextlib import contextmanager
from contextvars import ContextVar

from .profile_variants import (
    CURRENT_PROFILE, RELIABILITY_PROFILE, TOKEN_PROFILE, current_profile_fingerprint, profile_assets_for,
)

_active_release = ContextVar("paper_extraction_release", default=CURRENT_PROFILE)
TOKEN_WORKFLOW = "token-efficient-20261119"
RELIABLE_WORKFLOW = "reliable-delivery-20261119"
PREFLIGHT_WORKFLOW = "local-preflight-20261119"
EVIDENCE_WORKFLOW = "evidence-links-20261119"
BLOCK_WORKFLOW = "block-evidence-20261119"
TARGETED_WORKFLOW = "targeted-revision-20261119"
WORKFLOW_REVISION = "grounded-revision-20261119"


@contextmanager
def using_profile(release: str | None):
    """A resumed worker and its LangGraph tasks read their original release."""
    token = _active_release.set(release or CURRENT_PROFILE)
    try:
        yield
    finally:
        _active_release.reset(token)


def installed_profile(release: str | None = None) -> dict[str, str]:
    release = release or CURRENT_PROFILE
    return {"extraction_profile": release, "profile_fingerprint": current_profile_fingerprint(release)}


def new_batch_profile(release: str | None = None) -> dict[str, str]:
    profile = installed_profile(release)
    if profile["extraction_profile"] == CURRENT_PROFILE:
        profile["workflow_revision"] = WORKFLOW_REVISION
    elif profile["extraction_profile"] == RELIABILITY_PROFILE:
        profile["workflow_revision"] = RELIABLE_WORKFLOW
    elif profile["extraction_profile"] == TOKEN_PROFILE:
        profile["workflow_revision"] = TOKEN_WORKFLOW
    return profile


def is_optimized_workflow(settings: dict) -> bool:
    # Absence is meaningful: existing checkpoints used the original graph.
    return is_reliable_workflow(settings) or (
        settings.get("workflow_revision") == TOKEN_WORKFLOW and settings.get("extraction_profile") == TOKEN_PROFILE)


def is_reliable_workflow(settings: dict) -> bool:
    return uses_local_preflight(settings) or (
        settings.get("workflow_revision") == RELIABLE_WORKFLOW
        and settings.get("extraction_profile") in (CURRENT_PROFILE, RELIABILITY_PROFILE))


def uses_local_preflight(settings: dict) -> bool:
    return (settings.get("workflow_revision") in (PREFLIGHT_WORKFLOW, EVIDENCE_WORKFLOW, BLOCK_WORKFLOW, TARGETED_WORKFLOW, WORKFLOW_REVISION)
            and settings.get("extraction_profile") == CURRENT_PROFILE)


def uses_evidence_link_patch(settings: dict) -> bool:
    return (settings.get("workflow_revision") in (EVIDENCE_WORKFLOW, BLOCK_WORKFLOW, TARGETED_WORKFLOW, WORKFLOW_REVISION)
            and settings.get("extraction_profile") == CURRENT_PROFILE)


def uses_block_workflow(settings: dict) -> bool:
    return (settings.get("workflow_revision") in (BLOCK_WORKFLOW, TARGETED_WORKFLOW, WORKFLOW_REVISION)
            and settings.get("extraction_profile") == CURRENT_PROFILE)


def uses_targeted_revision(settings: dict) -> bool:
    return (settings.get("workflow_revision") in (TARGETED_WORKFLOW, WORKFLOW_REVISION)
            and settings.get("extraction_profile") == CURRENT_PROFILE)


def uses_grounded_revision(settings: dict) -> bool:
    """Only newly created batches use source-bound nomination and revision contracts."""
    return (settings.get("workflow_revision") == WORKFLOW_REVISION
            and settings.get("extraction_profile") == CURRENT_PROFILE)


def current_rules(*sections: str, release: str | None = None) -> str:
    """Return original section text in the requested order, or the full rules for selection."""
    text = (profile_assets_for(release or _active_release.get()) / "rules.txt").read_text(encoding="utf-8").strip()
    if not sections:
        return text
    chunks = re.split(r"(?m)(?=^## )", text)
    parts = {}
    for chunk in chunks[1:]:
        heading = chunk.splitlines()[0]
        key = "budget" if heading == "## Deliverable and budgets" else heading[3:].split(".", 1)[0]
        parts[key] = chunk.strip()
    return "\n\n".join(parts[key] for key in sections)


def node_prompt(name: str, release: str | None = None) -> str:
    text = (profile_assets_for(release or _active_release.get()) / "prompt.txt").read_text(encoding="utf-8")
    chunks = re.split(r"(?m)^\[([a-z_]+)\]\s*$", text)
    instructions = dict(zip(chunks[1::2], chunks[2::2]))
    return instructions["common"].strip() + "\n\n" + instructions[name].strip()
