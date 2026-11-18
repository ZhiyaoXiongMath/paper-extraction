"""Model-facing parts of the versioned extraction profile."""

import re
from contextlib import contextmanager
from contextvars import ContextVar

from .profile_variants import CURRENT_PROFILE, current_profile_fingerprint, profile_assets_for

_active_release = ContextVar("paper_extraction_release", default=CURRENT_PROFILE)
WORKFLOW_REVISION = "token-efficient-20261119"


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
    return profile


def is_optimized_workflow(settings: dict) -> bool:
    # Absence is meaningful: existing checkpoints used the original graph.
    return settings.get("workflow_revision") == WORKFLOW_REVISION


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
