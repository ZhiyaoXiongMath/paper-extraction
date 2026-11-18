"""Keep the historical format reproducible and derive the current reader rules."""

import hashlib
import shutil
from pathlib import Path

from filelock import FileLock

from .profile_assets.tools import v13_check


STANDARD_CURRENT_MAX = 4
UNLIMITED_CURRENT = v13_check.UNLIMITED_CURRENT_RELEASE
READER_PROFILE = "reader"
COUNT_OVERRIDE = (
    "User-authorized change to the v13 scope: choose the number of current theorems from the "
    "paper's admitted contributions. The original four-current-theorem limit is waived. Preserve "
    "the source and admission rules. Use the separately specified reader settings for layout. "
    "Keep flat consecutive current-theorem numbering."
)


def reader_guidance() -> str:
    limits = v13_check.EXPANDED_BLOCK_LIMITS
    targets = ", ".join(f"{name} about {limit['words']} words and up to {limit['units']}U"
                        for name, limit in limits.items())
    return (
        "Apply the following user-authorized editorial changes in place of the older numerical and "
        "exposition limits in the supplied companions. Select the number of current theorems from the "
        "source-supported contributions and keep flat consecutive numbering. Aim to finish the body within "
        "three pages. Use more space when needed for the mathematics or source-supported explanation, "
        f"up to {v13_check.EXPANDED_BODY_PAGES} body pages. Start the bibliography on a separate page. "
        "Use explanatory paragraphs to motivate the problem, explain "
        "what changes relative to earlier results, or interpret a conclusion's scope and its relationship "
        "to independent applications. Add source-supported explanation where it helps the reader, avoiding "
        "repeated statements and proof-only technical details. "
        f"Editorial targets and measured heights: {targets}. Word counts are advisory. A result group "
        f"has up to {v13_check.EXPANDED_GROUP_HEIGHT}U; total prose and shared definitions have up to "
        f"{v13_check.EXPANDED_PROSE_HEIGHT}U and {v13_check.EXPANDED_SHARED_HEIGHT}U respectively. "
        f"Cumulative display rows may reach {limits['historical']['rows']} for history, "
        f"{limits['definition']['rows']} for definitions and {limits['theorem']['rows']} for current results. "
        "Keep source attribution, necessary hypotheses, numbering, measured environments, formula "
        "alignment, typography and separate bibliography."
    )

_CHANGES = {
    "intro-rules-v13.txt": (
        ("Theorem 1, 2, 3, 4        1–4", "Theorem 1, 2, ...         1+ "),
        ("followed by Theorems 1/2/3/4.", "followed by Theorems 1, 2, and so on."),
    ),
    "prompt-v13.txt": (("Theorem 1/2/3/4 for", "Theorem 1, 2, and so on for"),),
    "template-v13.tex": (("another resultgroup, no more than FOUR in all.", "another resultgroup for each admitted core."),),
    "preamble-v13.tex": (
        (r"  \ifnum\value{rawtheorem}>4 \IntroBudgetError{More than four current theorems}\fi" + "\n",
         "  % The current-theorem count is selected under the user-authorized scope.\n"),
    ),
}


def release_for_draft(draft: dict) -> str:
    if draft.get("editorial_profile") == READER_PROFILE:
        return v13_check.EXPANDED_RELEASE
    return UNLIMITED_CURRENT if len(draft.get("groups", [])) > STANDARD_CURRENT_MAX else v13_check.RELEASE


def _expanded_changes(name: str):
    if name == "preamble-v13.tex":
        old, new = v13_check.BLOCK_LIMITS, v13_check.EXPANDED_BLOCK_LIMITS
        changes = [(rf"\IntroMeasured{{{limit['kind']}}}{{{limit['units']}}}",
                    rf"\IntroMeasured{{{limit['kind']}}}{{{new[block]['units']}}}") for block, limit in old.items()]
        for old_units, units, message in (
            (22, v13_check.EXPANDED_GROUP_HEIGHT, "Result group exceeds"),
            (16, v13_check.EXPANDED_PROSE_HEIGHT, "Total connecting prose exceeds"),
            (14, v13_check.EXPANDED_SHARED_HEIGHT, "Shared definitions exceed"),
        ):
            changes.extend(((rf"\IntroLimit={old_units}\IntroUnit", rf"\IntroLimit={units}\IntroUnit"),
                            (f"{message} {old_units}U", f"{message} {units}U")))
        changes.append(("Every block is now bounded by at most 12U:", "Each measured block is bounded:"))
        return changes
    if name == "prompt-v13.txt":
        return [("Use the rules' exact word, rendered-height, formula-row and 22U group budgets.",
                 "Use the active reader targets and measured budgets in the editorial settings.")]
    if name == "template-v13.tex":
        return [("At most 45 prose words and 4U per introparagraph.", "See the active reader targets for prose."),
                ("each <=65 prose words / 7U / 3 rows.", "within the definition budget."),
                ("<=75 prose words / 8U / 2 rows.", "within the historical budget."),
                ("Entire group <=22U; theorem <=100 prose words / 12U / 3 display rows.",
                 "Use the active group and theorem budgets.")]
    return []


def companion_text(base: Path, name: str, release: str) -> str:
    text = (base / name).read_text(encoding="utf-8")
    if release == v13_check.RELEASE:
        return text
    if release not in {UNLIMITED_CURRENT, v13_check.EXPANDED_RELEASE}:
        raise ValueError("Unknown Introduction profile")
    # Exact substitutions make changes to the frozen upstream profile fail loudly.
    changes = list(_CHANGES[name])
    if release == v13_check.EXPANDED_RELEASE:
        changes.extend(_expanded_changes(name))
        if name == "intro-rules-v13.txt":
            text += "\nACTIVE READER SETTINGS\n" + reader_guidance() + "\n"
    for before, after in changes:
        if text.count(before) != 1:
            raise ValueError(f"Unexpected v13 count clause in {name}")
        text = text.replace(before, after)
    return text.replace(v13_check.RELEASE, release)


def profile_bundle(base: Path, work: Path, release: str) -> Path:
    if release == v13_check.RELEASE:
        return base
    work.mkdir(parents=True, exist_ok=True)
    bundle_id = v13_check.digest(base / "SHA256SUMS.txt")[:12]
    edition = "reader" if release == v13_check.EXPANDED_RELEASE else "unlimited"
    companions = {name: companion_text(base, name, release) for name in v13_check.FILES}
    if edition == "reader":
        # Changes to the effective rules must not reuse an older on-disk bundle.
        fingerprint = "\0".join((bundle_id, *companions.values())).encode("utf-8")
        bundle_id = hashlib.sha256(fingerprint).hexdigest()[:12]
    target = work / f".v13-{edition}-{bundle_id}"
    with FileLock(target.with_suffix(".lock")):
        if (target / "SHA256SUMS.txt").is_file():
            if v13_check.verify(target)["release"] != release:
                raise ValueError("Unexpected current-theorem scope in the local v13 bundle")
            return target
        v13_check.verify(base)
        (target / "tools").mkdir(parents=True, exist_ok=True)
        for name, content in companions.items():
            (target / name).write_text(content, encoding="utf-8", newline="\n")
        shutil.copyfile(base / v13_check.TOOL, target / v13_check.TOOL)
        manifest = target / "SHA256SUMS.txt"
        pending = manifest.with_suffix(".tmp")
        pending.write_text("".join(f"{v13_check.digest(target / name)}  {name}\n"
                                   for name in (*v13_check.FILES, v13_check.TOOL)), encoding="ascii")
        pending.replace(manifest)
    return target
