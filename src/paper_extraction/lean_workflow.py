"""Contracts shared by the whole-source and segmented admission paths."""

from __future__ import annotations

import re
import unicodedata
from copy import deepcopy
from typing import ClassVar, Literal

from pydantic import BaseModel, Field, ValidationInfo, model_validator

from .citation_types import FormalSource
from .models import AdmissionReview, AuthoritativeDraft, Candidate, Reference, SourceCheck, json_text


class ScreeningNote(BaseModel):
    kind: Literal["no_qualifying_result", "insufficient_evidence"]
    question: str = Field(min_length=1, description="The paper's mathematical question, briefly in Chinese")
    reason: str = Field(min_length=1, description="Source-specific reason for the outcome, in Chinese")
    evidence_ids: list[str]


class IndependentRole(BaseModel):
    role: Literal["principal_result", "independent_application", "explicit_comparison"]
    source_id: str
    quote: str = Field(min_length=1, description="Verbatim source words identifying this result's role outside its own statement and proof")


def _flat(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


_LEMMA = r"(?:lemma|lemme|引理)"
_ROLE = {
    "principal_result": re.compile(r"\b(?:main|principal|central|fundamental|major|key result|highlight|most important|purpose of (?:this|the)|objet principal|résultat principal|hauptresultat)\b|主[要定]|核心", re.I),
    "independent_application": re.compile(r"\b(?:application|appliqu\w*|as a consequence|in particular.{0,60}(?:solves?|settles?|answers?)|settles?|answers?.{0,12}(?:the|an|a) (?:question|problem))\b|应用|解决.{0,12}问题", re.I),
    "explicit_comparison": re.compile(r"\b(?:compar\w*|in contrast|unlike|whereas|while|extend\w*|generaliz\w*|generalises?|strengthen\w*|weaken\w*|improv\w*|replac\w*|beyond|analog\w*|generalization|généralis\w*|extension|baseline|earlier.{0,60}(?:require|assum)\w*)\b|相比|推广|加强|弱化", re.I),
}


def _lemma_locator(source_number, evidence):
    label = _flat(source_number or "")
    if re.search(rf"\b{_LEMMA}\b", label):
        return re.sub(rf".*?{_LEMMA}\s*[:~.]?\s*", "", label)
    number = re.fullmatch(r"[a-z]?\d+(?:\.\d+)*(?:[a-z])?", label)
    if number and re.search(rf"\b{_LEMMA}[.\s~:]*{re.escape(label)}(?!\d|\.\d|[a-z])", evidence):
        return label
    return None


def requires_role(kind, source_number, evidence_ids, records):
    evidence = _flat(" ".join(records[i]["text"] for i in evidence_ids if i in records))
    label = _lemma_locator(source_number, evidence)
    return kind == "historical" or label is not None, label


def independent_role_supported(kind, source_number, claim, evidence_ids, role, records):
    """A lemma's statement and an invocation in a proof establish truth/use,
    but neither is an attribution of independent importance by the paper."""
    required, label = requires_role(kind, source_number, evidence_ids, records)
    if not required:
        return True
    if not role or role.source_id not in records:
        return False
    quote = _flat(role.quote)
    passage = _flat(records[role.source_id]["text"])
    if len(quote) < 12 or quote not in passage:
        return False
    # A caption may be in another language or say "our aim is ..." instead
    # of "main". Accept author framing in the introduction as an alternative.
    intro = records[role.source_id].get("page") in {1, 2} or bool(re.search(
        r"abstract|introduct|résumé|einleitung|引言|摘要", str(records[role.source_id].get("section", "")), re.I))
    framing = intro and bool(re.search(r"\b(?:we|our|nous|notre|wir|unser\w*|this (?:paper|article))\b|我们|本文", quote))
    if not (_ROLE[role.role].search(quote) or framing):
        return False
    stop = {"theorem", "lemma", "there", "exists", "existence", "unique", "result", "which", "every",
            "where", "class", "operator", "paper", "following", "positive", "function", "functions",
            "bounded", "manifold", "space", "dimension", "assume", "given", "statement", "earlier",
            "known", "study", "establish"}
    def words(text):
        return set(re.findall(r"[^\W\d_]{4,}", _flat(text))) - stop
    if kind == "historical":
        # The phrase "comparison principle" can simply be the name of a theorem.
        # It is not by itself a comparison with a current contribution.
        prose = re.sub(r"comparison principles?", "", quote)
        if role.role != "explicit_comparison" or not (_ROLE[role.role].search(prose) or framing):
            return False
        number = _flat(source_number or "")
        labels = re.findall(r"\d+(?:\.\d+)+", number)
        if any(re.search(r"(?<![\w.])" + re.escape(n) + r"(?!\d|\.\d)", quote) for n in labels):
            return True
        if len(words(claim) & words(quote)) >= 2 or words(number) & words(quote):
            return True
        # A comparative sentence can refer to the preceding quoted theorem as
        # "this result". Accept the literal juxtaposition within its own source.
        if role.source_id in evidence_ids and re.search(r"\b(?:this|that|previous|preceding) (?:result|theorem|criterion|statement)", quote):
            idx = passage.index(quote)
            return len(words(claim) & words(passage[max(0, idx - 400):idx])) >= 3
        return False
    if role.role == "explicit_comparison":
        return False
    # Link the author's framing to this particular lemma. A generic paragraph
    # announcing a different main theorem cannot certify every lemma on its page.
    if label and re.search(rf"(?<![\w.]){re.escape(label)}(?!\d|\.\d|[a-z])", quote):
        mentions = re.search(rf"(?P<intro>[^.;]{{0,100}})\b{_LEMMA}[\s~:]*{re.escape(label)}", quote)
        if mentions and re.search(r"\b(?:by|using|uses?|applying|follows from|in the proof of)\s*$", mentions["intro"]):
            return False
        return not re.search(r"\b(?:used|useful|needed|applied|ingredient|step)\s+(?:in|for|to)\s+(?:the |our )?(?:proof|prove|main theorem)", quote)
    # Some introductions announce a result in words instead of by number.
    # Require connection through its mathematical language beyond generic terms.
    return len(words(claim) & words(quote)) >= 3


class DirectCore(BaseModel):
    candidate_id: str = Field(pattern=r"^c[1-9]\d*$")
    kind: Literal["current", "application", "historical"]
    source_number: str | None
    claim: str = Field(min_length=1, description="Short mathematical identification of the retained assertion")
    independent_question: str = Field(min_length=1)
    source_reason: str = Field(min_length=1, description="What this adds over the nearest known or retained result and why that matters")
    evidence_ids: list[str] = Field(min_length=1)
    significance_source_id: str | None = Field(default=None, description="Passage supporting the gain if it differs from the primary assertion's first location")
    independent_role: IndependentRole | None = None

    def register_row(self):
        return {**self.model_dump(), "decision": "admitted", "decision_reason": self.source_reason,
                "significance": self.source_reason, "assumptions": "", "dependencies": [],
                "local_kind": self.kind, "local_evidence_ids": self.evidence_ids,
                "selection_evidence_ids": self.evidence_ids,
                "b2_source_facts": [{"source_id": self.significance_source_id or self.evidence_ids[0], "fact": self.source_reason}]}


def direct_admission(result, records):
    """Apply the same source-role contract as segmented selection before a draft
    can be checkpointed. Keep excluded entries in the source register."""
    candidates = [c.register_row() for c in result.cores]
    for row, core in zip(candidates, result.cores):
        if not independent_role_supported(core.kind, core.source_number, core.claim, core.evidence_ids,
                                          core.independent_role, records):
            row.update(decision="excluded", decision_reason="The paper did not establish this result's independent comparative or contribution role.")
    approved = approved_cores(candidates)
    if result.summary:
        return candidates, None, result.summary.model_dump()
    if not any(c["kind"] in {"current", "application"} for c in approved.values()):
        labels = "、".join(c.source_number or c.candidate_id for c in result.cores)
        return candidates, None, {"kind": "no_qualifying_result", "question": f"《{result.draft.title}》给出了哪些独立的主要成果？",
            "reason": f"候选 {labels} 可以从原文定位，但现有依据只支持其陈述、背景或证明用途，未能确立其独立贡献地位。",
            "evidence_ids": list(dict.fromkeys(i for core in result.cores for i in core.evidence_ids))}
    draft = project_cores(result.draft.model_dump(), approved)
    return candidates, AuthoritativeDraft.model_validate(draft, context={"approved_cores": approved}), None


def project_cores(original, approved):
    """Retain complete old groups and remap their coordinates without changing ownership."""
    draft = deepcopy(original)
    owners = draft["core_bindings"]
    kept_groups = [i for i in range(len(draft["groups"])) if
                   approved.get(owners[f"groups.{i}.theorem"], {}).get("kind") in {"current", "application"}]
    kept_history = [i for i in range(len(draft["history"])) if approved.get(owners[f"history.{i}"], {}).get("kind") == "historical"]
    def project(path):
        role, _, rest = path.partition(".")
        if role not in {"history", "groups"}:
            return path
        number, _, tail = rest.partition(".")
        kept = kept_groups if role == "groups" else kept_history
        if int(number) in kept:
            return f"{role}.{kept.index(int(number))}" + ("." + tail if tail else "")
        return None
    draft["history"] = [draft["history"][i] for i in kept_history]
    draft["groups"] = [draft["groups"][i] for i in kept_groups]
    for name in ("formal_sources", "core_bindings"):
        draft[name] = {mapped: value for key, value in draft[name].items() if (mapped := project(key))}
    draft["evidence_links"] = [{**row, "block_id": mapped} for row in draft["evidence_links"]
                               if (mapped := project(row["block_id"]))]
    if len(kept_groups) != len(original["groups"]) or len(kept_history) != len(original["history"]):
        _rebind_cross_references(original, draft, project)
    return draft


def _rebind_cross_references(original, draft, project):
    """Resolve references to our printed headings before any group is renumbered."""
    from .profile import renderer_context
    old = renderer_context(original)["visible_headings"]
    new = renderer_context(draft)["visible_headings"]
    mapping = {}
    source_links = {row["block_id"]: row["evidence_ids"] for row in original["evidence_links"]}
    removed_sources = {}
    for path, heading in old.items():
        successor = project(path)
        if successor in new:
            mapping[heading] = new[successor]
            continue
        formal = original["formal_sources"].get(path, {})
        removed_sources[heading] = source_links.get(path, [])
        cite = re.search(r"\\cite\[([^\]]+)\]", formal.get("source", ""))
        prior = formal.get("prior", "").split(r"\cite", 1)[0].strip(" ,;")
        mapping[heading] = ("the source paper’s " + cite[1]) if cite else ("the result of " + prior) if prior else None

    if not mapping:
        return
    # Skip genuine source citations and mathematical delimiters. Local theorem
    # numbers are integers; never rewrite the prefix of a source number like 1.2.
    reference = r"\b(Theorem|Definition)[ ~\t]+([A-C]|[1-9]\d*)\b(?!\.\d)"
    protected = r"\\cite(?:\[[^\]]*\])*\{[^}]*\}|\$\$.*?\$\$|\$[^$]*\$|\\\(.*?\\\)|\\\[.*?\\\]"
    pattern = re.compile(protected + "|" + reference, re.S)
    discarded = set()
    for row in draft["evidence_links"]:
        path = row["block_id"]
        old_text = block_text(draft, path)
        if not isinstance(old_text, str):
            continue
        def translate(match):
            if not match[1]:
                return match[0]
            if (re.search(r"\b(?:source|original(?: paper)?(?:’s|'s)?|cited paper(?:’s|'s)?)[~ ]+$",
                          old_text[:match.start()], re.I)
                or re.match(r"[~ ]+(?:of|in)[~ ]+(?:\\cite|\[\d|the (?:original|source))", old_text[match.end():], re.I)):
                return match[0]
            key = f"{match[1]} {match[2]}"
            if key not in mapping or mapping[key] == key:
                return match[0]
            if key in removed_sources:
                row["evidence_ids"] = list(dict.fromkeys([*row["evidence_ids"], *removed_sources[key]]))
            if mapping[key] is None:
                # An optional comparison can simply be omitted. A necessary
                # statement retains the formal source in its theorem heading.
                if path.endswith(".paragraph"):
                    discarded.add(path)
                return "the source result cited here"
            return mapping[key]
        text = pattern.sub(translate, old_text)
        parent, _, name = path.rpartition(".")
        container = block_text(draft, parent) if parent else draft
        container[int(name) if name.isdecimal() else name] = "" if path in discarded else text
    if discarded:
        draft["evidence_links"] = [r for r in draft["evidence_links"] if r["block_id"] not in discarded]


class NewStatement(BaseModel):
    text: str = Field(min_length=1)
    formal_source: FormalSource
    evidence_ids: list[str] = Field(min_length=1)


class NewCore(NewStatement):
    candidate_id: str
    definitions: list[NewStatement] = Field(default_factory=list, max_length=2)
    paragraph: str = ""
    paragraph_evidence_ids: list[str] = Field(default_factory=list)


class AddedCores(BaseModel):
    cores: list[NewCore]
    additional_references: list[Reference] = Field(default_factory=list)

    @model_validator(mode="after")
    def verify_admission(self, info: ValidationInfo):
        if info.context is None:
            return self
        expected = info.context["requested_cores"]
        ids = [r.candidate_id for r in self.cores]
        if set(ids) != set(expected) or len(ids) != len(expected):
            raise ValueError("Return precisely the newly approved cores")
        registered = {r["id"] for r in info.context["passages"]}
        for core in self.cores:
            requirement = expected[core.candidate_id]
            if not set(core.evidence_ids).intersection(requirement["evidence_ids"]):
                raise ValueError("Each new formal block must cite its approved primary source")
            if bool(core.paragraph) != bool(core.paragraph_evidence_ids):
                raise ValueError("A connecting paragraph must include its evidence IDs")
            if requirement["kind"] == "historical" and (core.definitions or core.paragraph):
                raise ValueError("Historical statements do not introduce a result group")
            ids = [*core.evidence_ids, *core.paragraph_evidence_ids, *(i for d in core.definitions for i in d.evidence_ids)]
            if not set(ids) <= registered:
                raise ValueError("An addition cited an unknown location")
        return self

    def merge(self, draft, approved):
        def link(path, row):
            draft["formal_sources"][path] = row.formal_source.model_dump()
            draft["evidence_links"].append({"block_id": path, "evidence_ids": row.evidence_ids})
        for row in self.cores:
            if approved[row.candidate_id]["kind"] == "historical":
                path = f'history.{len(draft["history"])}'
                draft["history"].append(row.text)
            else:
                i = len(draft["groups"])
                path = f"groups.{i}.theorem"
                draft["groups"].append({"theorem": row.text, "definitions": [d.text for d in row.definitions],
                                        "paragraph": row.paragraph, "paragraph_position": "after"})
                for j, definition in enumerate(row.definitions):
                    link(f"groups.{i}.definitions.{j}", definition)
                if row.paragraph:
                    draft["evidence_links"].append({"block_id": f"groups.{i}.paragraph", "evidence_ids": row.paragraph_evidence_ids})
            draft["core_bindings"][path] = row.candidate_id
            link(path, row)
        used = {r["key"] for r in draft["references"]}
        draft["references"].extend(r.model_dump() for r in self.additional_references if r.key not in used)
        return draft


def approved_cores(candidates):
    return {c["candidate_id"]: {"kind": c["kind"], "evidence_ids": c.get("selection_evidence_ids") or c["evidence_ids"]}
            for c in candidates if c.get("decision") == "admitted" and c["kind"] in {"current", "historical", "application"}}


def verify_bindings(draft, approved, registered):
    owners = draft["core_bindings"]
    links = {row["block_id"]: set(row["evidence_ids"]) for row in draft["evidence_links"]}
    if set(owners.values()) != set(approved) or len(owners) != len(approved):
        raise ValueError("Write one formal statement for every approved core ID, without introducing any other result")
    for field, core_id in owners.items():
        item = approved[core_id]
        if field.startswith("history.") != (item["kind"] == "historical"):
            raise ValueError(f"{field}: role differs from approved core {core_id}")
        if not links.get(field, set()).intersection(item["evidence_ids"]):
            raise ValueError(f"{field}: cite a source supporting approved core {core_id}")
    if any(not locations or not locations.issubset(registered) for locations in links.values()):
        raise ValueError("Use only registered source IDs for each draft block")


class DirectResult(BaseModel):
    """One selection-and-writing request. A negative selection has no TeX draft."""
    cores: list[DirectCore] = Field(default_factory=list)
    draft: AuthoritativeDraft | None = None
    summary: ScreeningNote | None = None
    source_checks: list[SourceCheck] = Field(default_factory=list, description="Only for a specific material uncertainty that the supplied paper cannot resolve as text")

    @model_validator(mode="after")
    def validate_decision(self, info: ValidationInfo):
        if self.source_checks:
            if self.cores or self.draft or self.summary or (info.context or {}).get("source_questions"):
                raise ValueError("Request source evidence before a decision; after the evidence response return the draft or the supported screening note")
            return self
        ids = [row.candidate_id for row in self.cores]
        if ids != [f"c{i}" for i in range(1, len(ids) + 1)]:
            raise ValueError("Number retained cores c1, c2, ... in register order")
        has_current = any(core.kind != "historical" for core in self.cores)
        if has_current != bool(self.draft) or bool(self.draft) == bool(self.summary):
            raise ValueError("A qualifying current result requires a draft; otherwise supply only a screening note")
        locations = {row["id"] for row in (info.context or {}).get("passages", [])}
        if self.summary and self.summary.kind == "no_qualifying_result" and not self.summary.evidence_ids:
            raise ValueError("Cite the examined source for a no-result finding")
        if info.context is not None:
            source_ids = [key for core in self.cores for key in core.evidence_ids]
            source_ids.extend(c.significance_source_id for c in self.cores if c.significance_source_id)
            if self.summary:
                source_ids.extend(self.summary.evidence_ids)
            if not set(source_ids) <= locations:
                raise ValueError("Unknown source ID in the direct screening result")
        if self.draft:
            draft = self.draft.model_dump()
            if info.context is None:
                locations = {i for row in draft["evidence_links"] for i in row["evidence_ids"]}
            verify_bindings(draft, approved_cores([c.register_row() for c in self.cores]), locations)
        return self


def fits_direct(catalog, instructions, packet, settings):
    """Conservative byte upper bound including schema, message framing and output.

    Character-count truncation can accidentally discard an appendix. When the
    complete input cannot fit, use the segmented path with the original snapshot.
    """
    if any(not row["text"].strip() or row.get("transcribed_from_image") for row in catalog):
        return False
    if any(row.get("page") and ("\ufffd" in row["text"] or re.search(r"\(cid:\d+\)", row["text"]))
           for row in catalog):
        return False
    # A conservative envelope for supported services, also accept a smaller
    # explicitly declared context for an OpenAI-compatible local deployment.
    input_bytes = len((instructions + json_text(DirectResult.model_json_schema()) + json_text(packet)).encode("utf-8"))
    window = int(settings.get("context_window") or 65536)
    return input_bytes + 8192 + 1024 <= window


class FindingUpdate(BaseModel):
    candidate_index: int = Field(ge=0, description="Zero-based row of initial candidates")
    candidate: Candidate


class FindingsDelta(BaseModel):
    updates: list[FindingUpdate]
    additions: list[Candidate] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def known_indices(self, info: ValidationInfo):
        ids = [row.candidate_index for row in self.updates]
        if len(ids) != len(set(ids)):
            raise ValueError("Each corrected candidate may appear once")
        if info.context is not None and any(i not in {c["candidate_index"] for c in info.context["affected_candidates"]} for i in ids):
            raise ValueError("Only update the affected candidate indices")
        return self


class EvidenceReview(BaseModel):
    """Read an identified source first, or give a complete editorial verdict."""
    retry_with_previous: ClassVar[bool] = True
    verdict: AdmissionReview | None = None
    source_checks: list[SourceCheck] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def accept_recognizable_envelope(cls, value):
        if not isinstance(value, dict):
            return value
        if value.get("source_checks"):
            # A real source question makes a simultaneous verdict provisional.
            # Read the requested evidence, then obtain a new, independent verdict.
            return {"source_checks": value["source_checks"]}
        if value.get("verdict") is None and {"route", "issues", "evidence_ids"} <= value.keys():
            return {"verdict": {key: value[key] for key in ("route", "issues", "evidence_ids")}}
        return value

    @model_validator(mode="after")
    def one_action(self, info: ValidationInfo):
        if bool(self.verdict) == bool(self.source_checks):
            raise ValueError('Return {"verdict":{"route":"pass","issues":[],"evidence_ids":[]}} only for a clean draft; '
                             'otherwise put the complete repair verdict under verdict, or put concrete questions under source_checks.')
        if info.context and self.source_checks:
            known = {row["id"] for key in ("passages", "source_locations") for row in info.context.get(key, [])}
            requested = {q.source_id for q in self.source_checks}
            if info.context.get("source_locations"):
                requested.update(q.target_id for q in self.source_checks if q.target_id)
            if known and requested - known:
                raise ValueError(f"Use a registered source_id/target_id; unknown locations: {sorted(requested - known)}")
        return self


def selection_change(defect, draft):
    """Only the selector may introduce, remove or reclassify a formal result."""
    field = defect["draft_field"]
    operation = defect["operation"]
    if field in {"groups", "history"} or re.fullmatch(r"groups\.\d+", field):
        return True
    formal = bool(re.fullmatch(r"history\.\d+|groups\.\d+\.theorem", field))
    if not formal:
        return False
    return (field not in draft["core_bindings"] or
            (operation == "insert" and not defect["current_excerpt"].strip()) or
            (operation == "remove" and defect["current_excerpt"].strip() == block_text(draft, field).strip()))


def block_text(draft, field):
    value = draft
    for token in field.split("."):
        value = value[int(token) if token.isdecimal() else token]
    return value
