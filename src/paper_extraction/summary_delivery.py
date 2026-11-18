"""Publish a source-bound Markdown explanation when no current theorem can be delivered."""

import hashlib
from pathlib import Path


SUMMARY_NAME = "summary.md"
FORMAL_NAMES = ("introduction.tex", "introduction.pdf", "preamble-v13.tex")
OUTCOMES = {
    "no_qualifying_result": "已核对原文，未选出值得单列的当前定理。",
    "insufficient_evidence": "原文证据不足，尚无法核实可单列的当前定理。",
}


def artifact_names(delivery: dict | None) -> tuple[str, ...]:
    """Old receipts have no delivery kind and contain the formal TeX bundle."""
    if delivery and delivery.get("kind") == "summary":
        return (SUMMARY_NAME,)
    return FORMAL_NAMES


def write_summary(work: Path, source: Path, *, kind: str, question: str, reason: str,
                  evidence_ids: list[str], catalog: list[dict]) -> tuple[Path, dict]:
    """Use only registered locations. Store the note under its content hash for resumable publication."""
    if kind not in OUTCOMES or not reason.strip() or not question.strip():
        raise ValueError("筛选说明需要明确的研究问题、结论类型和具体理由。")
    labels = {row["id"]: row.get("label") or row["id"] for row in catalog}
    if any(loc not in labels for loc in evidence_ids):
        raise ValueError("筛选说明引用了未登记的原文位置。")
    if kind == "no_qualifying_result" and not evidence_ids:
        raise ValueError("已核实的筛选结论需要注明原文依据。")
    evidence = [{"id": loc, "label": labels[loc]} for loc in dict.fromkeys(evidence_ids)]
    summary = {"question": question.strip(), "reason": reason.strip(), "evidence": evidence}
    note = OUTCOMES[kind]
    lines = ["# 论文筛选说明", "", f"论文：{source.name}", "", "## 研究问题", "",
             summary["question"], "", "## 筛选结论", "", note, "", summary["reason"], "", "## 原文依据", ""]
    lines.extend(f"- {item['label']}（{item['id']}）" for item in evidence)
    if not evidence:
        lines.append("未能取得可核实的原文段落。")
    content = ("\n".join(lines) + "\n").encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    directory = work / ("summary-" + digest[:20])
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / SUMMARY_NAME
    if not destination.exists() or destination.read_bytes() != content:
        pending = destination.with_suffix(".tmp")
        pending.write_bytes(content)
        pending.replace(destination)
    return directory, {"status": "delivered", "kind": "summary", "outcome": kind, "draft_round": 0,
                       "note": note, "summary": summary}
