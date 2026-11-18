"""Prepare and record an operator-reviewed local repair of a saved engineering failure."""
import hashlib
import json
from pathlib import Path

from .draft_artifacts import draft_digest
from .models import DraftPatch
from .pdfops import page_count, render_page
from .profile import ASSETS, build_case, render_tex, v13_check
from .profile_variants import companion_text, release_for_draft
from .sources import SourceIndex
from .store import NAMES, now, sha
from .writer_patch import apply_patch, repair_targets


def _load(path):
    return json.loads(Path(path).read_text("utf-8-sig"))


def _save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _revision(store, paper, basis, patch_data):
    previous = paper["result"]
    reviews = previous.get("reviews", [])
    if (paper["status"] != "failed" or paper["attempt"] != basis["attempt"] or not previous.get("draft")
            or draft_digest(previous["draft"]) != basis["draft_sha256"] or not reviews):
        raise ValueError("论文当前尝试与指定的失败草稿不匹配")
    last = reviews[-1]
    round_no = basis["round"]
    content = next((r for r in reversed(reviews[:-1]) if r["stage"] == "review" and r["attempt"] == round_no), {})
    if (last["stage"] != "compile" or last["passed"] is not False or last["attempt"] != round_no
            or content.get("passed") is not True or content.get("route") != "pass" or content.get("issues")):
        raise ValueError("指定轮次须有已通过的内容核验和未通过的工程检查")
    work = store.work(paper)
    source = Path(paper["source"])
    root = Path(paper["root"]) if paper["root"] else None
    records = {r["id"]: r for r in SourceIndex(source, root, work).catalog()}
    targets = repair_targets(previous["draft"], last, records.keys())
    if not targets:
        raise ValueError("工程诊断未能绑定到此次草稿的字段")
    packet = {"passages": list(records.values()), "repair_targets": targets}
    patch = DraftPatch.model_validate(patch_data, context=packet)
    by_block = {row.block_id: row for row in patch.replacements}
    changes = []
    for target in targets:
        block = target["block_id"]
        row = by_block[block]
        ids = target["bound_evidence_ids"]
        if row.evidence_ids != ids:
            raise ValueError("此局部排版修订必须保留原字段的证据绑定")
        changes.append({"block_id": block, "before": target["current_text"], "after": row.text,
                        "evidence_ids": ids, "diagnostics": target["diagnostics"],
                        "source_passages": [{"id": i, "label": records[i]["label"], "text": records[i]["text"]} for i in ids]})
    return apply_patch(previous["draft"], patch, packet, records), changes, content


def _verified_build(case, paper, draft):
    status = _load(case / "build-status.json")
    binding = _load(case / "input-manifest.json")
    source = Path(paper["source"])
    root = Path(paper["root"]) if paper["root"] else None
    hashes = {name: sha(case / name) for name in NAMES}
    release = release_for_draft(draft)
    expected_preamble = (sha(ASSETS / "preamble-v13.tex") if release == v13_check.RELEASE else
                         hashlib.sha256(companion_text(ASSETS, "preamble-v13.tex", release).encode("utf-8")).hexdigest())
    if (status.get("engineering_accepted") is not True or status.get("input_binding_checked") is not True
            or status.get("returncodes") != [0, 0] or status.get("failed_checks") or status.get("blocking_diagnostics")
            or status.get("input_binding", {}).get("manifest_sha256") != sha(case / "input-manifest.json")
            or binding.get("source_sha256") != sha(source)
            or binding.get("snapshot", {}).get("sha256") != v13_check.source_snapshot(source, root)["sha256"]
            or status.get("tex_sha256") != hashes["introduction.tex"]
            or status.get("pdf_sha256") != hashes["introduction.pdf"]
            or binding.get("release") != release or hashes["preamble-v13.tex"] != expected_preamble
            or (case / "introduction.tex").read_text("utf8") != render_tex(draft)):
        raise ValueError("本轮稿件、来源绑定或双遍 v13 工程检查与本次修订不一致")
    count = page_count(case / "introduction.pdf")
    if count < 1 or count != status.get("pdf_pages"):
        raise ValueError("待审核的 PDF 页数与编译回执不匹配")
    return status, hashes, count


def prepare_revision(store, pid, attempt, round_no, original_digest, patch_path):
    paper = store.paper(pid)
    basis = {"paper_id": pid, "attempt": attempt, "round": round_no, "draft_sha256": original_digest}
    patch = _load(patch_path)
    draft, changes, content = _revision(store, paper, basis, patch)
    work = store.work(paper)
    case = build_case(work / "reviewed-revisions", Path(paper["source"]), Path(paper["root"]) if paper["root"] else None, draft)
    status, hashes, count = _verified_build(case, paper, draft)
    review_dir = case / "inspection"
    review_dir.mkdir(exist_ok=True)
    pages = []
    for page in range(1, count + 1):
        file = review_dir / f"page-{page}.png"
        file.write_bytes(render_page(case / "introduction.pdf", page))
        pages.append({"page": page, "sha256": sha(file)})
    _save(case / "revision.json", {"basis": basis, "patch": patch, "changes": changes,
          "content_review": content, "draft_sha256": draft_digest(draft), "hashes": hashes,
          "run_id": status["run_id"], "pages": pages})
    assessment = {"operator": "", "changes": [{"block_id": c["block_id"], "evidence_ids": c["evidence_ids"],
                    "passed": None, "observation": ""} for c in changes],
                  "pages": [{**p, "passed": None, "observation": ""} for p in pages]}
    _save(review_dir / "audit-template.json", assessment)
    return {"case": str(case), "draft_round": round_no + 1, "audit_template": str(review_dir / "audit-template.json"),
            "page_images": [str(review_dir / f"page-{p['page']}.png") for p in pages]}


def _verified_rows(actual, expected, fields):
    if not isinstance(actual, list) or len(actual) != len(expected):
        raise ValueError("每一处局部修改和每一张渲染页都须记录实际核对意见")
    for entry, original in zip(actual, expected):
        if (not isinstance(entry, dict) or any(entry.get(k) != original[k] for k in fields)
                or entry.get("passed") is not True or not isinstance(entry.get("observation"), str)
                or not entry["observation"].strip()):
            raise ValueError("审核意见、字段、证据位置或逐页图像哈希不匹配")


def accept_revision(store, pid, case, audit_path):
    paper = store.paper(pid)
    case = Path(case).resolve()
    if not case.is_relative_to((store.work(paper) / "reviewed-revisions").resolve()):
        raise ValueError("修订目录不属于此论文工作尝试")
    proposal = _load(case / "revision.json")
    audit = _load(audit_path)
    basis = proposal["basis"]
    if basis["paper_id"] != pid:
        raise ValueError("核对记录不属于该论文")
    if (paper["status"] == "completed" and paper["result"].get("revision_basis") == basis
            and paper["result"].get("result_dir") == str(case) and paper["result"].get("operator_audit") == audit):
        return {"paper_id": pid, "attempt": paper["attempt"], "round": paper["result"]["draft_round"], "status": "completed"}
    draft, changes, content = _revision(store, paper, basis, proposal["patch"])
    status, hashes, count = _verified_build(case, paper, draft)
    if (proposal["draft_sha256"] != draft_digest(draft) or proposal["hashes"] != hashes
            or proposal["run_id"] != status["run_id"] or proposal["content_review"] != content or proposal["changes"] != changes):
        raise ValueError("已准备的稿件或核对依据发生变化")
    if not isinstance(audit.get("operator"), str) or not audit["operator"].strip():
        raise ValueError("须填写进行实际核对的操作者")
    pages = []
    for n in range(1, count + 1):
        rendered = hashlib.sha256(render_page(case / "introduction.pdf", n)).hexdigest()
        if sha(case / "inspection" / f"page-{n}.png") != rendered:
            raise ValueError("待审页图与当前 PDF 不一致")
        pages.append({"page": n, "sha256": rendered})
    if pages != proposal["pages"]:
        raise ValueError("逐页图像与准备回执不一致")
    _verified_rows(audit.get("changes"), changes, ("block_id", "evidence_ids"))
    _verified_rows(audit.get("pages"), pages, ("page", "sha256"))
    round_no = basis["round"] + 1
    base_review = {"attempt": round_no, "passed": True, "issues": [], "operator": audit["operator"]}
    reviews = [*paper["result"]["reviews"],
               {**base_review, "stage": "review", "route": "pass", "method": "operator_delta_review",
                "basis_round": basis["round"], "changes": audit["changes"]},
               {"stage": "compile", "attempt": round_no, "passed": True, "issues": [], "method": "v13_checker",
                "run_id": status["run_id"], "returncodes": status["returncodes"], "checks": status["checks"]},
               *[{**base_review, "stage": "visual", "method": "operator_rendered_page", **p} for p in audit["pages"]]]
    result = {**paper["result"], "status": "completed", "reason": "", "resumable": False, "draft_round": round_no,
              "profile_release": status["release"],
              "draft": draft, "links": draft["evidence_links"], "evidence_links": draft["evidence_links"],
              "result_dir": str(case), "hashes": hashes, "reviews": reviews, "revision_basis": basis,
              "operator_audit": audit, "reviewed_at": now()}
    store.publish_reviewed(paper, result)
    return {"paper_id": pid, "attempt": paper["attempt"] + 1, "round": round_no, "status": "completed"}
