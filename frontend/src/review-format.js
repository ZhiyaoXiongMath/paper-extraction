const fieldNames = {
  paragraph: "引言开场段",
  history: "历史定理",
  shared_definitions: "共用定义",
  groups: "本文定理组",
  references: "参考文献",
  title: "论文标题",
  authors: "作者",
};

export function reviewField(field = "") {
  if (Object.hasOwn(fieldNames, field)) return fieldNames[field];
  let match = /^history\.(\d+)$/.exec(field);
  if (match && Number(match[1]) < 26)
    return "历史定理 " + String.fromCharCode(65 + Number(match[1]));
  match = /^shared_definitions\.(\d+)$/.exec(field);
  if (match) return "共用定义 " + (Number(match[1]) + 1);
  match = /^groups\.(\d+)(?:\.(theorem|paragraph|definitions)(?:\.(\d+))?)?$/.exec(field);
  if (match) {
    const n = Number(match[1]) + 1;
    if (match[2] === "theorem") return "本文定理 " + n;
    if (match[2] === "paragraph") return "定理 " + n + " 的说明段";
    if (match[2] === "definitions")
      return "定理 " + n + (match[3] === undefined ? " 的定义" : " 的专用定义 " + (Number(match[3]) + 1));
    return "定理组 " + n;
  }
  match = /^references\.(\d+)(?:\.(key|text))?$/.exec(field);
  if (match) return "参考文献 " + (Number(match[1]) + 1);
  return field || "草稿";
}

export function evidenceLabel(id) {
  const page = /^p(\d+)-\d+$/.exec(id);
  return page ? "原文第 " + page[1] + " 页 · " + id : "原文依据 " + id;
}

export function defectKind(kind) {
  return {
    incorrect_claim: "数学陈述",
    missing_required_content: "缺少内容",
    incorrect_attribution_or_locator: "归属或引用位置",
    improper_selection: "成果取舍",
  }[kind] || kind || "需修订";
}

export function editLabel(operation) {
  return { replace: "建议改写", insert: "建议补入", remove: "建议删除", recover: "建议回查" }[operation] || "建议修改";
}

export function readableReviewText(raw, references = []) {
  const expanded = raw.replace(/\\begin\s*\{(historical|theorem|definition|introparagraph)\}[\s\S]*?\\end\s*\{\1\}/g,
    (fragment) => {
      const { text, caption } = readerFragment(fragment, "", references);
      return caption ? text + "〔" + caption + "〕" : text;
    });
  return readerFragment(expanded, "", references).text;
}

function legacyDefect(issue) {
  // Parse only the envelope recorded by the old reviewer. Free-form messages stay intact.
  const envelope = /^([\w.]+): ([\s\S]+) \(draft: ([\s\S]*?); original ([pt]\d+-\d+): ([\s\S]+)\)$/.exec(issue);
  if (!envelope || reviewField(envelope[1]) === envelope[1]) return null;
  const [, draft_field, change, current, evidence_id, source_excerpt] = envelope;
  const wording = /^([\s\S]+) \[(replace|insert): ([\s\S]+)\]$/.exec(change);
  return { draft_field, required_change: wording?.[1] || change, current_excerpt: current === "[missing]" ? "" : current,
    evidence_id, source_excerpt, operation: wording?.[2], replacement_text: wording?.[3] };
}

export function reviewDetails(review) {
  const recorded = (review.defects || []).filter((item) =>
    item && typeof item.required_change === "string" && item.required_change.trim());
  const defects = [...recorded], plain = [];
  for (const issue of review.issues || []) {
    if (typeof issue !== "string" || !issue.trim()) continue;
    // Older and current API records both retain a formatted copy of each defect.
    // Keep unmatched diagnostics, including two different edits of the same block.
    if (recorded.some((d) => {
      const core = d.draft_field + ": " + d.required_change;
      const action = d.replacement_text ? "[" + d.operation + ": " + d.replacement_text + "] " : "";
      const source = "(draft: " + (d.current_excerpt || "[missing]") +
        "; original " + d.evidence_id + ": " + d.source_excerpt + ")";
      return issue === core || issue === core + " " + source || issue === core + " " + action + source;
    })) continue;
    const parsed = legacyDefect(issue);
    if (parsed) defects.push(parsed);
    else plain.push(issue);
  }
  return { defects, plain };
}
import { readerFragment } from "./reader.js";
