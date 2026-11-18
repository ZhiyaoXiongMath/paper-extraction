import { tone } from "./format.js";
import { reviewDetails, reviewField } from "./review-format.js";

export function paperDownloads(paper, prefix) {
  if (tone(paper.status) === "good") {
    if (paper.delivery?.kind === "summary") {
      return { label: "已交付 · 筛选说明", markdown: prefix + "/artifacts/summary.md" };
    }
    let label = "已通过 · 成品";
    if (paper.delivery?.status === "reviewed") label = "已交付 · 已通过检查";
    if (paper.delivery?.status === "delivered") {
      const round = paper.delivery.draft_round;
      label = round ? `已交付 · 第 ${round} 稿` : "已交付 · 成品";
    }
    return {
      label, tex: prefix + "/artifacts/introduction.tex",
      pdf: prefix + "/artifacts/introduction.pdf", restored: false, recompiled: false, missingPdf: "",
    };
  }
  // The server binds this flag to the saved draft digest and the active job attempt.
  const saved = (paper.draft_artifacts || []).find(
    entry => entry.current && entry.attempt === paper.attempt,
  );
  const hasDraft = !!saved || !!paper.draft?.paragraph || !!paper.draft?.groups?.length;
  const round = saved?.round || (hasDraft ? paper.reviews?.at(-1)?.attempt : undefined);
  const state = paper.status === "cancelled" ? "已终止" : paper.status === "paused" ? "已暂停" :
    tone(paper.status) === "bad" ? "未通过" : "处理中";
  const version = round ? "第 " + round + " 轮草稿" : (hasDraft ? "本轮草稿" : "本轮尚无草稿");
  return {
    label: state + " · " + version,
    tex: saved?.files?.tex?.url, pdf: saved?.files?.pdf?.url,
    restored: !!saved?.restored, recompiled: !!saved?.recompiled,
    missingPdf: saved?.files?.pdf?.url ? "" : "本轮未生成 PDF",
  };
}

function layoutDetails(raw) {
  const result = [];
  for (const m of raw.matchAll(/\b(introparagraph|historical|theorem|definition)_(\d+)_word_limit\s*:\s*(\d+)\s*>\s*(\d+)/gi)) {
    const [, kind, n, count, cap] = m;
    const name = kind === "introparagraph" && n === "1" ? "开场段" :
      "第 " + n + " 个" + ({ introparagraph: "引言段", historical: "历史定理块", theorem: "本文定理块", definition: "定义块" })[kind];
    result.push(name + " " + count + " 词，本次上限 " + cap + " 词");
  }
  for (const m of raw.matchAll(/Overfull\s+\\hbox\s*\(([\d.]+)\s*pt\s+too wide\)\s+in paragraph at lines?\s+(\d+)(?:--(\d+))?/gi)) {
    const [, width, from, to] = m;
    result.push("第 " + (to && to !== from ? from + "–" + to : from) + " 行超宽 " + width + " pt");
  }
  return result;
}

export function visibleFailureReasons(paper) {
  if (tone(paper.status) !== "bad") return [];
  const reason = String(paper.reason || "").trim();
  const latest = paper.reviews?.at(-1);
  const review = latest && latest.passed === false ? latest : undefined;
  const unique = values => [...new Set(values.filter(Boolean))];
  const layout = unique(layoutDetails([reason, ...(review?.issues || [])].join("\n")));
  if (layout.length) return layout;
  const detail = review && reviewDetails(review);
  const defects = detail?.defects.map(d =>
    d.required_change ? [d.draft_field && reviewField(d.draft_field), d.required_change].filter(Boolean).join(" · ") : "",
  );
  if (defects?.some(Boolean)) return unique(defects);
  if (detail?.plain.length) return unique(detail.plain);
  return reason ? [reason] : [];
}
