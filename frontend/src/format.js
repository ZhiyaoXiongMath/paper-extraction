const statuses = {
  queued: ["排队", "neutral"],
  pending: ["排队中", "neutral"],
  running: ["处理中", "working"],
  pausing: ["正在暂停", "working"],
  paused: ["已暂停", "neutral"],
  processing: ["运行中", "working"],
  passed: ["已通过", "good"],
  accepted: ["已通过", "good"],
  succeeded: ["已通过", "good"],
  success: ["已通过", "good"],
  completed: ["已完成", "good"],
  failed: ["失败", "bad"],
  rejected: ["未通过", "bad"],
  error: ["运行失败", "bad"],
  partial: ["证据不足", "bad"],
  blocked: ["已阻断", "bad"],
  stopped: ["已停止", "bad"],
  cancelled: ["已终止", "bad"],
  canceled: ["已取消", "bad"],
};
const stages = {
  queued: "等待运行",
  pausing: "正在暂停",
  paused: "已暂停",
  cancelled: "已终止",
  start: "启动任务",
  ingest: "建立来源索引",
  index: "建立来源索引",
  bind: "绑定来源快照",
  extract: "提取候选",
  research: "核查原文",
  plan: "筛选贡献",
  select: "筛选贡献",
  evidence: "查证原文",
  failed: "检查结束",
  draft: "撰写引言",
  review: "核查数学内容",
  compile: "编译与版式检查",
  visual: "逐页视觉检查",
  repair: "定向修复",
  publish: "打包成品",
  finished: "已结束",
  completed: "已完成",
};
export const label = (status) => statuses[status]?.[0] ?? status ?? "—";
export const tone = (status) => statuses[status]?.[1] ?? "neutral";
export const paperTone = (paper) =>
  paper.delivery?.checks && paper.delivery.status === "delivered" ? "neutral" : tone(paper.status);
export const terminal = (status) => ["bad", "good"].includes(tone(status));
export const stageName = (stage) => stages[stage] ?? stage ?? "—";
export function deliveryLabel(delivery) {
  if (!delivery?.checks) return "";
  const round = delivery.draft_round ? `第 ${delivery.draft_round} 稿` : "交付稿";
  if (delivery.status === "reviewed") return `已交付 · ${round}全检查通过`;
  return "已交付";
}

export function batchPaperLabel(paper) {
  if (paper.status === "completed" && paperTone(paper) === "neutral") return "已交付";
  return paperOutcome(paper).title;
}

export function canDeleteBatch({ counts, total }, allowPaused = true) {
  const finished = (counts.completed ?? 0) + (counts.failed ?? 0) + (counts.cancelled ?? 0);
  return finished + (allowPaused ? counts.paused ?? 0 : 0) === total;
}
export function paperOutcome(paper) {
  const hasDraft = readBlocks(paper.draft || {}).some((block) =>
    block.text?.trim(),
  );
  const failed = tone(paper.status) === "bad";
  const recentReview = paper.reviews?.at(-1);
  const sourceTab = paper.candidates?.length ? "candidates" : "log";
  const base = {
    hasDraft,
    title: label(paper.status),
    summary: "",
    initialTab: hasDraft
      ? failed && recentReview
        ? "checks"
        : "reader"
      : sourceTab,
  };
  if (paper.status === "cancelled") return { ...base, title: "已终止" };
  if (paper.status === "completed" && paper.delivery) {
    if (paper.delivery.kind === "summary") {
      return { ...base, title: "已交付 · 筛选说明", initialTab: "summary", summary: paper.delivery.note };
    }
    const reviewed = paper.delivery.status === "reviewed";
    return { ...base, title: deliveryLabel(paper.delivery) || (reviewed ? "已交付 · 已通过检查" : "已交付") };
  }
  if (!failed) return base;
  const lastStage = [...(paper.events || [])]
    .reverse()
    .find((e) => !["queued", "completed", "failed"].includes(e.stage))?.stage;
  if (!hasDraft) {
    const legacy =
      /the draft entries are empty|no introduction text drafted/i.test(
        paper.reason || "",
      );
    const stopped = lastStage
      ? "最近记录的环节为「" + stageName(lastStage) + "」。"
      : "";
    return {
      ...base,
      title: "未生成成品",
      summary: legacy
        ? "初稿尚未生成，当前结果不能判断论文是否合格。"
        : stopped + "本次任务尚未生成引言初稿。",
    };
  }
  const stage = [
    "review",
    "compile",
    "visual",
    "publish",
    "draft",
    "extract",
  ].includes(lastStage)
    ? lastStage
    : recentReview?.stage;
  const stages = {
    review: "数学内容复核",
    compile: "编译与版式检查",
    visual: "逐页视觉检查",
    publish: "成品打包",
    draft: "撰写引言",
    extract: "提取候选",
  };
  return {
    ...base,
    title: stage
      ? (stages[stage] || stageName(stage)) + "未完成"
      : "未生成成品",
    summary: stage
      ? "本次任务已生成初稿，停止在「" +
        (stages[stage] || stageName(stage)) +
        "」环节。"
      : "已有引言初稿，尚未生成通过检查的成品。",
  };
}

export function metrics(batches) {
  const result = { total: 0, queued: 0, running: 0, completed: 0, failed: 0 };
  const groups = {
    neutral: "queued",
    working: "running",
    good: "completed",
    bad: "failed",
  };
  for (const batch of batches) {
    result.total += Number(batch.total || 0);
    for (const [status, count] of Object.entries(batch.counts || {})) {
      if (status === "paused" || status === "cancelled") {
        result[status] = (result[status] || 0) + Number(count || 0);
        continue;
      }
      if (Object.hasOwn(statuses, status))
        result[groups[tone(status)]] += Number(count || 0);
    }
  }
  return result;
}
export const formatDate = (v) => {
  if (!v) return "—";
  const d = new Date(v);
  return Number.isNaN(d.getTime())
    ? String(v)
    : new Intl.DateTimeFormat("zh-CN", {
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
      }).format(d);
};
export function eventSummary(event) {
  const d = event.data || {};
  const tools = {
    search_source: "检索当前论文",
    read_span: "读取原文片段",
    read_page_image: "检查原页图像",
  };
  const parts = [stageName(event.stage)];
  if (d.tool && tools[d.tool]) parts.push(tools[d.tool]);
  if (d.chunk && d.total) parts.push("第 " + d.chunk + " / " + d.total + " 段");
  if (d.page) parts.push("第 " + d.page + " 页");
  if (d.attempt) parts.push("第 " + d.attempt + " 轮");
  if (event.time) parts.push(formatDate(event.time));
  return parts.join(" · ");
}
export const shortId = (s) => (s || "").replace(/^batch[-_]/, "").slice(0, 8);
export const kindName = (v) =>
  ({
    historical: "历史结果",
    current: "本文贡献",
    definition: "定义",
    application: "应用",
    setting: "背景",
  })[v] ?? v;
export function readBlocks(draft = {}) {
  const blocks = [];
  function add(id, heading, text) {
    if (text) blocks.push({ id, heading, text });
  }
  add("paragraph", "Setting", draft.paragraph);
  (draft.history || []).forEach((s, i) =>
    add("history." + i, "Theorem " + String.fromCharCode(65 + i), s),
  );
  let n = 0;
  (draft.shared_definitions || []).forEach((s, i) =>
    add("shared_definitions." + i, "Definition " + ++n, s),
  );
  (draft.groups || []).forEach((g, i) => {
    const base = "groups." + i + ".";
    if (g.paragraph_position === "before") add(base + "paragraph", "", g.paragraph);
    (g.definitions || []).forEach((s, j) =>
      add(base + "definitions." + j, "Definition " + ++n, s),
    );
    add(base + "theorem", "Theorem " + (i + 1), g.theorem);
    if (g.paragraph_position !== "before") add(base + "paragraph", "", g.paragraph);
  });
  return blocks;
}
