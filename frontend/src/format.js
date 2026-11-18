const statuses = {
  queued: ["排队中", "neutral"],
  pending: ["排队中", "neutral"],
  running: ["运行中", "working"],
  processing: ["运行中", "working"],
  passed: ["已通过", "good"],
  accepted: ["已通过", "good"],
  succeeded: ["已通过", "good"],
  success: ["已通过", "good"],
  completed: ["已通过", "good"],
  failed: ["未通过", "bad"],
  rejected: ["未通过", "bad"],
  error: ["运行失败", "bad"],
  partial: ["证据不足", "bad"],
  blocked: ["已阻断", "bad"],
  stopped: ["已停止", "bad"],
  cancelled: ["已取消", "bad"],
  canceled: ["已取消", "bad"],
};
const stages = {
  queued: "等待运行",
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
export const terminal = (status) => ["bad", "good"].includes(tone(status));
export const stageName = (stage) => stages[stage] ?? stage ?? "—";
export function metrics(batches) {
  const result = { total: 0, working: 0, good: 0, bad: 0 };
  for (const batch of batches) {
    result.total += Number(batch.total || 0);
    for (const [status, count] of Object.entries(batch.counts || {})) {
      const group = tone(status);
      result[group === "neutral" ? "working" : group] += Number(count || 0);
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
    (g.definitions || []).forEach((s, j) =>
      add(base + "definitions." + j, "Definition " + ++n, s),
    );
    add(base + "theorem", "Theorem " + (i + 1), g.theorem);
    add(base + "paragraph", "", g.paragraph);
  });
  return blocks;
}
