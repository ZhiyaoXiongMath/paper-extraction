<script setup lang="ts">
import { ref, watch } from "vue";
import { errorMessage, request } from "./api";
import { stageName } from "./format.js";

interface Counts {
  model_calls?: number;
  http_requests?: number;
  http_requests_unknown_calls?: number;
  input_tokens?: number;
  input_tokens_unknown_calls?: number;
  output_tokens?: number;
  output_tokens_unknown_calls?: number;
  cache_hits?: number;
  schema_retry_calls?: number;
}
interface UsageGroup extends Counts {
  stage: string;
  operation: string;
  model: string;
  attempt: number;
  draft_round: number;
}
interface Usage { recorded: boolean; totals: Counts; groups: UsageGroup[] }

const props = defineProps<{ paperId: string; eventCount: number; status: string }>();
const usage = ref<Usage | null>(null);
const error = ref("");
const labels: Record<string, string> = {
  direct: "整篇直写", direct_evidence: "定向查证", direct_finish: "补证后整篇收尾",
  extract: "分段提取", write_draft: "写稿", select: "整篇筛选",
  review: "内容复核", review_with_fallback: "内容复核",
  visual: "成品逐页检查", source_image_lookup: "定向查原页", prerequisites: "定向补查",
};
const operations: Record<string, string> = {
  evidence_decision: "查证决策", source_transcription: "原页转写",
  Findings: "候选提取", TextFindings: "候选提取", WholePaperSelection: "贡献取舍",
  DirectResult: "选取贡献并生成初稿", SegmentLeads: "分段候选", ReadingAnswers: "核实具体疑点",
  SourceSelection: "按原文筛选", EvidenceReview: "核对原文", AuthoritativeDraft: "生成草稿",
  AddedCores: "补充候选",
  Draft: "生成草稿", CurrentDraft: "生成草稿", DraftPatch: "局部修订",
  AdmissionReview: "陈述核对", Review: "页图复核", page_review: "页图复核",
  SelectionCorrection: "筛选修正",
};
function quantity(counts: Counts, key: "input_tokens" | "output_tokens" | "http_requests") {
  const count = counts[key];
  const missing = counts[`${key}_unknown_calls`];
  if (missing && count == null) return "未知";
  return (count ?? 0).toLocaleString() + (missing ? " + 未知" : "");
}
watch(() => [props.paperId, props.eventCount, props.status], async (_, __, onCleanup) => {
  const controller = new AbortController();
  onCleanup(() => controller.abort());
  try {
    const result = await request<Usage>("/papers/" + encodeURIComponent(props.paperId) + "/usage",
                                        { signal: controller.signal });
    if (!controller.signal.aborted) {
      usage.value = result;
      error.value = "";
    }
  } catch (e) {
    if (!controller.signal.aborted) error.value = errorMessage(e);
  }
}, { immediate: true });
</script>

<template>
  <details class="usage" v-if="usage?.recorded">
    <summary>
      <strong>模型用量</strong> · 调用 {{ usage.totals.model_calls ?? 0 }} 次 ·
      输入 {{ quantity(usage.totals, "input_tokens") }} / 输出 {{ quantity(usage.totals, "output_tokens") }} tokens
    </summary>
    <p>HTTP 请求 {{ quantity(usage.totals, "http_requests") }} 次 ·
      结构化输出重试 {{ usage.totals.schema_retry_calls ?? 0 }} 次 · 本地缓存命中 {{ usage.totals.cache_hits ?? 0 }} 次</p>
    <div class="table-wrap">
      <table>
        <thead><tr><th>尝试 / 稿次</th><th>阶段</th><th>模型</th><th>调用</th><th>输入 tokens</th><th>输出 tokens</th></tr></thead>
        <tbody><tr v-for="(row, index) in usage.groups" :key="index">
          <td>{{ row.attempt }} / {{ row.draft_round || "首稿前" }}</td>
          <td>{{ labels[row.stage] || stageName(row.stage) }} · {{ operations[row.operation] || row.operation }}</td>
          <td>{{ row.model }}</td><td>{{ row.model_calls ?? 0 }}</td>
          <td>{{ quantity(row, "input_tokens") }}</td><td>{{ quantity(row, "output_tokens") }}</td>
        </tr></tbody>
      </table>
    </div>
  </details>
  <p class="usage" role="status" v-if="error">用量记录读取失败。{{ error }}</p>
</template>

<style scoped>
.usage { margin: 0 0 1rem; padding: .75rem 1rem; border: 1px solid var(--line); border-radius: 8px; }
summary { cursor: pointer; }
p { color: var(--muted); font-size: .85rem; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: .85rem; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: .4rem .7rem .4rem 0; white-space: nowrap; border-bottom: 1px solid var(--line); }
</style>
