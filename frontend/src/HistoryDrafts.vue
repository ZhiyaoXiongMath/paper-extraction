<script setup lang="ts">
import { computed } from "vue";
import type { DraftArtifact } from "./contracts";
import { label, tone } from "./format.js";
import { visibleFailureReasons } from "./paper-downloads.js";
const props = defineProps<{ versions: DraftArtifact[] }>();
const historical = computed(() => props.versions.filter(v => !v.current));
const attempts = computed(() => {
  const grouped = new Map<number, DraftArtifact[]>();
  for (const version of historical.value) {
    const group = grouped.get(version.attempt);
    if (group) group.push(version);
    else grouped.set(version.attempt, [version]);
  }
  return [...grouped.entries()].map(([attempt, drafts]) => ({
    attempt, drafts, status: drafts[0].attempt_status,
    reasons: visibleFailureReasons({ status: drafts[0].attempt_status, reason: drafts[0].attempt_reason }),
  }));
});
</script>
<template>
  <details v-if="historical.length" class="draft-history">
    <summary>历史草稿 <small>{{ historical.length }} 个版本</small></summary>
    <section v-for="a in attempts" :key="a.attempt">
      <header><strong>第 {{ a.attempt }} 次尝试</strong>
        <span v-if="a.status" class="pill" :class="tone(a.status)">{{ label(a.status) }}</span>
      </header>
      <p v-for="reason in a.reasons" :key="reason" class="reason">{{ reason }}</p>
      <ul>
        <li v-for="draft in a.drafts" :key="draft.round + '-' + draft.draft_sha256">
          <span>第 {{ draft.round }} 轮草稿</span>
          <small v-if="draft.recompiled">按保存稿重新编译</small>
          <span class="links">
            <a v-if="draft.files.tex" :href="draft.files.tex.url" download>下载 TeX</a>
            <a v-if="draft.files.pdf" :href="draft.files.pdf.url" target="_blank" rel="noopener">查看 PDF</a>
            <small v-if="!draft.files.tex && !draft.files.pdf">无草稿文件</small>
          </span>
        </li>
      </ul>
    </section>
  </details>
</template>
<style scoped>
.draft-history { border: 1px solid var(--line); border-radius: 8px; background: white; margin: 12px 0 18px; padding: 10px 16px; }
summary { cursor: pointer; font-weight: 600; }
summary small { margin-left: 5px; font-weight: normal; }
section { margin-top: 12px; border-top: 1px solid var(--line); padding-top: 12px; }
header, li, .links { display: flex; align-items: center; flex-wrap: wrap; gap: 10px; }
header, li { font-size: 14px; }
small, .reason { color: var(--muted); }
.reason { margin: 7px 0; font-size: 13px; overflow-wrap: anywhere; }
ul { list-style: none; margin: 8px 0 0; padding: 0; }
li { padding: 5px 0; }
.links { margin-left: auto; }
.links a { color: var(--ink); text-decoration: underline; text-underline-offset: 3px; }
</style>
