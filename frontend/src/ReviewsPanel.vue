<script setup lang="ts">
import { computed } from "vue";
import type { Draft, Review, EditorialDefect } from "./contracts";
import Icon from "./Icon.vue";
import MathText from "./MathText.vue";
import { stageName } from "./format.js";
import { readerFragment } from "./reader.js";
import { defectKind, editLabel, evidenceLabel, readableReviewText, reviewDetails, reviewField } from "./review-format.js";

const props = defineProps<{ reviews: Review[]; draft?: Draft; stage: string }>();
const emit = defineEmits<{ evidence: [id: string] }>();
const routeNames = { draft: "修改草稿", extract: "回查原文", pass: "通过" };

function excerpts(item: EditorialDefect) {
  const rows = [
    ["被指出的草稿文字", item.current_excerpt],
    [editLabel(item.operation), item.replacement_text],
    ["审稿引用的原文", item.source_excerpt],
  ];
  return rows.flatMap(([label, raw]) => raw
    ? [{ label, raw, ...readerFragment(raw, "", props.draft?.references || []) }]
    : []);
}

const records = computed(() => props.reviews.map((review, index) => {
  const { defects, plain } = reviewDetails(review);
  return { review, index, plain: plain.map(raw => ({ raw, text: readableReviewText(raw, props.draft?.references) })),
    defects: defects.map(item => ({ item, excerpts: excerpts(item) })) };
}).reverse());
</script>

<template>
  <div class="panel-head">
    <h2>自动检查</h2>
    <small>{{ stageName(stage) }}</small>
  </div>
  <details v-for="({ review, index, defects, plain }, order) in records" :key="index"
           class="record audit-record" :open="order === 0">
    <summary>
      <span class="round-title">第 {{ review.attempt || 1 }} 轮 · {{ stageName(review.stage) }}<template v-if="review.page"> · 第 {{ review.page }} 页</template></span>
      <span class="pill" :class="review.passed ? 'good' : 'bad'">{{ review.passed ? "通过" : "未通过" }}</span>
    </summary>
    <div class="audit-body">
      <p v-if="review.route && review.route !== 'pass'" class="route">
        审稿建议{{ routeNames[review.route] }}<template v-if="defects.length"> · {{ defects.length }} 条意见</template>
      </p>
      <section v-for="({ item, excerpts: parts }, j) in defects" :key="j" class="audit-defect">
        <header>
          <h3>{{ reviewField(item.draft_field) }}</h3>
          <small>{{ defectKind(item.kind) }}</small>
        </header>
        <MathText :text="item.required_change || ''" />
        <div v-if="item.evidence_id" class="citations">
          <button @click="emit('evidence', item.evidence_id)">
            <Icon name="link" :size="12" />{{ evidenceLabel(item.evidence_id) }}
          </button>
        </div>
        <details v-if="parts.length" class="compare">
          <summary>对照草稿、修改建议与原文</summary>
          <dl>
            <template v-for="part in parts" :key="part.label">
              <dt>{{ part.label }}</dt>
              <dd>
                <small v-if="part.caption">{{ part.caption }}</small>
                <MathText :text="part.text" />
              </dd>
            </template>
          </dl>
        </details>
      </section>
      <section v-for="target in review.repair_targets || []" :key="target.draft_field" class="audit-defect">
        <header><h3>{{ reviewField(target.draft_field) }}</h3><small>排版诊断</small></header>
        <MathText v-for="(diagnostic, j) in target.diagnostics" :key="j" :text="diagnostic" />
        <details v-if="target.current_text" class="compare">
          <summary>查看被检查的草稿文字</summary>
          <MathText :text="target.current_text" />
        </details>
      </section>
      <ul v-if="plain.length" class="issues">
        <li v-for="(issue, i) in plain" :key="i">
          <MathText :text="issue.text" />
          <details v-if="issue.raw !== issue.text" class="compare">
            <summary>查看保存的原始诊断</summary><pre>{{ issue.raw }}</pre>
          </details>
        </li>
      </ul>
      <div v-if="review.evidence_ids?.length" class="citations">
        <button v-for="id in review.evidence_ids" :key="id" @click="emit('evidence', id)">
          <Icon name="link" :size="12" />{{ evidenceLabel(id) }}
        </button>
      </div>
    </div>
  </details>
  <div v-if="!records.length" class="empty">暂无检查结果。</div>
</template>

<style scoped>
.audit-record > summary, .compare > summary { cursor: pointer; }
.round-title { margin-right: .7rem; font-weight: 600; }
.audit-body { min-width: 0; }
.route { margin: .8rem 0 .2rem; color: var(--muted); font-size: .85rem; }
.audit-defect { border-top: 1px solid var(--line); padding: 1rem 0 .25rem; margin-top: .8rem; }
.audit-defect header { display: flex; align-items: baseline; gap: .7rem; flex-wrap: wrap; margin-bottom: .35rem; }
.audit-defect h3 { margin: 0; font-size: 1rem; }
.audit-defect small, .compare dt { color: var(--muted); }
.audit-body :deep(.math-text) { font-size: .9rem; }
.compare { margin-top: .65rem; }
.compare > summary { font-size: .82rem; color: #41624d; }
.compare dl { border-left: 2px solid var(--line); margin: .6rem 0 0 .25rem; padding-left: .8rem; }
.compare dt { font-size: .78rem; margin-top: .7rem; font-weight: 600; }
.compare dd { margin: .15rem 0 .4rem; }
.compare pre { white-space: pre-wrap; overflow-wrap: anywhere; }
</style>
