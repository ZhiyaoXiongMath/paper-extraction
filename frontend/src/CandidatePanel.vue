<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue";
import type { Candidate } from "./contracts";
import { kindName } from "./format.js";
import { requiredRelations } from "./candidate-relations.js";
import Icon from "./Icon.vue";
import MathText from "./MathText.vue";

const props = defineProps<{ candidates: Candidate[] }>();
const emit = defineEmits<{ evidence: [location: string] }>();
const container = ref<HTMLElement>();
const expanded = ref(new Set<string>());
const current = ref("");
const knownIds = computed(() => new Set(props.candidates.map(c => c.candidate_id).filter((id): id is string => !!id)));
const decisions: Record<string, string> = {admitted:"采纳", excluded:"排除", unresolved:"依据待定"};
const scope: Record<string, string> = {admitted:"采纳", independent:"独立贡献", ingredient:"证明步骤", equivalent:"等价表述", linked_clause:"同组陈述", excluded:"排除", unresolved:"依据待定"};
const itemKey = (c: Candidate, i: number) => c.candidate_id || "row-" + i;

function setOpen(key: string, value: boolean) {
  const next = new Set(expanded.value);
  if (value) next.add(key);
  else next.delete(key);
  expanded.value = next;
}

async function goTo(id: string) {
  if (!knownIds.value.has(id)) return;
  current.value = id;
  setOpen(id, true);
  await nextTick();
  const node = container.value?.querySelector<HTMLElement>('[data-candidate-id="' + CSS.escape(id) + '"]');
  node?.focus({preventScroll:true});
  node?.scrollIntoView({block:"start"});
}

function records(c: Candidate): [string, string][] {
  return ([
    ["逐段记录的类别", c.local_kind && kindName(c.local_kind)],
    ["逐段记录的判定", c.local_decision && (decisions[c.local_decision] || c.local_decision)],
    ["逐段记录的理由", c.local_reason], ["整篇复核角色", c.scope_role && (scope[c.scope_role] || c.scope_role)],
    ["来源摘录", c.scope_excerpt], ["理由类型", c.scope_reason_type === "paraphrase" ? "转述" : c.scope_reason_type],
    ["判定版本", c.scope_evidence_version],
    ["核心准入待定", c.scope_central_open === undefined ? undefined : c.scope_central_open ? "是" : "否"],
  ] as [string, string | null | undefined][]).filter((pair): pair is [string, string] => !!pair[1]);
}
function relations(c: Candidate): [string, string[]][] {
  return ([
    ...requiredRelations(c.decision_reason, c.needed_by), ["为候选提供背景", c.context_for],
    ["等价表述归入", c.represented_by ? [c.represented_by] : []],
    ["所属定理组", c.scope_group_id ? [c.scope_group_id] : []],
  ] as [string, string[] | undefined][]).filter((pair): pair is [string, string[]] => !!pair[1]?.length);
}
function locations(c: Candidate): [string, string[]][] {
  return ([
    ["逐段取证位置", c.local_evidence_ids], ["整篇选入依据", c.selection_evidence_ids],
    ["范围判断依据", c.scope_evidence_id ? [c.scope_evidence_id] : []],
  ] as [string, string[] | undefined][]).filter((pair): pair is [string, string[]] => !!pair[1]?.length);
}
watch(() => props.candidates, () => {
  if (current.value && !knownIds.value.has(current.value)) current.value = "";
});
</script>

<template>
  <section ref="container" class="candidates-panel">
    <div class="panel-head"><h2>候选贡献</h2><small>{{ candidates.length }} 条</small></div>
    <article v-for="(c, i) in candidates" :key="itemKey(c, i)" class="record candidate-record"
      :class="{ 'candidate-focus': current && current === c.candidate_id }" :data-candidate-id="c.candidate_id" tabindex="-1">
      <div class="candidate-heading">
        <span v-if="c.candidate_id" class="mono candidate-id">{{ c.candidate_id }}</span>
        <span class="kind">{{ kindName(c.kind) }}</span>
        <span v-if="c.decision" class="candidate-status" :class="c.decision">{{ decisions[c.decision] || c.decision }}</span>
      </div>
      <MathText :text="c.claim" :candidate-ids="knownIds" @candidate="goTo" />
      <div v-if="c.decision_reason" class="candidate-reason">
        <span>判定理由</span>
        <MathText :text="c.decision_reason" :candidate-ids="knownIds" @candidate="goTo" />
      </div>
      <div v-if="c.evidence_ids?.length" class="citations">
        <button v-for="id in c.evidence_ids" :key="id" @click="emit('evidence', id)">
          <Icon name="link" :size="12" />原文 {{ id }}
        </button>
      </div>
      <button class="expand-candidate" :aria-expanded="expanded.has(itemKey(c, i))"
        @click="setOpen(itemKey(c, i), !expanded.has(itemKey(c, i)))">
        {{ expanded.has(itemKey(c, i)) ? "收起候选详情" : "展开候选详情" }}
      </button>
      <dl v-if="expanded.has(itemKey(c, i))" class="facts candidate-facts">
        <template v-if="c.assumptions">
          <dt>成立条件</dt><dd><MathText :text="c.assumptions" :candidate-ids="knownIds" @candidate="goTo" /></dd>
        </template>
        <template v-if="c.dependencies?.length">
          <dt>依赖定义</dt><dd><ul><li v-for="(dep, j) in c.dependencies" :key="j">
            <MathText :text="dep" :candidate-ids="knownIds" @candidate="goTo" /></li></ul></dd>
        </template>
        <template v-if="c.significance">
          <dt>数学意义</dt><dd><MathText :text="c.significance" :candidate-ids="knownIds" @candidate="goTo" /></dd>
        </template>
        <template v-if="c.independent_question">
          <dt>独立问题</dt><dd><MathText :text="c.independent_question" :candidate-ids="knownIds" @candidate="goTo" /></dd>
        </template>
        <template v-if="c.source_number">
          <dt>原文编号</dt><dd><MathText :text="c.source_number" :candidate-ids="knownIds" @candidate="goTo" /></dd>
        </template>
        <template v-for="[label, ids] in relations(c)" :key="label">
          <dt>{{ label }}</dt><dd class="candidate-relations"><template v-for="id in ids" :key="id">
            <button v-if="knownIds.has(id)" class="candidate-link" @click="goTo(id)">{{ id }}</button>
            <span v-else class="mono">{{ id }}</span>
          </template></dd>
        </template>
        <template v-for="[label, ids] in locations(c)" :key="label">
          <dt>{{ label }}</dt><dd class="citations"><button v-for="id in ids" :key="id" @click="emit('evidence', id)">
            <Icon name="link" :size="12" />{{ id }}</button></dd>
        </template>
      </dl>
      <details v-if="expanded.has(itemKey(c, i)) && records(c).length" class="candidate-judgments">
        <summary>判定记录</summary>
        <dl class="facts candidate-facts">
          <template v-for="[label, value] in records(c)" :key="label">
            <dt>{{ label }}</dt><dd><MathText :text="value" :candidate-ids="knownIds" @candidate="goTo" /></dd>
          </template>
        </dl>
      </details>
    </article>
    <div v-if="!candidates.length" class="empty">暂无候选贡献记录。</div>
  </section>
</template>

<style scoped>
.candidate-record { scroll-margin-top: 18px; outline: none; }
.candidate-focus { border-left: 3px solid #537a63; padding-left: 19px; }
.candidate-heading, .candidate-relations { display: flex; align-items:center; gap: .6em; flex-wrap: wrap; }
.candidate-heading { margin-bottom: 8px; }
.candidate-id { font-size: 13px; color: #46564e; }
.candidate-status { font-size: 12px; color: #59645e; }
.candidate-status.admitted { color: #315a43; }
.candidate-reason { margin-top: 8px; color: #49594f; }
.candidate-reason > span, .expand-candidate, .candidate-link { font-size: 12px; }
.candidate-reason > span { font-weight: 600; }
.candidate-reason :deep(.math-text), .candidate-facts :deep(.math-text) { font-size: 14px; }
.expand-candidate, .candidate-link { background: transparent; border: 0; color: #365c48; cursor: pointer; }
.expand-candidate { padding: 3px 0; margin-top: 8px; text-decoration: underline; text-underline-offset: .19em; }
.candidate-link { text-decoration: underline; padding: 0 .08em; }
.candidate-facts { margin-top: 12px; padding: 12px; border-radius: 5px; background: #f6f8f4; }
.candidate-judgments { font-size: 13px; margin-top: 8px; }
.candidate-judgments summary { cursor: pointer; color: #49594f; }
.candidate-facts dd { min-width: 0; }
.candidate-facts .citations { margin-top: 0; }
.candidate-facts ul { padding-left: 1.2em; margin: 0; }
@media (max-width: 580px) { .candidate-facts { grid-template-columns: 1fr; } .candidate-facts dt { font-weight: 600; } }
</style>
