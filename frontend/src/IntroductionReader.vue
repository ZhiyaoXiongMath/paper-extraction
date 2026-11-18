<script setup lang="ts">
import { computed } from "vue";
import type { Draft } from "./contracts";
import { readBlocks, stageName } from "./format.js";
import { readerFragment } from "./reader.js";
import Icon from "./Icon.vue";
import MathText from "./MathText.vue";

const props = defineProps<{
  draft?: Draft;
  stage: string;
  links: Record<string, string[]>;
  activeEvidence: string;
}>();
const emit = defineEmits<{ evidence: [id: string] }>();
const blocks = computed(() => readBlocks(props.draft || {}).map((block) => ({
  ...block, ...readerFragment(block.text, block.heading, props.draft?.references),
})));
function evidenceLabel(id: string, index: number) {
  const page = /^p(\d+)-\d+$/.exec(id)?.[1];
  return page ? "原文依据 · PDF 第 " + page + " 页" : "原文依据 " + (index + 1);
}
</script>

<template>
  <div v-if="blocks.length" class="reader" lang="en">
    <h2>Introduction</h2>
    <article v-for="b in blocks" :key="b.id" class="read-block">
      <h3 v-if="b.heading">{{ b.heading }}</h3>
      <div v-if="b.caption" class="reader-locator" lang="zh-CN">
        <MathText :text="b.caption" />
      </div>
      <MathText :text="b.text" />
      <div class="citations" lang="zh-CN">
        <button v-for="(id, i) in links[b.id]" :key="id" :class="{ active: id === activeEvidence }"
          @click="emit('evidence', id)">
          <Icon name="link" :size="12" />{{ evidenceLabel(id, i) }}
        </button>
      </div>
    </article>
    <div v-if="draft?.references?.length" class="reference-list">
      <h3>References</h3>
      <ol>
        <li v-for="r in draft.references" :key="r.key"><MathText :text="readerFragment(r.text, '', draft.references).text" /></li>
      </ol>
    </div>
  </div>
  <div v-else class="empty">
    <Icon name="book" :size="26" />
    <h3>暂无引言文本</h3>
    <p>当前阶段为{{ stageName(stage) }}，可查看运行记录。</p>
  </div>
</template>

<style scoped>
.reader-locator { margin: 0 0 6px; color: var(--muted); }
.reader-locator :deep(.math-text) { font-family: inherit; font-size: 13px; }
</style>
