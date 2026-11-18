<script setup lang="ts">
import { computed } from "vue";
import "katex/dist/katex.min.css";
import { renderMarkdown } from "./rich-text.js";

const props = defineProps<{ text?: string; candidateIds?: ReadonlySet<string>; inline?: boolean }>();
const emit = defineEmits<{ candidate: [id: string] }>();
const content = computed(() => {
  const html = renderMarkdown(props.text || "", props.candidateIds);
  return props.inline ? html.trim().replace(/^<p>(.*)<\/p>$/s, "$1") : html;
});
function navigate(event: MouseEvent) {
  const target = event.target;
  if (!(target instanceof Element)) return;
  const id = target.closest<HTMLElement>("[data-candidate-ref]")?.dataset.candidateRef;
  if (id && props.candidateIds?.has(id)) emit("candidate", id);
}
</script>

<template><component :is="inline ? 'span' : 'div'" class="math-text" v-html="content" @click="navigate" /></template>

<style scoped>
.math-text { white-space: normal; overflow-wrap: anywhere; min-width: 0; }
.math-text :deep(p) { margin: 0; }
.math-text :deep(p + p), .math-text :deep(ul), .math-text :deep(ol),
.math-text :deep(blockquote), .math-text :deep(pre) { margin-block: .55em; }
.math-text :deep(ul), .math-text :deep(ol) { padding-inline-start: 1.5em; }
.math-text :deep(pre) { overflow-x: auto; white-space: pre; }
.math-text :deep(code) { font-size: .9em; overflow-wrap: anywhere; }
.math-text :deep(.katex) { font-size: 1.08em; }
.math-text :deep(.katex-display) { overflow-x: auto; overflow-y: hidden; margin-block: .65em; }
.math-text :deep(.candidate-ref) {
  color: #315c47; background: transparent; border: 0; padding: 0 .07em; font: inherit;
  font-weight: 600; text-decoration: underline; text-underline-offset: .16em; cursor: pointer;
}
.math-text :deep(.candidate-ref:hover) { color: #193d2a; }
</style>
