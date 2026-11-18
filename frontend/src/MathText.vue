<script setup lang="ts">
import { ref, watch, onMounted } from "vue";
import renderMathInElement from "katex/contrib/auto-render";
import "katex/dist/katex.min.css";
const props = defineProps({ text: String });
const el = ref<HTMLElement>();
function render() {
  if (!el.value) return;
  el.value.textContent = props.text || "";
  renderMathInElement(el.value, {
    throwOnError: false,
    trust: false,
    strict: "ignore",
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "\\[", right: "\\]", display: true },
      { left: "\\(", right: "\\)", display: false },
      { left: "$", right: "$", display: false },
    ],
  });
}
onMounted(render);
watch(() => props.text, render, { flush: "post" });
</script>
<template><div class="math-text" ref="el"></div></template>
