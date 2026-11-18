<script setup lang="ts">
import { computed } from "vue";
import type { Draft } from "./contracts";
import { readBlocks } from "./format.js";
import { readerFragment } from "./reader.js";
import Icon from "./Icon.vue";
import MathText from "./MathText.vue";

const props = defineProps<{
  draft?: Draft;
  links: Record<string, string[]>;
  activeEvidence: string;
}>();
const emit = defineEmits<{ evidence: [id: string] }>();
const blocks = computed(() => (props.draft?.reader_blocks ?? readBlocks(props.draft || {})).map((block) => {
  const body = readerFragment(block.text, block.heading, props.draft?.references, "en");
  const caption = block.caption === undefined ? body.caption
    : readerFragment(block.caption, "", props.draft?.references, "en").text;
  return { ...block, text: body.text, caption };
}));
function evidenceLabel(id: string, index: number) {
  const page = /^p(\d+)-\d+$/.exec(id)?.[1];
  return page ? "Evidence · PDF p. " + page : "Evidence " + (index + 1);
}
</script>

<template>
  <div v-if="blocks.length" class="reader" lang="en">
    <h2>Introduction</h2>
    <article v-for="b in blocks" :key="b.id" class="read-block">
      <h3 v-if="b.heading">{{ b.heading }}<template v-if="b.caption"> <MathText
        class="reader-locator" :text="'(' + b.caption + ').'" inline /></template></h3>
      <MathText :text="b.text" />
      <div class="citations">
        <button v-for="(id, i) in links[b.id]" :key="id" :class="{ active: id === activeEvidence }"
          @click="emit('evidence', id)">
          <Icon name="link" :size="12" />{{ evidenceLabel(id, i) }}
        </button>
      </div>
    </article>
    <div v-if="draft?.references?.length" class="reference-list">
      <h3>References</h3>
      <ol>
        <li v-for="r in draft.references" :key="r.key"><MathText :text="readerFragment(r.text, '', draft.references, 'en').text" /></li>
      </ol>
    </div>
  </div>
  <div v-else class="empty" lang="en">
    <Icon name="book" :size="26" />
    <h3>No introduction available yet</h3>
    <p>See the run log for progress and check results.</p>
  </div>
</template>

<style scoped>
.reader-locator { font: inherit; font-weight: normal; }
</style>
