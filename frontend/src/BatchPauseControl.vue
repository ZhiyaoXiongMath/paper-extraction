<script setup lang="ts">
import { computed } from "vue";
import type { BatchSummary } from "./contracts";
import Icon from "./Icon.vue";

const props = defineProps<{ batch: BatchSummary; busy?: "pause" | "resume" }>();
const emit = defineEmits<{ change: [action: "pause" | "resume"] }>();

const active = computed(() => Boolean(props.batch.counts.queued || props.batch.counts.running));
const pausing = computed(() => Boolean(props.batch.batch_pause?.pausing || props.busy === "pause"));
const paused = computed(() => Boolean(props.batch.batch_pause?.paused));
</script>

<template>
  <div class="batch-pause-actions">
    <button v-if="active || pausing || !paused" type="button" class="btn" :disabled="!active || pausing || !!busy"
      :title="active || pausing ? '排队任务立即暂停，运行中的任务在当前环节结束后暂停' : '本批次暂无运行或排队任务'"
      @click="emit('change', 'pause')">
      <Icon name="pause" :size="15" />{{ pausing ? "正在暂停批次…" : "暂停批次" }}
    </button>
    <button v-if="paused && !pausing" type="button" class="btn" :disabled="!!busy"
      title="继续由此次批量操作暂停的任务" @click="emit('change', 'resume')">
      <Icon name="play" :size="15" />{{ busy === "resume" ? "正在继续…" : "继续批次" }}
    </button>
  </div>
</template>

<style scoped>
.batch-pause-actions { display: flex; gap: 8px; }
</style>
