<script setup lang="ts">
import { onMounted, ref } from "vue";
import { ApiError, errorMessage, request } from "./api";

const props = defineProps<{ batchId?: string; batchIds: string[]; name: string; paperCount: number }>();
const emit = defineEmits<{ cancel: []; deleted: [ids: string[]] }>();
const dialog = ref<HTMLDialogElement | null>(null);
const busy = ref(false), error = ref(""), unavailable = ref(false);
onMounted(() => dialog.value?.showModal());

async function confirmDeletion() {
  if (busy.value || unavailable.value) return;
  busy.value = true;
  error.value = "";
  try {
    const url = props.batchId ? "/batches/" + encodeURIComponent(props.batchId) : "/batches";
    const options: RequestInit = { method: "DELETE" };
    if (!props.batchId) {
      options.headers = { "Content-Type": "application/json" };
      options.body = JSON.stringify({ batch_ids: props.batchIds });
    }
    const result = await request<{ deleted_batch_ids: string[] }>(url, options);
    emit("deleted", result.deleted_batch_ids);
  } catch (e) {
    error.value = errorMessage(e);
    unavailable.value = e instanceof ApiError && [404, 409].includes(e.status);
  } finally {
    busy.value = false;
  }
}
</script>

<template>
  <dialog ref="dialog" class="delete-history-dialog" aria-labelledby="delete-history-title"
    aria-describedby="delete-history-description" @cancel.prevent="!busy && emit('cancel')">
    <h2 id="delete-history-title">{{ batchId ? "删除此批次？" : "清除已结束批次？" }}</h2>
    <div id="delete-history-description">
      <p v-if="batchId">即将删除「{{ name }}」中的 {{ paperCount }} 篇论文记录。</p>
      <p v-else>即将清除 {{ batchIds.length }} 个已结束批次，共 {{ paperCount }} 篇论文。</p>
      <p>工作台中的运行记录、草稿和成品文件将永久删除。</p>
    </div>
    <p v-if="error" class="notice error" role="alert">{{ error }}</p>
    <div class="history-actions dialog-actions">
      <button type="button" class="btn" autofocus :disabled="busy" @click="emit('cancel')">取消</button>
      <button type="button" class="btn destructive" :disabled="busy || unavailable" @click="confirmDeletion">
        {{ busy ? "正在删除…" : "确认删除" }}
      </button>
    </div>
  </dialog>
</template>
