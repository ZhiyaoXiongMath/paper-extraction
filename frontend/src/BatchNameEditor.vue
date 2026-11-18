<script setup lang="ts">
import { nextTick, ref } from "vue";
import { request, errorMessage } from "./api";
import Icon from "./Icon.vue";

const props = defineProps<{ id: string; name?: string | null }>();
const emit = defineEmits<{ saved: [id: string, name: string] }>();
const editing = ref(false), saving = ref(false), value = ref(""), error = ref("");
const input = ref<HTMLInputElement>(), toggle = ref<HTMLButtonElement>();

async function edit() {
  value.value = props.name ?? "";
  error.value = "";
  editing.value = true;
  await nextTick();
  input.value?.focus();
  input.value?.select();
}

async function close() {
  editing.value = false;
  await nextTick();
  toggle.value?.focus();
}

async function save() {
  const name = value.value.trim();
  if (!name || saving.value) return;
  saving.value = true;
  error.value = "";
  try {
    const saved = await request<{ id: string; name: string }>("/batches/" + encodeURIComponent(props.id) + "/name", {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    emit("saved", saved.id, saved.name);
    await close();
  } catch (e) {
    error.value = errorMessage(e);
  } finally {
    saving.value = false;
  }
}
</script>
<template>
  <div class="batch-name">
    <form v-if="editing" class="name-editor" @submit.prevent="save" @keydown.esc.prevent="!saving && close()">
      <div class="edit-line">
        <input ref="input" v-model="value" aria-label="批次名称" placeholder="批次名称"
          :disabled="saving" required @keydown.enter="$event.isComposing && $event.preventDefault()" />
        <button type="submit" class="icon-action confirm" :disabled="saving || !value.trim()"
          aria-label="保存名称" title="保存名称"><Icon name="check" :size="16" /></button>
        <button type="button" class="icon-action" :disabled="saving" @click="close"
          aria-label="取消编辑" title="取消编辑"><Icon name="close" :size="15" /></button>
      </div>
      <span v-if="error" class="name-error" role="alert">{{ error }}</span>
    </form>
    <template v-else>
      <div class="name-label"><slot /></div>
      <button ref="toggle" type="button" class="icon-action rename"
        :aria-label="(name ? '修改批次名称：' : '填写批次名称：') + (name || id.slice(0, 8))"
        :title="name ? '修改名称' : '填写名称'" @click="edit"><Icon name="pencil" :size="15" /></button>
    </template>
  </div>
</template>
<style scoped>
.batch-name { display: flex; align-items: center; gap: 8px; min-width: 0; }
.name-label { min-width: 0; }
.name-editor { width: 100%; }
.edit-line { display: flex; align-items: center; gap: 3px; height: 34px; padding: 2px 3px 2px 9px;
  border: 1px solid #99b2a4; border-radius: 5px; background: white; }
.edit-line:focus-within { border-color: #478064; box-shadow: 0 0 0 2px #e3eee7; }
.edit-line input { flex: 1; width: 0; min-width: 0; padding: 0; border: 0; outline: 0; font: inherit; color: var(--ink); }
.icon-action { display: inline-grid; place-items: center; flex-shrink: 0; width: 28px; height: 28px;
  padding: 0; border: 0; border-radius: 4px; background: transparent; color: #65776c; }
.icon-action:hover, .icon-action:focus-visible { background: #e8eee9; color: var(--accent); }
.icon-action:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.confirm { color: var(--accent); background: #e4eee7; }
.confirm:disabled { color: #839089; background: #f0f3f0; }
.name-error { display: block; white-space: normal; margin-top: 4px; color: var(--red); }
</style>
