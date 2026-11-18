<script setup lang="ts">
import type { Provider, ImportPreview, SourcePaper } from "./contracts";
import { ref, computed } from "vue";
import { request, post, errorMessage } from "./api";
import Icon from "./Icon.vue";
import { droppedFiles, pickedFiles, type UploadFile } from "./files";
const props = defineProps<{ providers: Provider[]; provider: string }>();
const emit = defineEmits(["created", "provider"]);
const picker = ref<HTMLInputElement>(),
  folder = ref<HTMLInputElement>(),
  preview = ref<ImportPreview | null>(null),
  rows = ref<(SourcePaper & { selected: boolean })[]>([]),
  pending = ref(""),
  error = ref(""),
  drag = ref(false);
const selected = computed(() => rows.value.filter((r) => r.selected));
const chosen = computed(() =>
  props.providers.find((p) => p.id === props.provider),
);
const ready = computed(
  () =>
    selected.value.length > 0 &&
    selected.value.every((r) => !r.entries?.length || r.entry) &&
    chosen.value?.configured &&
    !pending.value,
);
async function upload(files: UploadFile[]) {
  if (pending.value || !files?.length) return;
  const items = files.filter(
    ({ file, path }) =>
      !path.split("/").some((s) => s.startsWith(".")) &&
      /\.(pdf|tex|bib|bbl|sty|cls|bst|def|png|jp[e]?g|eps)$/i.test(file.name),
  );
  if (!items.some((f) => /\.(pdf|tex)$/i.test(f.file.name))) {
    error.value = "所选文件中没有 PDF 或 TeX。";
    return;
  }
  const body = new FormData();
  for (const { file } of items) body.append("files", file, file.name);
  body.append("paths", JSON.stringify(items.map((item) => item.path)));
  pending.value = "upload";
  error.value = "";
  try {
    const result = await request<ImportPreview>("/imports", {
      method: "POST",
      body,
    });
    preview.value = result;
    rows.value = result.papers.map((p) => ({
      ...p,
      entry: p.entry || (p.entries?.length === 1 ? p.entries[0] : ""),
      selected: true,
    }));
    if (!rows.value.length)
      error.value = "没有识别到可处理的论文，请检查源文件。";
  } catch (e) {
    error.value = errorMessage(e);
  } finally {
    pending.value = "";
  }
}
async function start() {
  if (!ready.value || !preview.value) return;
  pending.value = "start";
  error.value = "";
  try {
    const b = await post<{ id: string }>("/batches", {
      import_id: preview.value.id,
      provider: props.provider,
      papers: selected.value.map((r) => ({ id: r.id, entry: r.entry })),
    });
    emit("created", b.id);
  } catch (e) {
    error.value = errorMessage(e);
  } finally {
    pending.value = "";
  }
}
async function drop(event: DragEvent) {
  drag.value = false;
  if (pending.value || !event.dataTransfer) return;
  pending.value = "read";
  try {
    const items = await droppedFiles(event.dataTransfer);
    pending.value = "";
    await upload(items);
  } catch (err) {
    error.value = errorMessage(err);
    pending.value = "";
  }
}
function choose(e: Event) {
  const el = e.target as HTMLInputElement;
  upload(pickedFiles(el.files));
  el.value = "";
}
</script>
<template>
  <section class="page-title">
    <div>
      <h1>导入论文</h1>
      <p>选择一批本地论文，确认主文件和模型后开始提取。</p>
    </div>
  </section>
  <div class="import-grid">
    <section class="panel">
      <div class="panel-head"><h2>论文来源</h2></div>
      <div class="panel-pad">
        <div
          class="drop"
          :class="{ drag }"
          @dragover.prevent="drag = true"
          @dragleave.prevent="drag = false"
          @drop.prevent="drop"
        >
          <span class="upload-icon"><Icon name="upload" :size="27" /></span>
          <h3>
            {{
              pending === "upload" || pending === "read"
                ? "正在识别论文…"
                : "拖入论文文件或文件夹"
            }}
          </h3>
          <p>支持 PDF、单文件 TeX 和带主文件的源码目录。</p>
          <div class="actions">
            <button class="btn" :disabled="!!pending" @click="picker?.click()">
              <Icon name="file" :size="16" />选择文件</button
            ><button class="btn" :disabled="!!pending" @click="folder?.click()">
              <Icon name="folder" :size="16" />选择文件夹
            </button>
          </div>
        </div>
        <input
          ref="picker"
          type="file"
          class="sr"
          tabindex="-1"
          accept=".pdf,.tex"
          multiple
          @change="choose"
        /><input
          ref="folder"
          class="sr"
          tabindex="-1"
          type="file"
          webkitdirectory
          multiple
          @change="choose"
        />
        <div v-if="error" class="notice error" role="alert">
          <Icon name="fail" />{{ error }}
        </div>
        <div v-if="preview" class="preview">
          <div class="section-line">
            <h3>导入预览</h3>
            <small>已选 {{ selected.length }} / {{ rows.length }}</small>
          </div>
          <div v-for="r in rows" :key="r.id" class="import-row">
            <label class="check-row"
              ><input v-model="r.selected" type="checkbox" /><span>{{
                r.name
              }}</span
              ><small class="kind">{{ r.kind }}</small></label
            >
            <details v-if="r.files?.length" class="import-files">
              <summary>
                来源文件 · {{ r.file_count ?? r.files.length }} 个
              </summary>
              <ul>
                <li v-for="file in r.files" :key="file">{{ file }}</li>
              </ul>
            </details>
            <label v-if="r.entries?.length > 1" class="control subfield"
              >TeX 主文件<select v-model="r.entry" :disabled="!r.selected">
                <option disabled value="">选择主文件</option>
                <option v-for="f in r.entries" :key="f">{{ f }}</option>
              </select></label
            >
          </div>
        </div>
      </div>
    </section>
    <section class="panel run-config">
      <div class="panel-head"><h2>批次配置</h2></div>
      <div class="panel-pad">
        <label class="control"
          >模型服务<select
            :value="provider"
            @change="
              emit('provider', ($event.target as HTMLSelectElement).value)
            "
          >
            <option v-for="p in providers" :value="p.id" :key="p.id">
              {{ p.label }}
            </option>
          </select></label
        >
        <div class="config-pair">
          <span>提取规范</span><b>Introduction v13</b>
        </div>
        <div class="config-pair"><span>检查方式</span><b>全自动</b></div>
        <div v-if="chosen && !chosen.configured" class="notice">
          <span
            >此服务尚未配置 API Key。<a href="#/settings">查看模型设置</a></span
          >
        </div>
        <button class="btn primary wide" :disabled="!ready" @click="start">
          <Icon name="play" :size="16" />{{
            pending === "start" ? "正在创建…" : "开始提取"
          }}<template v-if="selected.length">
            · {{ selected.length }} 篇</template
          >
        </button>
      </div>
    </section>
  </div>
</template>
