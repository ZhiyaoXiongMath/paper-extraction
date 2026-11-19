<script setup lang="ts">
import type { BatchSummary, BatchDetail, Config, ReasoningSettings } from "./contracts";
import { ref, computed, onMounted, onBeforeUnmount, watch } from "vue";
import Icon from "./Icon.vue";
import ImportPanel from "./ImportPanel.vue";
import BatchNameEditor from "./BatchNameEditor.vue";
import BatchPauseControl from "./BatchPauseControl.vue";
import DeleteHistoryDialog from "./DeleteHistoryDialog.vue";
import SettingsPanel from "./SettingsPanel.vue";
import PaperDetail from "./PaperDetail.vue";
import { ApiError, request, post, errorMessage } from "./api";
import {
  canDeleteBatch,
  metrics,
  formatDate,
  shortId,
  batchPaperLabel,
  paperTone,
  terminal,
  stageName,
  eventSummary,
} from "./format.js";

function readRoute() {
  const parts = location.hash
    .slice(1)
    .split("/")
    .filter(Boolean)
    .map((x) => {
      try {
        return decodeURIComponent(x);
      } catch {
        return x;
      }
    });
  return {
    view: ["import", "settings", "batch"].includes(parts[0])
      ? parts[0]
      : location.hash
        ? "home"
        : "import",
    batch: parts[1],
    paper: parts[2],
  };
}
const route = ref(readRoute()),
  batches = ref<BatchSummary[]>([]),
  config = ref<Config>({ providers: [] }),
  detail = ref<BatchDetail | null>(null);
const error = ref(""),
  loading = ref(false),
  listReady = ref(false),
  provider = ref(""),
  retrying = ref(false);
const changingPaperState = ref(false);
const changingBatchState = ref<Record<string, "pause" | "resume">>({});
const selectedPaper = computed(() =>
  detail.value?.papers?.find((p) => p.id === route.value.paper),
);
const stats = computed(() => metrics(batches.value));
const batchRows = computed(() =>
  batches.value.map((b) => ({ ...b, stats: metrics([b]) })),
);
const endedBatches = computed(() => batches.value.filter((batch) => canDeleteBatch(batch, false)));
const deleteBlockedHint = "全部暂停或结束后可删除";
const deletion = ref<{ batchId?: string; batchIds: string[]; name: string; paperCount: number } | null>(null);
const historyNotice = ref("");
function askDeletion(batch?: BatchSummary) {
  if (deletion.value || (batch && (!canDeleteBatch(batch) || changingBatchState.value[batch.id]))) return;
  const targets = batch ? [batch] : endedBatches.value;
  if (!targets.length) return;
  historyNotice.value = "";
  deletion.value = {
    batchId: batch?.id, name: batch?.name || (batch ? "批次 #" + shortId(batch.id) : ""),
    batchIds: targets.map((item) => item.id), paperCount: targets.reduce((sum, item) => sum + item.total, 0),
  };
}
function clearDeletedBatches(ids: string[]) {
  const removed = new Set(ids);
  ++listSeq;
  batches.value = batches.value.filter((batch) => !removed.has(batch.id));
  try {
    const last = localStorage.getItem("paper-extraction.last-batch");
    if (last && removed.has(last)) localStorage.removeItem("paper-extraction.last-batch");
  } catch {}
  if (route.value.view === "batch" && removed.has(route.value.batch)) {
    ++detailSeq;
    closeStream();
    detail.value = null;
    history.replaceState(null, "", "#/");
    onHash();
  }
}
function finishDeletion(ids: string[]) {
  const skipped = (deletion.value?.batchIds.length ?? ids.length) - ids.length;
  deletion.value = null;
  clearDeletedBatches(ids);
  historyNotice.value = "已删除 " + ids.length + " 个批次。" +
    (skipped > 0 ? "另有 " + skipped + " 个所选批次状态已变化或已被删除。" : "");
  void fetchList();
}
function cancelDeletion() {
  deletion.value = null;
  void fetchList();
  if (route.value.view === "batch") void fetchDetail();
}
const completeCount = computed(
  () => detail.value?.papers?.filter((p) => terminal(p.status)).length || 0,
);
const queueFilter = ref("all");
const filters = [
  ["all", "全部"],
  ["queued", "排队"],
  ["running", "处理中"],
  ["completed", "已交付"],
  ["failed", "失败"],
];
const visiblePapers = computed(() =>
  (detail.value?.papers || []).filter(
    (p) => queueFilter.value === "all" || p.status === queueFilter.value,
  ),
);
const deliveredCount = computed(
  () =>
    detail.value?.papers.filter((p) => p.status === "completed").length ?? 0,
);
let events: EventSource | null = null,
  interval: number | undefined,
  detailSeq = 0,
  listSeq = 0,
  alive = true,
  streamBatch: string | null = null,
  lastEvent = 0,
  refreshTimer: number | undefined;
const path = (batch: string, paper?: string) =>
  "#/batch/" +
  encodeURIComponent(batch) +
  (paper ? "/" + encodeURIComponent(paper) : "");
const providerName = (id: string) =>
  config.value.providers.find((p) => p.id === id)?.label ?? id;
const nav = [
  { view: "home", icon: "grid", name: "批次工作台" },
  { view: "import", icon: "upload", name: "导入论文" },
  { view: "settings", icon: "settings", name: "模型设置" },
];
const statList = [
  ["queued", "排队", "clock"],
  ["running", "处理中", "activity"],
  ["completed", "已交付", "check"],
  ["failed", "失败", "fail"],
] as const;
function onHash() {
  historyNotice.value = "";
  route.value = readRoute();
  window.scrollTo(0, 0);
}
function nameSaved(id: string, name: string) {
  const batch = batches.value.find(b => b.id === id);
  if (batch) batch.name = name;
  if (detail.value?.id === id) detail.value.name = name;
}
async function fetchList() {
  const seq = ++listSeq;
  try {
    const data = await request<BatchSummary[]>("/batches");
    if (alive && seq === listSeq) {
      batches.value = data;
      listReady.value = true;
    }
  } catch (e) {
    if (alive && seq === listSeq && route.value.view === "home")
      error.value = errorMessage(e);
  }
}
async function fetchConfig() {
  try {
    config.value = await request<Config>("/config");
    let stored: string | null = null;
    try {
      const configuredDefault = config.value.default_provider || "";
      if (localStorage.getItem("paper-extraction.default-provider") === configuredDefault) {
        stored = localStorage.getItem("paper-extraction.provider");
      } else {
        localStorage.setItem("paper-extraction.default-provider", configuredDefault);
        localStorage.removeItem("paper-extraction.provider");
      }
    } catch {}
    const sources = config.value.providers;
    provider.value =
      sources.find((p) => p.id === stored)?.id ||
      config.value.default_provider ||
      sources[0]?.id ||
      "";
  } catch (e) {
    error.value = errorMessage(e);
  }
}
function setProvider(p: string) {
  provider.value = p;
  try {
    localStorage.setItem("paper-extraction.provider", p);
  } catch {}
}
function setReasoning(id: string, settings: ReasoningSettings) {
  const item = config.value.providers.find((p) => p.id === id);
  if (item) item.reasoning = settings;
}
function closeStream() {
  events?.close();
  events = null;
  streamBatch = null;
  clearTimeout(refreshTimer);
}
async function fetchDetail() {
  const id = route.value.batch;
  if (route.value.view !== "batch" || !id) return;
  if (detail.value?.id !== id) queueFilter.value = "all";
  const seq = ++detailSeq;
  try {
    const data = await request<BatchDetail>(
      "/batches/" + encodeURIComponent(id),
    );
    if (
      !alive ||
      seq !== detailSeq ||
      route.value.batch !== id ||
      route.value.view !== "batch"
    )
      return;
    detail.value = data;
    try {
      localStorage.setItem("paper-extraction.last-batch", data.id);
    } catch {}
    error.value = "";
    if (data.papers.every((p) => terminal(p.status))) {
      closeStream();
      await fetchList();
    } else if (!events || streamBatch !== id) subscribe(id);
  } catch (e) {
    if (alive && seq === detailSeq && route.value.view === "batch") {
      if (e instanceof ApiError && e.status === 404) {
        clearDeletedBatches([id]);
        historyNotice.value = "该批次已不存在。";
      } else {
        error.value = errorMessage(e);
      }
    }
  } finally {
    if (seq === detailSeq) loading.value = false;
  }
}
function subscribe(id: string) {
  closeStream();
  streamBatch = id;
  lastEvent = Date.now();
  events = new EventSource(
    "/api/batches/" + encodeURIComponent(id) + "/events",
  );
  events.addEventListener("update", () => {
    lastEvent = Date.now();
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(fetchDetail, 100);
  });
}
function focusMain() {
  document.getElementById("main")?.focus();
}
function openBatch(id: string) {
  fetchList();
  location.hash = path(id);
}
function changeView() {
  error.value = "";
  if (route.value.view === "batch" && route.value.batch) {
    if (detail.value?.id !== route.value.batch) {
      detail.value = null;
      closeStream();
      loading.value = true;
    }
    fetchDetail();
  } else {
    ++detailSeq;
    closeStream();
    detail.value = null;
    loading.value = false;
    if (route.value.view === "home") fetchList();
  }
}
async function retry() {
  const paper = selectedPaper.value;
  if (!paper || retrying.value) return;
  retrying.value = true;
  error.value = "";
  try {
    await post("/papers/" + encodeURIComponent(paper.id) + "/retry", { provider: provider.value });
    await fetchDetail();
  } catch (e) {
    error.value = errorMessage(e);
  } finally {
    retrying.value = false;
  }
}
async function changePaperState(action: "pause" | "resume") {
  const paper = selectedPaper.value;
  if (!paper || changingPaperState.value) return;
  changingPaperState.value = true;
  error.value = "";
  try {
    await post("/papers/" + encodeURIComponent(paper.id) + "/" + action, {});
    await fetchDetail();
  } catch (err) {
    error.value = errorMessage(err);
  } finally {
    changingPaperState.value = false;
  }
}
async function changeBatchState(id: string, action: "pause" | "resume") {
  if (changingBatchState.value[id]) return;
  changingBatchState.value[id] = action;
  error.value = "";
  try {
    await post("/batches/" + encodeURIComponent(id) + "/" + action, {});
    await Promise.all([fetchList(), detail.value?.id === id ? fetchDetail() : undefined]);
  } catch (err) {
    error.value = errorMessage(err);
  } finally {
    delete changingBatchState.value[id];
  }
}
watch(route, changeView);
onMounted(async () => {
  addEventListener("hashchange", onHash);
  await Promise.all([fetchList(), fetchConfig()]);
  if (!location.hash) {
    let last: string | null = null;
    try {
      last = localStorage.getItem("paper-extraction.last-batch");
    } catch {}
    if (last && batches.value.some((b) => b.id === last)) {
      history.replaceState(null, "", path(last));
      route.value = readRoute();
    }
  }
  changeView();
  interval = setInterval(() => {
    if (document.visibilityState !== "visible") return;
    if (route.value.view === "home") fetchList();
    if (
      route.value.view === "batch" &&
      (!detail.value || detail.value.papers.some((p) => !terminal(p.status))) &&
      Date.now() - lastEvent > 5500
    )
      fetchDetail();
  }, 6000);
});
onBeforeUnmount(() => {
  alive = false;
  removeEventListener("hashchange", onHash);
  clearInterval(interval);
  closeStream();
});
</script>
<template>
  <div class="shell">
    <a href="#main" class="skip" @click.prevent="focusMain">跳至主要内容</a>
    <aside class="sidebar">
      <a class="brand" href="#/"
        ><span class="mark"><Icon name="book" :size="25" /></span
        ><span class="brand-name">paper-extraction-v13</span></a
      >
      <nav aria-label="主导航">
        <a
          v-for="item in nav"
          :key="item.view"
          :href="item.view === 'home' ? '#/' : '#/' + item.view"
          :aria-current="
            route.view === item.view ||
            (route.view === 'batch' && item.view === 'home')
              ? 'page'
              : undefined
          "
          ><Icon :name="item.icon" /><span>{{ item.name }}</span></a
        >
      </nav>
      <div class="side-bottom">
        <Icon name="monitor" /><span>本机工作区</span>
      </div>
    </aside>
    <div class="area">
      <header class="topbar">
        <div class="crumb">
          <span>工作区</span><Icon name="chevron" :size="13" /><a
            v-if="route.view === 'batch'"
            href="#/"
            >批次工作台</a
          ><span v-else>{{ nav.find((n) => n.view === route.view)?.name }}</span
          ><template v-if="route.view === 'batch'"
            ><Icon name="chevron" :size="13" /><span>{{
              selectedPaper ? "论文详情" : "批次详情"
            }}</span></template
          >
        </div>
        <span class="profile"><i></i> paper-extraction-v13</span>
      </header>
      <main id="main" tabindex="-1">
        <div v-if="historyNotice" class="notice" role="status">{{ historyNotice }}</div>
        <div v-if="error" class="notice error" role="alert">
          <Icon name="fail" /><span>{{ error }}</span
          ><button
            class="text-btn"
            @click="
              error = '';
              fetchConfig();
              changeView();
            "
          >
            重试
          </button>
        </div>
        <template v-if="route.view === 'home'">
          <section class="page-title">
            <div>
              <h1>批次工作台</h1>
              <p>管理论文批次、跟踪提取进度并下载结果。</p>
            </div>
            <a class="btn primary" href="#/import"
              ><Icon name="plus" />新建批次</a
            >
          </section>
          <div class="stats">
            <article
              v-for="[key, name, icon] in statList"
              :key="key"
              class="stat"
            >
              <div class="stat-head">
                <span>{{ name }}</span
                ><Icon :name="icon" :size="17" />
              </div>
              <strong>{{ listReady ? stats[key] : "—" }}</strong>
            </article>
          </div>
          <section class="panel">
            <div class="panel-head">
              <h2>批次记录</h2>
              <div v-if="listReady" class="history-actions">
                <span class="muted mono">{{ batches.length }} 个批次</span>
                <button type="button" class="btn history-clear" :disabled="!endedBatches.length || !!deletion"
                  @click="askDeletion()"><Icon name="trash" :size="15" />清除已结束批次</button>
              </div>
            </div>
            <div v-if="!listReady" class="empty">正在读取批次记录…</div>
            <div v-else-if="!batches.length" class="empty home-empty">
              <div class="empty-icon"><Icon name="folder" :size="31" /></div>
              <h3>还没有论文批次</h3>
              <p>选择本地 PDF 或 TeX，创建第一批论文。</p>
              <a class="btn" href="#/import"
                ><Icon name="upload" :size="15" />导入论文</a
              >
            </div>
            <div v-else class="table-wrap">
              <table class="batch-record-table">
                <thead>
                  <tr>
                    <th>批次名称</th>
                    <th>创建时间</th>
                    <th>模型服务</th>
                    <th>论文数</th>
                    <th>进度</th>
                    <th><span class="sr">批次操作</span></th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="b in batchRows" :key="b.id">
                    <td>
                      <BatchNameEditor class="batch-name-cell" :id="b.id" :name="b.name" @saved="nameSaved">
                        <a :href="path(b.id)" class="batch-name-link" :title="b.name || '#' + shortId(b.id)">
                          <span v-if="b.name" class="named-batch">{{ b.name }}</span>
                          <template v-else><span class="unnamed-batch">未命名</span><span class="batch-code mono">#{{ shortId(b.id) }}</span></template>
                        </a>
                      </BatchNameEditor>
                    </td>
                    <td>{{ formatDate(b.created_at) }}</td>
                    <td>{{ providerName(b.provider) }}</td>
                    <td class="mono">{{ b.total }}</td>
                    <td>
                      <div class="progress">
                        <i
                          :style="{
                            width:
                              (b.total
                                ? (100 * (b.stats.completed + b.stats.failed + (b.stats.cancelled || 0))) /
                                  b.total
                                : 0) + '%',
                          }"
                        ></i>
                      </div>
                      <small
                        >排队 {{ b.stats.queued }} · 处理中
                        {{ b.stats.running }}</small
                      ><small
                        >已交付 {{ b.stats.completed }} · 失败
                        {{ b.stats.failed }}</small
                      >
                      <small v-if="b.summary_count">其中筛选说明 {{ b.summary_count }} 篇</small>
                      <small v-if="b.stats.paused || b.stats.cancelled">
                        <span v-if="b.stats.paused">已暂停 {{ b.stats.paused }}</span>
                        <span v-if="b.stats.paused && b.stats.cancelled"> · </span>
                        <span v-if="b.stats.cancelled">已终止 {{ b.stats.cancelled }}</span>
                      </small>
                    </td>
                    <td>
                      <div class="row-batch-actions">
                      <a
                        class="icon-link"
                        :href="path(b.id)"
                        :aria-label="'查看批次 ' + (b.name || shortId(b.id))"
                        ><Icon name="chevron" :size="16"
                      /></a>
                      <BatchPauseControl :batch="b" :busy="changingBatchState[b.id]" @change="changeBatchState(b.id, $event)" />
                      <div class="batch-delete-action">
                        <button type="button" class="btn history-delete"
                          :disabled="!canDeleteBatch(b) || !!changingBatchState[b.id] || !!deletion"
                          :aria-label="'删除批次 ' + (b.name || shortId(b.id))"
                          @click="askDeletion(b)"><Icon name="trash" :size="15" />删除批次</button>
                        <small v-if="!canDeleteBatch(b)">{{ deleteBlockedHint }}</small>
                      </div>
                      </div>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>
        </template>
        <ImportPanel
          v-else-if="route.view === 'import'"
          :providers="config.providers"
          :max-import-files="config.max_import_files"
          :concurrency="config.concurrency"
          :provider="provider"
          @provider="setProvider"
          @created="openBatch"
        />
        <SettingsPanel
          v-else-if="route.view === 'settings'"
          :providers="config.providers"
          :provider="provider"
          :concurrency="config.concurrency"
          @provider="setProvider"
          @concurrency="config.concurrency = $event"
          @reasoning="setReasoning"
        />
        <template v-else>
          <div v-if="loading && !detail" class="empty">正在读取批次…</div>
          <template v-else-if="detail">
            <PaperDetail
              :retry-provider="providerName(provider)"
              v-if="selectedPaper"
              :key="selectedPaper.id"
              :paper="selectedPaper"
              :back="path(detail.id)"
              :retrying="retrying"
              :changing-state="changingPaperState"
              @retry="retry"
              @pause="changePaperState('pause')"
              @resume="changePaperState('resume')"
            />
            <section v-else-if="route.paper" class="empty">
              <h2>当前批次没有这篇论文</h2>
              <a :href="path(detail.id)">返回批次</a>
            </section>
            <template v-else
              ><a class="back" href="#/"
                ><Icon name="back" :size="15" />批次工作台</a
              >
              <div class="page-title">
                <div>
                  <BatchNameEditor class="batch-heading-editor" :key="detail.id" :id="detail.id" :name="detail.name" @saved="nameSaved">
                    <h1 class="batch-title">{{ detail.name || "未命名批次" }}</h1>
                  </BatchNameEditor>
                  <p>
                    {{ detail.total }} 篇论文 <span class="sep">/</span> {{ formatDate(detail.created_at) }} 创建
                    <span class="sep">/</span>
                    {{ providerName(detail.provider) }}
                    <span class="sep">/</span>
                    <span class="mono">#{{ shortId(detail.id) }}</span>
                  </p>
                </div>
                <div class="history-actions">
                <BatchPauseControl :batch="detail" :busy="changingBatchState[detail.id]" @change="changeBatchState(detail.id, $event)" />
                <a
                  class="btn"
                  :href="
                    '/api/batches/' +
                    encodeURIComponent(detail.id) +
                    '/download'
                  "
                  download
                  ><Icon name="download" />下载批次 ZIP ·
                  {{ deliveredCount }} 篇<span v-if="detail.summary_count">（含 {{ detail.summary_count }} 篇筛选说明）</span></a
                >
                <button type="button" class="btn history-delete"
                  :disabled="!canDeleteBatch(detail) || !!changingBatchState[detail.id] || !!deletion"
                  :title="canDeleteBatch(detail) ? undefined : deleteBlockedHint"
                  @click="askDeletion(detail)"><Icon name="trash" :size="15" />删除此批次</button>
                </div>
              </div>
              <section class="panel">
                <div class="panel-head">
                  <h2>论文队列</h2>
                  <span class="muted"
                    >已结束 {{ completeCount }} /
                    {{ detail.papers.length }}</span
                  >
                </div>
                <div class="queue-filters" aria-label="按状态筛选">
                  <button
                    v-for="[key, name] in filters"
                    :key="key"
                    :aria-pressed="queueFilter === key"
                    @click="queueFilter = key"
                  >
                    {{ name }}
                    <span>{{
                      key === "all" ? detail.total : detail.counts[key] || 0
                    }}</span>
                  </button>
                </div>
                <div v-if="!visiblePapers.length" class="empty">
                  此状态下没有论文。
                </div>
                <div class="paper-row" v-for="p in visiblePapers" :key="p.id">
                  <div class="file-icon"><Icon name="file" /></div>
                  <div class="row-main">
                    <a class="row-title" :href="path(detail.id, p.id)">{{
                      p.name
                    }}</a>
                    <small v-if="p.events?.length">{{
                      eventSummary(p.events[p.events.length - 1])
                    }}</small>
                  </div>
                  <span class="stage">{{ stageName(p.stage) }}</span
                  ><span class="pill" :class="paperTone(p)">{{
                    batchPaperLabel(p)
                  }}</span
                  ><a
                    class="icon-link"
                    :href="path(detail.id, p.id)"
                    :aria-label="'查看 ' + p.name"
                    ><Icon name="chevron" :size="16"
                  /></a>
                </div>
              </section>
            </template>
          </template>
        </template>
      </main>
    </div>
  </div>
  <DeleteHistoryDialog v-if="deletion" v-bind="deletion" @deleted="finishDeletion" @cancel="cancelDeletion" />
</template>
