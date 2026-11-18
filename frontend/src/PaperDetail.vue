<script setup lang="ts">
import type { Paper, Evidence } from "./contracts";
import { computed, ref, nextTick, watch } from "vue";
import { request, paperUrl, errorMessage } from "./api";
import { paperDownloads, visibleFailureReasons } from "./paper-downloads.js";
import {
  shortId,
  formatDate,
  paperOutcome,
  tone,
  stageName,
} from "./format.js";
import IntroductionReader from "./IntroductionReader.vue";
import HistoryDrafts from "./HistoryDrafts.vue";
import Icon from "./Icon.vue";
import CandidatePanel from "./CandidatePanel.vue";
import MathText from "./MathText.vue";
import ReviewsPanel from "./ReviewsPanel.vue";
import UsagePanel from "./UsagePanel.vue";
const props = defineProps<{ paper: Paper; back: string; retrying: boolean; retryProvider: string; changingState: boolean }>();
const emit = defineEmits(["retry", "pause", "resume"]);
const outcome = computed(() => paperOutcome(props.paper));
const tabSelected = ref(false);
const tab = ref(outcome.value.initialTab),
  evidence = ref<Evidence | null>(null),
  evidenceKey = ref(""),
  evError = ref(""),
  evLoading = ref(false),
  evPanel = ref<HTMLElement>();
const readerView = computed(() => tab.value === "reader");
const evidenceLabels = computed(() => readerView.value ? {
  title: "Source evidence",
  close: "Close source evidence",
  loading: "Locating the source…",
  retry: "Retry",
  transcription: "Machine transcription · Check the original page",
  line: (number: number) => `Line ${number}`,
  openPdf: (page: number | undefined) => page ? `Open page ${page}` : "Open source PDF",
} : {
  title: "原文证据",
  close: "关闭原文证据",
  loading: "正在定位原文…",
  retry: "重试",
  transcription: "机器转写 · 可对照原页",
  line: (number: number) => `第 ${number} 行`,
  openPdf: (page: number | undefined) => page ? `打开第 ${page} 页` : "打开原文",
});
const prefix = computed(() => paperUrl(props.paper.id));
const downloads = computed(() => paperDownloads(props.paper, prefix.value));
const summary = computed(() => props.paper.delivery?.kind === "summary" ? props.paper.delivery.summary : undefined);
const failureReasons = computed(() => visibleFailureReasons(props.paper));
const published = computed(() => tone(props.paper.status) === "good");
const links = computed(() =>
  Object.fromEntries(
    (props.paper.links?.length
      ? props.paper.links
      : props.paper.draft?.evidence_links || []
    ).map((v) => [v.block_id, v.evidence_ids]),
  ),
);
const tabs = computed(() => [
  summary.value ? ["summary", "筛选说明"] : ["reader", "引言阅读"],
  ["candidates", "候选贡献"],
  ...(summary.value ? [] : [["checks", "自动检查"]]),
  ["log", "运行记录"],
]);
const sourceHref = computed(
  () =>
    prefix.value +
    "/source" +
    (evidence.value?.page ? "#page=" + Number(evidence.value.page) : ""),
);
const firstSource = computed(
  () => links.value["groups.0.theorem"]?.[0] || links.value["paragraph"]?.[0],
);
let seq = 0;
async function openEvidence(id: string, focus = true) {
  const version = ++seq;
  evidenceKey.value = id;
  evidence.value = null;
  evError.value = "";
  evLoading.value = true;
  await nextTick();
  if (focus) evPanel.value?.focus();
  try {
    const data = await request<Evidence>(
      "/papers/" +
        encodeURIComponent(props.paper.id) +
        "/evidence/" +
        encodeURIComponent(id),
    );
    if (seq === version) evidence.value = data;
  } catch (e) {
    if (seq === version) evError.value = errorMessage(e);
  } finally {
    if (seq === version) evLoading.value = false;
  }
}
function closeEvidence() {
  ++seq;
  evidenceKey.value = "";
  evidence.value = null;
  evError.value = "";
}
function changeTab(t: string) {
  tabSelected.value = true;
  tab.value = t;
  closeEvidence();
}
watch(
  () => outcome.value.initialTab,
  (next) => {
    if (!tabSelected.value) tab.value = next;
  },
);
watch(
  firstSource,
  (id) => {
    if (id && published.value && window.matchMedia("(min-width: 1001px)").matches)
      openEvidence(id, false);
  },
  { immediate: true },
);
</script>
<template>
  <a class="back" :href="back"><Icon name="back" :size="15" />返回批次</a>
  <section class="page-title detail-heading">
    <div>
      <div class="kicker">
        <span class="kind">{{ paper.kind || "论文" }}</span
        ><span class="mono">#{{ shortId(paper.id) }}</span>
      </div>
      <h1>{{ paper.draft?.title || paper.name }}</h1>
      <p v-if="paper.draft?.authors?.length">
        {{ paper.draft.authors.join(" · ") }}
      </p>
    </div>
    <span class="pill" :class="tone(paper.status)">{{ outcome.title }}</span>
  </section>
  <section class="paper-downloads" aria-label="论文文件下载">
    <div class="download-context">
      <strong>{{ downloads.label }}</strong>
      <span v-if="downloads.recompiled">按本轮保存稿重新编译</span>
      <span v-else-if="downloads.restored">已恢复本轮保存稿</span>
      <span v-if="downloads.missingPdf">{{ downloads.missingPdf }}</span>
    </div>
    <div class="download-buttons">
      <a v-if="downloads.markdown" class="btn primary" :href="downloads.markdown" download>
        <Icon name="download" :size="15" />下载说明 .md</a>
      <template v-else>
      <a v-if="downloads.tex" class="btn primary" :href="downloads.tex" download>
        <Icon name="download" :size="15" />下载 .tex</a>
      <button v-else class="btn" disabled><Icon name="download" :size="15" />下载 .tex</button>
      <a v-if="downloads.pdf" class="btn" :href="downloads.pdf" download>
        <Icon name="download" :size="15" />下载 .pdf</a>
      <button v-else class="btn" disabled><Icon name="download" :size="15" />下载 .pdf</button>
      </template>
      <a v-if="published && !summary" class="btn" :href="prefix + '/download'" download>
        <Icon name="download" :size="15" />下载单篇 ZIP</a>
    </div>
    <p v-if="published && paper.delivery?.note && !summary" class="delivery-note">{{ paper.delivery.note }}</p>
  </section>
  <div class="file-actions" v-if="paper.kind === 'pdf' || ['failed','queued','running','pausing','paused'].includes(paper.status)">
    <a v-if="paper.kind === 'pdf'" :href="prefix + '/source'" class="btn" target="_blank" rel="noopener">
      <Icon name="file" :size="15" />打开原文</a>
    <button
      v-if="paper.status === 'failed'"
      class="btn"
      :disabled="retrying"
      @click="emit('retry')"
    >
      <Icon name="refresh" :size="15" />{{
        retrying ? "正在提交…" : "用 " + retryProvider + " 重试此篇"
      }}
    </button>
    <button v-if="['queued','running','pausing'].includes(paper.status)" type="button" class="btn"
      :disabled="changingState || paper.status === 'pausing'" @click="emit('pause')">
      {{ paper.status === 'pausing' ? '当前环节完成后暂停' : '暂停提取' }}
    </button>
    <button v-if="paper.status === 'paused'" type="button" class="btn"
      :disabled="changingState" @click="emit('resume')">继续提取</button>
  </div>
  <HistoryDrafts :versions="paper.draft_artifacts || []" />
  <div v-if="paper.status === 'paused' || paper.status === 'cancelled'" class="notice" role="status">
    {{ paper.reason || (paper.status === 'paused' ? '已暂停，可从检查点继续。' : '用户已终止此任务。') }}
  </div>
  <div v-else-if="!published && (paper.reason || failureReasons.length)" class="notice error" role="status">
    <Icon name="fail" />
    <div>
      <p>{{ outcome.summary }}</p>
      <ul v-if="failureReasons.length" class="failure-reasons">
        <li v-for="item in failureReasons" :key="item"><MathText :text="item" /></li>
      </ul>
      <details v-if="paper.reason" class="diagnostic">
        <summary>查看技术诊断</summary>
        <p>{{ paper.reason }}</p>
      </details>
    </div>
  </div>
  <div class="tabs" aria-label="论文详情栏目">
    <button
      v-for="[id, name] in tabs"
      :key="id"
      :aria-pressed="id === tab"
      @click="changeTab(id)"
    >
      {{ name }}
    </button>
  </div>
  <button
    v-if="readerView && firstSource"
    class="btn mobile-evidence"
    lang="en"
    @click="evidenceKey ? closeEvidence() : openEvidence(firstSource)"
  >
    <Icon name="link" :size="14" />{{
      evidenceKey ? "Back to introduction" : "View source evidence"
    }}
  </button>
  <div class="detail-grid" :class="{ withEvidence: !!evidenceKey }">
    <section class="panel content-panel">
      <article v-if="tab === 'summary' && summary" class="panel-pad">
        <h2>研究问题</h2>
        <MathText :text="summary.question" />
        <h2>筛选结论</h2>
        <p>{{ paper.delivery?.note }}</p>
        <MathText :text="summary.reason" />
        <template v-if="summary.evidence.length">
          <h2>原文依据</h2>
          <div class="citations">
            <button v-for="entry in summary.evidence" :key="entry.id" :class="{ active: evidenceKey === entry.id }"
              @click="openEvidence(entry.id)"><Icon name="link" :size="12" />{{ entry.label }}</button>
          </div>
        </template>
      </article>
      <IntroductionReader v-if="readerView" :draft="paper.draft"
        :links="links" :active-evidence="evidenceKey" @evidence="openEvidence" />
      <CandidatePanel v-if="tab === 'candidates'" :candidates="paper.candidates || []" @evidence="openEvidence" />
      <ReviewsPanel v-if="tab === 'checks'" :reviews="paper.reviews || []" :stage="paper.stage"
                    :draft="paper.draft" @evidence="openEvidence" />
      <template v-if="tab === 'log'"
        ><div class="panel-head">
          <h2>运行记录</h2>
          <small>按发生时间排列</small>
        </div>
        <UsagePanel :key="paper.id" :paper-id="paper.id" :event-count="paper.events?.length || 0" :status="paper.status" />
        <div v-if="paper.events?.length" class="timeline">
          <article v-for="(event, i) in paper.events" :key="i">
            <span class="timeline-dot"></span>
            <div>
              <b>{{ stageName(event.stage) }}</b
              ><time>{{ formatDate(event.time) }}</time>
              <details v-if="event.data && Object.keys(event.data).length">
                <summary>查看详情</summary>
                <pre>{{ JSON.stringify(event.data, null, 2) }}</pre>
              </details>
            </div>
          </article>
        </div>
        <div v-else class="empty">尚无运行事件。</div></template
      >
    </section>
    <aside
      v-if="evidenceKey"
      class="panel evidence"
      tabindex="-1"
      ref="evPanel"
      :lang="readerView ? 'en' : undefined"
      :aria-label="evidenceLabels.title"
    >
      <div class="panel-head">
        <h2>{{ evidenceLabels.title }}</h2>
        <button
          class="icon-link"
          :aria-label="evidenceLabels.close"
          @click="closeEvidence"
        >
          <Icon name="close" :size="16" />
        </button>
      </div>
      <div class="panel-pad">
        <p v-if="evLoading" role="status">{{ evidenceLabels.loading }}</p>
        <div v-else-if="evError" role="alert">
          {{ readerView ? "Unable to load source evidence." : evError }}
          <button class="text-btn" @click="openEvidence(evidenceKey)">
            {{ evidenceLabels.retry }}
          </button>
        </div>
        <template v-else-if="evidence"
          ><div class="source-label">
            <Icon name="file" />{{
              evidence.label || evidence.file || evidence.id
            }}
          </div>
          <small v-if="evidence.file"
            >{{ evidence.file }}
            <template v-if="evidence.line">
              · {{ evidenceLabels.line(evidence.line) }}</template
            ></small
          >
          <p
            v-if="
              evidence.transcribed_from_image ||
              evidence.evidence_type === 'machine_transcription'
            "
            class="source-label"
          >
            {{ evidenceLabels.transcription }}
          </p>
          <pre class="source-text">{{ evidence.text }}</pre>
          <a
            v-if="paper.kind === 'pdf'"
            class="btn"
            :href="sourceHref"
            target="_blank"
            rel="noopener"
            ><Icon name="ext" :size="15" />{{ evidenceLabels.openPdf(evidence.page) }}</a
          ></template
        >
      </div>
    </aside>
  </div>
</template>
<style scoped>
.paper-downloads { margin: 0 0 15px; }
.download-context, .download-buttons { display: flex; align-items: center; gap: 9px; flex-wrap: wrap; }
.download-context { margin-bottom: 8px; font-size: 13px; color: var(--muted); }
.download-context strong { color: var(--ink, #26372e); }
.delivery-note { margin: 8px 0 0; color: var(--muted); font-size: 13px; }
.download-buttons .btn:disabled { cursor: default; opacity: .52; }
.failure-reasons { margin: .45rem 0; padding-inline-start: 1.2rem; }
.failure-reasons li + li { margin-top: .25rem; }
</style>
