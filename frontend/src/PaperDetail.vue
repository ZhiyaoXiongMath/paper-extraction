<script setup lang="ts">
import type { Paper, Evidence } from "./contracts";
import { computed, ref, nextTick, watch } from "vue";
import { request, paperUrl, errorMessage } from "./api";
import {
  shortId,
  formatDate,
  paperOutcome,
  tone,
  stageName,
} from "./format.js";
import IntroductionReader from "./IntroductionReader.vue";
import Icon from "./Icon.vue";
import CandidatePanel from "./CandidatePanel.vue";
import MathText from "./MathText.vue";
import ReviewsPanel from "./ReviewsPanel.vue";
const props = defineProps<{ paper: Paper; back: string; retrying: boolean; retryProvider: string }>();
const emit = defineEmits(["retry"]);
const outcome = computed(() => paperOutcome(props.paper));
const tabSelected = ref(false);
const tab = ref(outcome.value.initialTab),
  evidence = ref<Evidence | null>(null),
  evidenceKey = ref(""),
  evError = ref(""),
  evLoading = ref(false),
  evPanel = ref<HTMLElement>();
const prefix = computed(() => paperUrl(props.paper.id));
const passed = computed(() => tone(props.paper.status) === "good");
const links = computed(() =>
  Object.fromEntries(
    (props.paper.links?.length
      ? props.paper.links
      : props.paper.draft?.evidence_links || []
    ).map((v) => [v.block_id, v.evidence_ids]),
  ),
);
const tabs = [
  ["reader", "引言阅读"],
  ["candidates", "候选贡献"],
  ["checks", "自动检查"],
  ["log", "运行记录"],
];
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
    if (id && passed.value && window.matchMedia("(min-width: 1001px)").matches)
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
  <div class="file-actions">
    <a
      v-if="paper.kind === 'pdf'"
      :href="prefix + '/source'"
      class="btn"
      target="_blank"
      rel="noopener"
      ><Icon name="file" :size="15" />打开原文</a
    >
    <template v-if="passed"
      ><a
        class="btn"
        :href="prefix + '/artifacts/introduction.pdf'"
        target="_blank"
        rel="noopener"
        ><Icon name="ext" :size="15" />成品 PDF</a
      ><a class="btn primary" :href="prefix + '/download'" download
        ><Icon name="download" :size="15" />下载单篇 ZIP</a
      ></template
    >
    <button
      v-if="tone(paper.status) === 'bad'"
      class="btn"
      :disabled="retrying"
      @click="emit('retry')"
    >
      <Icon name="refresh" :size="15" />{{
        retrying ? "正在提交…" : "用 " + retryProvider + " 重新提取"
      }}
    </button>
  </div>
  <div v-if="paper.reason" class="notice error" role="status">
    <Icon name="fail" />
    <div>
      <p>{{ outcome.summary }}</p>
      <details class="diagnostic">
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
    v-if="tab === 'reader' && firstSource"
    class="btn mobile-evidence"
    @click="evidenceKey ? closeEvidence() : openEvidence(firstSource)"
  >
    <Icon name="link" :size="14" />{{
      evidenceKey ? "返回读稿" : "对照原文证据"
    }}
  </button>
  <div class="detail-grid" :class="{ withEvidence: !!evidenceKey }">
    <section class="panel content-panel">
      <IntroductionReader v-if="tab === 'reader'" :draft="paper.draft" :stage="paper.stage"
        :links="links" :active-evidence="evidenceKey" @evidence="openEvidence" />
      <CandidatePanel v-if="tab === 'candidates'" :candidates="paper.candidates || []" @evidence="openEvidence" />
      <ReviewsPanel v-if="tab === 'checks'" :reviews="paper.reviews || []" :stage="paper.stage"
                    :draft="paper.draft" @evidence="openEvidence" />
      <template v-if="tab === 'log'"
        ><div class="panel-head">
          <h2>运行记录</h2>
          <small>按发生时间排列</small>
        </div>
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
      aria-label="原文证据"
    >
      <div class="panel-head">
        <h2>原文证据</h2>
        <button
          class="icon-link"
          aria-label="关闭原文证据"
          @click="closeEvidence"
        >
          <Icon name="close" :size="16" />
        </button>
      </div>
      <div class="panel-pad">
        <p v-if="evLoading" role="status">正在定位原文…</p>
        <div v-else-if="evError" role="alert">
          {{ evError }}
          <button class="text-btn" @click="openEvidence(evidenceKey)">
            重试
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
              · 第 {{ evidence.line }} 行</template
            ></small
          >
          <p
            v-if="
              evidence.transcribed_from_image ||
              evidence.evidence_type === 'machine_transcription'
            "
            class="source-label"
          >
            机器转写 · 可对照原页
          </p>
          <pre class="source-text">{{ evidence.text }}</pre>
          <a
            v-if="paper.kind === 'pdf'"
            class="btn"
            :href="sourceHref"
            target="_blank"
            rel="noopener"
            ><Icon name="ext" :size="15" />打开{{
              evidence.page ? "第 " + evidence.page + " 页" : "原文"
            }}</a
          ></template
        >
      </div>
    </aside>
  </div>
</template>
