<script setup lang="ts">
import type { ModelMode, Provider, Probe, ReasoningEffort, ReasoningSettings } from "./contracts";
import { ref, watch } from "vue";
import { post, errorMessage } from "./api";
import Icon from "./Icon.vue";
const props = defineProps<{ providers: Provider[]; provider: string; concurrency?: number }>();
const emit = defineEmits(["provider", "concurrency", "reasoning"]);
const count = ref("");
const saving = ref(false);
const saveError = ref("");
const savedMessage = ref("");
watch(() => props.concurrency, (n) => { count.value = n === undefined ? "" : String(n); }, { immediate: true });
async function saveConcurrency() {
  const value = Number(count.value);
  saveError.value = "";
  savedMessage.value = "";
  if (!Number.isSafeInteger(value) || value < 1) {
    saveError.value = "请填写大于零的整数。";
    return;
  }
  saving.value = true;
  try {
    const result = await post<{ concurrency: number }>("/config/concurrency", { concurrency: value });
    emit("concurrency", result.concurrency);
    savedMessage.value = "已保存，同时处理 " + result.concurrency + " 篇。";
  } catch (e) {
    saveError.value = errorMessage(e);
  } finally {
    saving.value = false;
  }
}
const tests = ref<Record<string, Partial<Probe> & { pending?: boolean }>>({});
async function test(id: string, mode: string) {
  const slot = id + "/" + mode;
  tests.value[slot] = { pending: true };
  const running = tests.value[slot];
  try {
    const r = await post<Probe>("/config/test", { provider: id, mode });
    if (tests.value[slot] === running) tests.value[slot] = { ok: r.ok, message: r.message };
  } catch (e) {
    if (tests.value[slot] === running) tests.value[slot] = { ok: false, message: errorMessage(e) };
  }
}
const modes = [
  ["text", "文本连接"],
  ["vision", "视觉连接"],
] as const;

const effortFields = [["text", "文本推理档位"], ["vision", "视觉推理档位"]] as const;
const effortNames: Record<ReasoningEffort, string> = {
  low: "低 (low)", medium: "中 (medium)", high: "高 (high)",
  xhigh: "很高 (xhigh)", max: "最高 (max)",
};
type EffortChanges = Partial<Record<ModelMode, ReasoningEffort>>;
const efforts = ref<Record<string, EffortChanges>>({});
const effortStatus = ref<Record<string, { pending?: boolean; message?: string; error?: string }>>({});

function setEffort(p: Provider, mode: ModelMode, event: Event) {
  const value = (event.target as HTMLSelectElement).value as ReasoningEffort;
  const changes = { ...efforts.value[p.id] };
  if (value === p.reasoning?.[mode]?.effort) {
    delete changes[mode];
  } else {
    changes[mode] = value;
  }
  efforts.value[p.id] = changes;
  delete effortStatus.value[p.id];
}

function effortDirty(id: string) {
  return Object.keys(efforts.value[id] ?? {}).length > 0;
}

function canSetEffort(p: Provider) {
  return effortFields.some(([mode]) => p.reasoning?.[mode]?.choices.length);
}

async function saveEfforts(id: string) {
  if (effortStatus.value[id]?.pending || !effortDirty(id)) return;
  effortStatus.value[id] = { pending: true };
  try {
    const result = await post<{ reasoning: ReasoningSettings }>(`/config/${id}/reasoning`, efforts.value[id]);
    emit("reasoning", id, result.reasoning);
    delete efforts.value[id];
    for (const [mode] of modes) delete tests.value[id + "/" + mode];
    effortStatus.value[id] = { message: "已保存推理档位。" };
  } catch (e) {
    effortStatus.value[id] = { error: errorMessage(e) };
  }
}
</script>
<template>
  <section class="page-title">
    <div>
      <h1>模型设置</h1>
      <p>为新批次选择模型服务，分别验证文本提取与页面视觉检查。</p>
    </div>
  </section>
  <section v-if="concurrency !== undefined" class="panel concurrency-panel" aria-label="批次运行设置">
    <div class="panel-head"><h2>批次运行设置</h2></div>
    <form class="panel-pad" @submit.prevent="saveConcurrency">
      <label class="control" for="concurrent-papers">同时处理论文数</label>
      <div class="concurrency-input">
        <input id="concurrent-papers" v-model="count" type="number" min="1" step="1" required
          :disabled="saving" aria-describedby="concurrency-help"
          @input="savedMessage = ''; saveError = ''" />
        <button type="submit" class="btn" :disabled="saving || Number(count) === concurrency">
          {{ saving ? "保存中…" : "保存" }}
        </button>
      </div>
      <p id="concurrency-help" class="config-help">对所有批次立即生效。调低后，正在处理的论文会继续完成。</p>
      <p v-if="savedMessage" class="test-result good" role="status">{{ savedMessage }}</p>
      <p v-if="saveError" class="test-result bad" role="alert">{{ saveError }}</p>
    </form>
  </section>
  <div class="models">
    <article class="panel model" v-for="p in providers" :key="p.id">
      <div class="model-head">
        <span class="file-icon"><Icon name="chip" :size="22" /></span>
        <h2>{{ p.label }}</h2>
        <span class="pill" :class="p.configured ? 'good' : 'neutral'">{{
          p.configured ? "已配置" : "未配置"
        }}</span>
      </div>
      <dl class="model-list">
        <div>
          <dt>文本模型</dt>
          <dd>{{ p.text_model || "—" }}</dd>
        </div>
        <div>
          <dt>视觉模型</dt>
          <dd>{{ p.vision_model || "—" }}</dd>
        </div>
      </dl>
      <form v-if="p.reasoning && Object.keys(p.reasoning).length" class="reasoning-form" @submit.prevent="saveEfforts(p.id)">
        <div class="reasoning-fields">
          <template v-for="[mode, text] in effortFields" :key="mode">
            <label v-if="p.reasoning[mode]?.choices.length" class="control">
              <span>{{ text }}</span>
              <select :value="efforts[p.id]?.[mode] ?? p.reasoning[mode]?.effort" :disabled="effortStatus[p.id]?.pending"
                :aria-describedby="'reasoning-help-' + p.id" @change="setEffort(p, mode, $event)">
                <option v-for="choice in p.reasoning[mode]?.choices" :key="choice" :value="choice">{{ effortNames[choice] }}</option>
              </select>
            </label>
            <div v-else-if="p.reasoning[mode]" class="control">
              <span>{{ mode === "text" ? "文本思考" : "视觉思考" }}</span>
              <p class="config-help">已开启，模型未提供强度档位。</p>
            </div>
          </template>
        </div>
        <div v-if="canSetEffort(p)" class="reasoning-actions">
          <p :id="'reasoning-help-' + p.id" class="config-help">档位用于新批次及连接测试。</p>
          <button type="submit" class="btn" :disabled="effortStatus[p.id]?.pending || !effortDirty(p.id)">
            {{ effortStatus[p.id]?.pending ? "保存中…" : "保存推理档位" }}
          </button>
        </div>
        <p v-if="effortStatus[p.id]?.message" class="test-result good" role="status">{{ effortStatus[p.id].message }}</p>
        <p v-if="effortStatus[p.id]?.error" class="test-result bad" role="alert">{{ effortStatus[p.id].error }}</p>
      </form>
      <p v-if="!p.configured" class="config-help">
        请在后端环境或项目私有配置中设置
        <code>{{
          p.key_env
        }}</code>
        并重新启动服务。
      </p>
      <label class="radio-row"
        ><input
          type="radio"
          name="provider"
          :checked="provider === p.id"
          @change="emit('provider', p.id)"
        />作为新批次的模型服务</label
      >
      <div class="model-tests">
        <div v-for="[mode, text] in modes" :key="mode">
          <button
            class="btn"
            :disabled="tests[p.id + '/' + mode]?.pending || !p.configured || effortDirty(p.id) || effortStatus[p.id]?.pending"
            @click="test(p.id, mode)"
          >
            <Icon name="activity" :size="15" />{{
              tests[p.id + "/" + mode]?.pending ? "测试中…" : "测试" + text
            }}
          </button>
          <div
            v-if="tests[p.id + '/' + mode]?.message"
            role="status"
            class="test-result"
            :class="tests[p.id + '/' + mode].ok ? 'good' : 'bad'"
          >
            {{ tests[p.id + "/" + mode].message }}
          </div>
        </div>
      </div>
    </article>
  </div>
  <p v-if="!providers.length" class="empty">模型服务信息暂未加载。</p>
</template>

<style scoped>
.reasoning-form { border-top: 1px solid var(--line); padding-top: 14px; }
.reasoning-fields { display: grid; grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr)); gap: 12px; }
.reasoning-actions { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 8px; }
.reasoning-actions .config-help { margin: 4px 0; }
</style>
