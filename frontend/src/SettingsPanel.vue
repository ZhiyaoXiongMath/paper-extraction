<script setup lang="ts">
import type { Provider, Probe } from "./contracts";
import { ref, watch } from "vue";
import { post, errorMessage } from "./api";
import Icon from "./Icon.vue";
const props = defineProps<{ providers: Provider[]; provider: string; concurrency?: number }>();
const emit = defineEmits(["provider", "concurrency"]);
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
  try {
    const r = await post<Probe>("/config/test", { provider: id, mode });
    tests.value[slot] = { ok: r.ok, message: r.message };
  } catch (e) {
    tests.value[slot] = { ok: false, message: errorMessage(e) };
  }
}
const modes = [
  ["text", "文本连接"],
  ["vision", "视觉连接"],
];
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
            :disabled="tests[p.id + '/' + mode]?.pending || !p.configured"
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
