<script setup lang="ts">
import type { Provider, Probe } from "./contracts";
import { ref } from "vue";
import { post, errorMessage } from "./api";
import Icon from "./Icon.vue";
defineProps<{ providers: Provider[]; provider: string }>();
const emit = defineEmits(["provider"]);
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
          p.id === "deepseek" ? "DEEPSEEK_API_KEY" : "ZAI_API_KEY"
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
