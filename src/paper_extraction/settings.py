"""Local configuration. Frozen batches carry model preferences without credentials."""
import base64
import io
import os
import re
from pathlib import Path

from dotenv import dotenv_values
from PIL import Image, ImageDraw
from platformdirs import user_data_path

PROVIDERS = {
    "openai": ("OpenAI", "OPENAI_API_KEY", "OPENAI", "https://api.openai.com/v1", "gpt-5.4-mini", "gpt-5.4-mini"),
    "deepseek": ("DeepSeek", "DEEPSEEK_API_KEY", "DEEPSEEK", "https://api.deepseek.com", "deepseek-flash", "deepseek-flash"),
    "zhipu": ("智谱 GLM", "ZAI_API_KEY", "ZHIPU", "https://open.bigmodel.cn/api/paas/v4", "glm-5.3", "glm-4.6v"),
}

DEFAULT_REASONING_EFFORT = "medium"
MODEL_MODES = ("text", "vision")
REASONING_EFFORTS = ("low", "medium", "high", "xhigh", "max")
THINKING_LEVELS = ("low", "high", "max")
DEEPSEEK_THINKING_MODELS = {"deepseek-flash", "deepseek-v4-flash", "deepseek-v4-pro",
                           "deepseek-v4-flash-vision-exp"}
GLM_EFFORT_MODELS = {"glm-5.2", "glm-5.3", "glm-5.3-flash", "glm-5.3-flashx", "glm-5.3-flashx-legal"}
GLM_VISION_THINKING_MODELS = {"glm-4.6v", "glm-4.6v-flashx", "glm-5v-turbo"}


def openai_reasoning_choices(model: str) -> tuple[str, ...]:
    """Use the documented GPT model choices and leave unrelated models untouched."""
    if re.match(r"^gpt-6(?:\.\d+)?-(?:astra|sol|luna|terra)(?:-|$)", model):
        return REASONING_EFFORTS
    if re.match(r"^gpt-5\.(?:2|3|4|5|6)(?:-|$)", model) and "-pro" not in model:
        return REASONING_EFFORTS[:-1]
    if re.fullmatch(r"gpt-5(?:-(?:mini|nano))?(?:-\d{4}-\d{2}-\d{2})?", model):
        return REASONING_EFFORTS[:3]
    return ()


def reasoning_choices(provider: str, model: str) -> tuple[str, ...]:
    model = model.lower()
    if provider == "openai":
        return openai_reasoning_choices(model)
    if provider == "deepseek" and model in DEEPSEEK_THINKING_MODELS:
        return THINKING_LEVELS
    if provider in {"zhipu", "glm"} and model in GLM_EFFORT_MODELS:
        return THINKING_LEVELS
    return ()


def thinking_enabled(provider: str, model: str) -> bool:
    model = model.lower()
    return (provider == "deepseek" and model in DEEPSEEK_THINKING_MODELS
            or provider in {"zhipu", "glm"} and model in GLM_EFFORT_MODELS | GLM_VISION_THINKING_MODELS)


def reasoning_effort(config: dict, mode: str, preference=None) -> str | None:
    provider = config["provider"]
    choices = reasoning_choices(provider, config.get(mode + "_model", ""))
    if not choices:
        return None
    if preference in choices:
        return preference
    # DeepSeek's documented default is high. Start GLM's always-on 5.3 at low.
    return {"openai": DEFAULT_REASONING_EFFORT, "deepseek": "high"}.get(provider, "low")


def apply_reasoning_preferences(config: dict, reasoning_efforts=None) -> None:
    for mode in MODEL_MODES:
        effort = reasoning_effort(config, mode, (reasoning_efforts or {}).get(mode))
        if effort is not None:
            config[mode + "_reasoning_effort"] = effort


def reasoning_config(config: dict) -> dict:
    return {
        mode: dict(effort=config.get(mode + "_reasoning_effort"),
                   choices=reasoning_choices(config["provider"], config[mode + "_model"]))
        for mode in MODEL_MODES if mode + "_reasoning_effort" in config
        or thinking_enabled(config["provider"], config[mode + "_model"])
    }


def chat_options(config: dict, mode: str, *, timeout: int, max_retries: int) -> dict:
    model = config[mode + "_model"]
    options = dict(model=model, api_key=config["api_key"], base_url=config["base_url"],
                   timeout=timeout, max_retries=max_retries)
    # Old batch snapshots can lack effort fields. Keep the legacy OpenAI path,
    # and use a supported default when enabling thinking on recognized models.
    effort = config.get(mode + "_reasoning_effort")
    if thinking_enabled(config["provider"], model):
        options["extra_body"] = {"thinking": {"type": "enabled"}}
        effort = reasoning_effort(config, mode, effort)
    if effort is not None:
        options["reasoning_effort"] = effort
    return options


def env():
    return {**dotenv_values(Path(os.environ.get("PAPER_EXTRACTION_ENV_FILE", ".env"))), **os.environ}


def workbench_limits(*, concurrency=None):
    settings = env()

    def integer(name, default, minimum):
        try:
            value = int(settings.get(name) or str(default))
        except ValueError:
            value = minimum - 1
        if value < minimum:
            raise ValueError(f"{name} 应为大于或等于 {minimum} 的整数。")
        return value

    return {
        "concurrency": concurrency if concurrency is not None else integer("PAPER_EXTRACTION_CONCURRENCY", 2, 1),
        "max_import_files": integer("PAPER_EXTRACTION_MAX_IMPORT_FILES", 1000, 0),
    }


def data_home():
    value = env().get("PAPER_EXTRACTION_HOME")
    return Path(value or user_data_path("paper-extraction", appauthor=False)).expanduser().resolve()


def provider_settings(provider: str, with_key=False, *, reasoning_efforts=None):
    if provider not in PROVIDERS:
        raise ValueError("未知模型服务商")
    _, key, prefix, base, text, vision = PROVIDERS[provider]
    e = env()
    config = dict(provider=provider, base_url=e.get(prefix+"_BASE_URL") or base,
                  text_model=e.get(prefix+"_TEXT_MODEL") or text,
                  vision_model=e.get(prefix+"_VISION_MODEL") or vision, timeout=120)
    apply_reasoning_preferences(config, reasoning_efforts)
    if with_key:
        config["api_key"] = e.get(key) or ""
    return config


def public_config(reasoning_efforts=None):
    options = []
    for id, (label, key, *_) in PROVIDERS.items():
        config = provider_settings(id, with_key=True, reasoning_efforts=(reasoning_efforts or {}).get(id))
        public = dict(id=id, label=label, configured=bool(config["api_key"]),
                      text_model=config["text_model"], vision_model=config["vision_model"], key_env=key,
                      reasoning=reasoning_config(config))
        options.append(public)
    default = env().get("PAPER_EXTRACTION_PROVIDER") or "openai"
    return dict(providers=options, default_provider=default if default in PROVIDERS else "openai")


def probe(provider: str, mode: str, *, reasoning_efforts=None):
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    config = provider_settings(provider, with_key=True, reasoning_efforts=reasoning_efforts)
    if not config["api_key"]:
        return {"ok": False, "message": "缺少 " + PROVIDERS[provider][1]}
    if mode not in MODEL_MODES:
        raise ValueError("测试类型应为 text 或 vision")
    model = ChatOpenAI(**chat_options(config, mode, timeout=45, max_retries=0))
    content = "Return just the word READY."
    if mode == "vision":
        import secrets
        number = str(secrets.randbelow(9000) + 1000)
        image = Image.new("RGB", (220, 80), "white")
        ImageDraw.Draw(image).text((15, 8), number, fill="black", font_size=48)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data = base64.b64encode(buf.getvalue()).decode()
        content = [{"type":"text","text":"Read the four digits in the image. Return only the digits."},
                   {"type":"image_url","image_url":{"url": "data:image/png;base64," + data}}]
    try:
        response = model.invoke([HumanMessage(content=content)])
        expected = number if mode == "vision" else "READY"
        ok = expected in str(response.content).upper()
        return {"ok": ok, "message": "连接正常" if ok else "收到响应，未通过能力测试"}
    except Exception as exc:
        return {"ok": False, "message": f"{type(exc).__name__}，请检查密钥、模型名与网络连接"}
