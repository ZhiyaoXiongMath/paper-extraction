"""Local configuration. Frozen batch settings contain model names, never keys."""
import base64
import io
import os
from pathlib import Path

from dotenv import dotenv_values
from PIL import Image, ImageDraw
from platformdirs import user_data_path

PROVIDERS = {
    "openai": ("OpenAI", "OPENAI_API_KEY", "OPENAI", "https://api.openai.com/v1", "gpt-5.4-mini", "gpt-5.4-mini"),
    "deepseek": ("DeepSeek", "DEEPSEEK_API_KEY", "DEEPSEEK", "https://api.deepseek.com", "deepseek-flash", "deepseek-flash"),
    "zhipu": ("智谱 GLM", "ZAI_API_KEY", "ZHIPU", "https://open.bigmodel.cn/api/paas/v4", "glm-5.3", "glm-4.6v"),
}


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


def provider_settings(provider: str, with_key=False):
    if provider not in PROVIDERS:
        raise ValueError("未知模型服务商")
    _, key, prefix, base, text, vision = PROVIDERS[provider]
    e = env()
    config = dict(provider=provider, base_url=e.get(prefix+"_BASE_URL") or base,
                  text_model=e.get(prefix+"_TEXT_MODEL") or text,
                  vision_model=e.get(prefix+"_VISION_MODEL") or vision, timeout=120)
    if with_key:
        config["api_key"] = e.get(key) or ""
    return config


def public_config():
    options = []
    for id, (label, key, *_) in PROVIDERS.items():
        config = provider_settings(id, with_key=True)
        options.append(dict(id=id, label=label, configured=bool(config["api_key"]),
                            text_model=config["text_model"], vision_model=config["vision_model"],
                            key_env=key))
    default = env().get("PAPER_EXTRACTION_PROVIDER") or "openai"
    return dict(providers=options, default_provider=default if default in PROVIDERS else "openai")


def probe(provider: str, mode: str):
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    config = provider_settings(provider, with_key=True)
    if not config["api_key"]:
        return {"ok": False, "message": "缺少 " + PROVIDERS[provider][1]}
    if mode not in ("text", "vision"):
        raise ValueError("测试类型应为 text 或 vision")
    model = ChatOpenAI(model=config[mode+"_model"], base_url=config["base_url"],
                       api_key=config["api_key"], timeout=45, max_retries=0)
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
