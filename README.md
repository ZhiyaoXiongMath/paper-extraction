# Paper Extraction

[English](README.en.md) · [发行版本](https://github.com/ZhiyaoXiongMath/paper-extraction/releases)

本地数学论文批处理工作台。导入 PDF 或 TeX，Agent 从原文选择历史定理、主要贡献与必要定义，按 Introduction v13 生成逐篇英文 TeX 与 PDF。中文网页支持点击生成稿中的定理，打开原论文页或源文件行号。

## 安装和启动

需要 Python 3.11–3.14 及可运行 `pdflatex`、`kpsewhich` 的 TeX Live。完整的 TeX Live 安装包含所需宏包。精简安装须包含 newtx、txfonts、amsmath、amsthm、mathrsfs、enumitem、needspace、microtype、etoolbox、hyperref、titlesec 及依赖。

发行 wheel 内置网页。下载 wheel 后，在 Python 虚拟环境中运行：

```shell
python -m pip install paper_extraction-0.1.0-py3-none-any.whl
paper-extraction --version
paper-extraction example compactness
```

从仓库运行需安装 [uv](https://docs.astral.sh/uv/) 和 Node.js 22.12 或更新版。使用以下命令构建网页并安装锁定的 Python 依赖：

```shell
git clone https://github.com/ZhiyaoXiongMath/paper-extraction.git
cd paper-extraction
npm --prefix frontend ci
npm --prefix frontend run build
uv sync --locked --python 3.12
uv run paper-extraction doctor
```

首次运行时，把 [.env.example](.env.example) 复制到**启动目录**并命名为 `.env`。默认使用 OpenAI，填写自己的 `OPENAI_API_KEY`。已有配置文件时，在原文件中添加该变量并设置 `PAPER_EXTRACTION_PROVIDER=openai`。DeepSeek 和智谱也可在网页的模型设置中选择。环境变量同样有效。若配置文件在其他位置，请用 `paper-extraction --env-file 配置路径 serve`。论文片段和需要识别的页图会发送给选定的模型服务。

```shell
paper-extraction doctor --live --provider openai
paper-extraction serve
```

从源码安装时在上述命令前加 `uv run`。用浏览器打开 **http://127.0.0.1:8765**。可用 `serve --port 8766` 更换端口。在“模型设置”中，可按服务商分别保存模型支持的文本和视觉推理档位。OpenAI 支持档位时默认使用 `medium`，DeepSeek Flash 默认使用 `high`，GLM-5.3 默认使用 `low`。GLM-4.6V 启用思考，由模型控制推理强度。新批次记录当时的设置，连接测试使用已保存的档位发起真实模型请求。

## 一次提取

1. 在“新建批次”选择多个文件或整个论文目录，核对预览清单，为有多个入口的 TeX 包选择主文件。
2. 选择模型并开始提取。默认并发两篇。关闭网页后后端继续工作；重新启动服务时恢复未完成任务。
3. 点击论文查看生成稿、证据和检查结果。标题下的“下载 .tex”和“下载 .pdf”分别提供当前轮生成的文件。版式检查或视觉复核失败时，已生成的可打开 PDF 会标为“未通过 · 第 N 轮草稿”，可下载对照失败原因。草稿也可单独重试。
4. 批次 ZIP 以批次名命名，每篇论文的 PDF 和可独立编译的 TeX 放在各自的目录里。单篇和批次下载中的论文文件均以输入文件名加 `-extraction` 命名。多文件 TeX 项目使用所选主文件名。批次清单也会列出排队、失败或仍在处理的项目。
5. 当批次中的每篇论文都已暂停或结束，可以在工作台或批次详情单独删除该批次。“清除已结束批次”用于批量清除所有论文均已结束的批次。确认后，批次记录、草稿和成品会一并移除。请先下载需要保留的文件。
6. 论文详情提供“暂停提取”和“继续提取”。排队中的任务立即暂停。运行中的任务在当前环节完成后保存检查点，继续时从下一环节接着处理。

**示例。** 选择 [examples/compactness](examples/compactness) 文件夹。主文件引用另一份章节文件，演示如何从结论跳到源文件行号。这是基于经典 Arzelà–Ascoli 定理的教学样本。使用模型实际生成时，数学取舍和措辞依本次机器复核而定。成功的 ZIP 包含：

```text
001-compactness/
  main-extraction.tex
  main-extraction.pdf
manifest.json
```

## 工作流与判定

LangGraph 为每篇论文保存独立的 SQLite 检查点。LangChain 的模型调用与只读工具让 Agent 搜索原文、读取段落、查看原页。每次启动或主动重试最多写三稿，包含初稿。前两稿按需复核、定向修订或补查原文，通过检查即可交付。第三稿直接双遍编译并确认 PDF 可以打开。若 TeX 编译报错，程序尝试复用原文依据相同的上一稿段落修复，或交付此前经验证的整稿。没有可复用版本时，程序用已有内容编译简明版；本机缺少编译环境时生成可读 PDF 并注明来源。定稿阶段无需再次调用模型。工作台提供 PDF 和内嵌前导内容的成品 TeX，检查结果单独记录。

成果包含正文和独立参考文献页。模型依据原文筛选定理、历史比较和必要定义，并用连接段交代背景、数学比较及结果适用范围。正文通常以三页内为目标，内容确有需要时最多五页。词数是写作建议，实际块高、公式和页数由编译后的检查验证。来源不足或版式无法满足要求时，批次会说明原因。[原始规则](src/paper_extraction/profile_assets/intro-rules-v13.txt)

## 配置与开发

| 配置 | 作用 |
|---|---|
| `PAPER_EXTRACTION_PROVIDER` | 默认 `openai`，可选 `deepseek`、`zhipu` |
| `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` / `ZAI_API_KEY` | 所选服务商的独立 API 密钥 |
| `OPENAI_TEXT_MODEL` / `OPENAI_VISION_MODEL` | OpenAI 文本与看图模型，均默认 `gpt-5.4-mini` |
| `DEEPSEEK_TEXT_MODEL` / `DEEPSEEK_VISION_MODEL` | DeepSeek 文本及图像模型，默认 `deepseek-flash` |
| `ZHIPU_TEXT_MODEL` / `ZHIPU_VISION_MODEL` | 智谱文本及图像模型，默认 `glm-5.3` / `glm-4.6v` |
| `OPENAI_BASE_URL` / `DEEPSEEK_BASE_URL` / `ZHIPU_BASE_URL` | 对应模型服务的 API 地址 |
| `PAPER_EXTRACTION_HOME` | 本机论文快照、成品和任务数据库目录，默认使用平台用户数据目录 |
| `PAPER_EXTRACTION_CONCURRENCY` | 同时处理的论文数量，默认 2，可填正整数。工作台保存的并发设置优先，并立即生效 |
| `PAPER_EXTRACTION_MAX_IMPORT_FILES` | 单次上传的文件数，默认 1000，填 0 表示不限文件数。修改后重启服务。每个 PDF、TeX 主文件、章节和附件各计一个文件 |

测试运行 `uv run pytest -q` 和 `npm --prefix frontend test`。生成安装包先构建网页，再运行 `uv build --wheel`。FastAPI 接口说明位于本地服务的 `/docs`。测试中的模型回放用于验证执行路径，真实模型验收使用独立配置。

项目采用 [MIT 许可证](LICENSE)。第三方组件见[许可说明](THIRD_PARTY.md)。
