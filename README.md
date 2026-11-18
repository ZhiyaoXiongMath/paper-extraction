# Paper Extraction

[English](README.en.md) · [发行版本](https://github.com/ZhiyaoXiongMath/paper-extraction/releases)

本地数学论文批处理工作台。导入 PDF 或 TeX，Agent 从原文选择历史定理、主要贡献与必要定义，按 Introduction v13 生成逐篇英文 TeX 与 PDF。中文网页支持点击生成稿中的定理，打开原论文页或源文件行号。

## 安装和启动

需要 Python 3.11–3.14 及可运行 `pdflatex`、`kpsewhich` 的 TeX Live。完整的 TeX Live 安装包含所需宏包。精简安装须包含 newtx、amsmath、amsthm、mathrsfs、enumitem、needspace、microtype、etoolbox、hyperref、titlesec 及依赖。

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

复制仓库中的 `.env.example` 为 **启动目录**中的 `.env`，填入自己的 `DEEPSEEK_API_KEY` 或智谱 `ZAI_API_KEY`。也可以直接在环境中设置变量。若配置文件在其他位置，请用 `paper-extraction --env-file 配置路径 serve`。输入论文的文本和用于识别的页面图像会发送给选定的模型服务。

```shell
paper-extraction doctor --live --provider deepseek
paper-extraction serve
```

从源码安装时在上述命令前加 `uv run`。用浏览器打开 **http://127.0.0.1:8765**。可用 `serve --port 8766` 更换端口。网页“模型设置”也可分别测试文本和视觉 API；连接测试会发起真实模型请求。

## 一次提取

1. 在“新建批次”选择多个文件或整个论文目录，核对预览清单，为有多个入口的 TeX 包选择主文件。
2. 选择模型并开始提取。默认并发两篇。关闭网页后后端继续工作；重新启动服务时恢复未完成任务。
3. 在批次页点击论文，查看生成稿、证据和处理记录。正式成品支持 PDF 预览和下载；失败项保留原因，可以单独重试。
4. 下载当前已完成论文的 ZIP。批次清单也会列出排队、失败或仍在处理的项目。

**示例。** 选择 [examples/compactness](examples/compactness) 文件夹。主文件引用另一份章节文件，演示如何从结论跳到源文件行号。这是基于经典 Arzelà–Ascoli 定理的教学样本。使用模型实际生成时，数学取舍和措辞依本次机器复核而定。成功的 ZIP 包含：

```text
001-compactness/
  introduction.tex
  introduction.pdf
  preamble-v13.tex
manifest.json
```

## 工作流与判定

LangGraph 为每篇论文保存独立的 SQLite 检查点。LangChain 的模型调用与只读工具让 Agent 搜索原文、读取段落、查看原页。复核发现漏选的重要应用时回到提取，已选陈述有误时回到撰写。程序使用固定的 v13 四件套渲染 TeX，双遍编译，检查真实排版与页数，并将最终每一页交给视觉模型检查。首次通过即可交付，有具体缺陷才返工，每篇最多两轮。工程和视觉检查均有记录；数学取舍属于机器的编辑判断。

标准成果最多三页正文，参考文献单独起页，含 1–3 条历史定理、1–4 条当前贡献、0–6 个必要定义。来源或篇幅不足时在批次中说明原因。[随包规则](src/paper_extraction/profile_assets/intro-rules-v13.txt)

## 配置与开发

| 配置 | 作用 |
|---|---|
| `PAPER_EXTRACTION_PROVIDER` | 默认服务商，`deepseek` 或 `zhipu` |
| `DEEPSEEK_API_KEY` / `ZAI_API_KEY` | 独立模型密钥 |
| `DEEPSEEK_TEXT_MODEL` / `DEEPSEEK_VISION_MODEL` | DeepSeek 文本及图像模型，默认 `deepseek-flash` |
| `ZHIPU_TEXT_MODEL` / `ZHIPU_VISION_MODEL` | 智谱文本及图像模型，默认 `glm-5.3` / `glm-4.6v` |
| `DEEPSEEK_BASE_URL` / `ZHIPU_BASE_URL` | 对应模型服务的 API 地址 |
| `PAPER_EXTRACTION_HOME` | 本机论文快照、成品和任务数据库目录，默认使用平台用户数据目录 |
| `PAPER_EXTRACTION_CONCURRENCY` | 同时处理的论文数量，默认 2 |

测试运行 `uv run pytest -q` 和 `npm --prefix frontend test`。生成安装包先构建网页，再运行 `uv build --wheel`。FastAPI 接口说明位于本地服务的 `/docs`。测试中的模型回放用于验证执行路径，真实模型验收使用独立配置。

项目采用 [MIT 许可证](LICENSE)。第三方组件见[许可说明](THIRD_PARTY.md)。
