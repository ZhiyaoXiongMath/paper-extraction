# Paper Extraction

[中文](README.md) · [Releases](https://github.com/ZhiyaoXiongMath/paper-extraction/releases)

A local workbench for mathematical papers. Import PDFs or TeX projects and get a selective English Introduction per paper, with source-grounded historical theorems, current contributions and definitions. The Chinese-language web interface links each generated statement to its source page or TeX lines.

## Install and run

Use Python 3.11–3.14 and TeX Live with `pdflatex` and `kpsewhich` on PATH. A full TeX Live installation includes the required packages. A minimal installation needs newtx, amsmath, amsthm, mathrsfs, enumitem, needspace, microtype, etoolbox, hyperref, titlesec and their dependencies.

The release wheel includes the web interface. Download the wheel and install it inside a virtual environment:

```shell
python -m pip install paper_extraction-0.1.0-py3-none-any.whl
paper-extraction --version
paper-extraction example compactness
```

For a source checkout, install [uv](https://docs.astral.sh/uv/) and Node.js 22.12 or later. Build the web interface and install the locked Python dependencies:

```shell
git clone https://github.com/ZhiyaoXiongMath/paper-extraction.git
cd paper-extraction
npm --prefix frontend ci
npm --prefix frontend run build
uv sync --locked --python 3.12
uv run paper-extraction doctor
```

Copy `.env.example` to `.env` in your **launch directory** and set your own `DEEPSEEK_API_KEY` or `ZAI_API_KEY`. Environment variables are also supported. Use `paper-extraction --env-file path/to/config serve` to select another file. Source passages and page images used in the workflow are sent to the selected model provider.

```shell
paper-extraction doctor --live --provider deepseek
paper-extraction serve
```

Prefix these commands with `uv run` in a source checkout. Open **http://127.0.0.1:8765** in your browser. Use `serve --port 8766` for another port. The model settings dialog also tests text and vision separately. Connection tests make live API requests.

## Process a batch

1. Choose files or a folder in the new-batch view. Inspect the preview and select the entry point for TeX packages with competing main files.
2. Select a provider and start. Two papers run concurrently by default. Closing the browser leaves the local worker active. Starting the service again resumes unfinished jobs.
3. Open a paper to read the generated content, examine original evidence and browse execution events. Preview and download accepted artifacts. Failed papers retain the reason and can be retried individually.
4. Download a ZIP containing all currently completed papers. The manifest also lists pending, running and failed items.

**Example.** Import the [examples/compactness](examples/compactness) folder. One main file includes a separate chapter, illustrating how generated statements link to original source lines. This teaching manuscript presents the classical Arzelà–Ascoli theorem. The selected statements and their wording depend on the model run. A completed batch yields:

```text
001-compactness/
  introduction.tex
  introduction.pdf
  preamble-v13.tex
manifest.json
```

## Workflow and acceptance

LangGraph keeps an independent SQLite checkpoint for each paper. LangChain binds read-only tools for searching the paper, reading passages and inspecting page images. Missing material contributions route back to extraction; errors in retained statements return to drafting. The program renders the fixed v13 profile, compiles twice, checks the actual layout and page counts, and submits every final page for visual inspection. A passing first draft is delivered immediately. Specific defects trigger at most two repair rounds. Engineering and visual checks are recorded separately. Mathematical selection is a machine editorial judgment.

The profile allows up to three body pages followed by a separate bibliography, 1–3 historical results, 1–4 current contributions and up to six definitions. Source and scope conflicts are recorded for the individual paper. [Profile rules](src/paper_extraction/profile_assets/intro-rules-v13.txt)

## Configuration and development

| Setting | Purpose |
|---|---|
| `PAPER_EXTRACTION_PROVIDER` | Default provider: `deepseek` or `zhipu` |
| `DEEPSEEK_API_KEY` / `ZAI_API_KEY` | Your independent model API key |
| `DEEPSEEK_TEXT_MODEL` / `DEEPSEEK_VISION_MODEL` | DeepSeek text and image models, default `deepseek-flash` |
| `ZHIPU_TEXT_MODEL` / `ZHIPU_VISION_MODEL` | GLM text and image models, default `glm-5.3` / `glm-4.6v` |
| `DEEPSEEK_BASE_URL` / `ZHIPU_BASE_URL` | Optional provider endpoint override |
| `PAPER_EXTRACTION_HOME` | Local data directory for source snapshots, checkpoints and artifacts. Defaults to the platform user-data directory |
| `PAPER_EXTRACTION_CONCURRENCY` | Number of active papers, default 2 |

Run `uv run pytest -q` and `npm --prefix frontend test`. To package, build the frontend and run `uv build --wheel`. The local FastAPI service serves API documentation at `/docs`. Offline model fixtures exercise execution paths; live model validation uses a separately configured account.

[MIT license](LICENSE). See [third-party notices](THIRD_PARTY.md).
