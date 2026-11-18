import assert from "node:assert/strict";
import { test } from "node:test";
import { paperDownloads, visibleFailureReasons } from "./paper-downloads.js";

const prefix = "/api/papers/fixture";
const tex = { url: prefix + "/drafts/1/2/introduction.tex", sha256: "tex" };
const pdf = { url: prefix + "/drafts/1/2/introduction.pdf", sha256: "pdf" };
const record = { current: true, attempt: 1, round: 2, restored: false, files: { tex, pdf } };

test("a failed round exposes exactly its available TeX and PDF", () => {
  for (const files of [{ tex }, { pdf }, { tex, pdf }, {}]) {
    const view = paperDownloads({ status: "failed", attempt: 1, draft_artifacts: [{ ...record, files }] }, prefix);
    assert.equal(view.label, "未通过 · 第 2 轮草稿");
    assert.equal(view.tex, files.tex?.url);
    assert.equal(view.pdf, files.pdf?.url);
    assert.equal(view.missingPdf, files.pdf ? "" : "本轮未生成 PDF");
  }
});

test("earlier attempt or obsolete draft files cannot become the visible downloads", () => {
  const paper = { status: "failed", attempt: 2, draft: { paragraph: "Latest" }, reviews: [{ attempt: 3 }],
    draft_artifacts: [record, { ...record, attempt: 2, current: false }] };
  const view = paperDownloads(paper, prefix);
  assert.equal(view.tex, undefined);
  assert.equal(view.pdf, undefined);
  assert.equal(view.label, "未通过 · 第 3 轮草稿");
  const restored = paperDownloads({ ...paper, draft_artifacts: [
    ...paper.draft_artifacts, { ...record, current: true, attempt: 2, round: 3, restored: true },
  ] }, prefix);
  assert.equal(restored.restored, true);
  assert.equal(restored.recompiled, false);
  const rebuilt = paperDownloads({ ...paper, draft_artifacts: [
    { ...record, current: true, attempt: 2, round: 3, restored: true, recompiled: true },
  ] }, prefix);
  assert.equal(rebuilt.recompiled, true);
  assert.equal(restored.pdf, pdf.url);
  assert.equal(restored.label, "未通过 · 第 3 轮草稿");
});

test("accepted files use the published artifacts while pre-draft failure has no download", () => {
  const view = paperDownloads({ status: "completed", draft_artifacts: [record] }, prefix);
  assert.equal(view.tex, prefix + "/artifacts/introduction.tex");
  assert.equal(view.pdf, prefix + "/artifacts/introduction.pdf");
  assert.equal(view.restored, false);
  const early = paperDownloads({ status: "failed", attempt: 1 }, prefix);
  assert.equal(early.label, "未通过 · 本轮尚无草稿");
  assert.equal(early.tex, undefined);
});

test("both final and reviewed deliveries enable the same published downloads", () => {
  for (const [delivery, expected] of [
    [{ status: "delivered", draft_round: 3 }, "已交付 · 第 3 稿"],
    [{ status: "delivered", draft_round: 2 }, "已交付 · 第 2 稿"],
    [{ status: "delivered" }, "已交付 · 成品"],
    [{ status: "reviewed", draft_round: 1 }, "已交付 · 已通过检查"],
  ]) {
    const paper = { status: "completed", delivery, draft_artifacts: [record], reviews: [{ stage: "review", passed: false }] };
    const view = paperDownloads(paper, prefix);
    assert.equal(view.label, expected);
    assert.equal(view.tex, prefix + "/artifacts/introduction.tex");
    assert.equal(view.pdf, prefix + "/artifacts/introduction.pdf");
    assert.deepEqual(visibleFailureReasons(paper), []);
  }
});

test("a screening note exposes only its Markdown download, including after an older formal draft", () => {
  const paper = { status: "completed", draft_artifacts: [record], delivery: {
    status: "delivered", kind: "summary", draft_round: 0, note: "未选出可单列的定理。",
  } };
  const view = paperDownloads(paper, prefix);
  assert.equal(view.markdown, prefix + "/artifacts/summary.md");
  assert.equal(view.tex, undefined);
  assert.equal(view.pdf, undefined);
  assert.equal(view.missingPdf, undefined);
});

test("checker counts and overfull widths are displayed at original precision and deduplicated", () => {
  const issue = "Engineering checks failed: no_blocking_diagnostics, v13_reader_format_contract | " +
    String.raw`introparagraph_1_word_limit:57>45 | Overfull \hbox (8.25000pt too wide) in paragraph at lines 13--13`;
  const paper = { status: "failed", reason: issue, reviews: [{ stage: "compile", passed: false, issues: [issue] }] };
  assert.deepEqual(visibleFailureReasons(paper), ["开场段 57 词，本次上限 45 词", "第 13 行超宽 8.25000 pt"]);
  assert.deepEqual(visibleFailureReasons({ ...paper, reason: "", reviews: [{ passed: false, issues: [
    String.raw`Overfull \hbox (4.3pt too wide) in paragraph at lines 7--9`,
  ] }] }), ["第 7–9 行超宽 4.3 pt"]);
});

test("structured mathematical reasons use the field and requested edit; other failures retain the message", () => {
  const paper = { status: "failed", reason: "Stopped after review", reviews: [
    { stage: "review", passed: false, issues: [], defects: [{ draft_field: "paragraph", required_change: "State the assumption." }] },
  ] };
  assert.deepEqual(visibleFailureReasons(paper), ["引言开场段 · State the assumption."]);
  assert.deepEqual(visibleFailureReasons({status: "failed", reason: "原文首页无法读取"}), ["原文首页无法读取"]);
  assert.deepEqual(visibleFailureReasons({status: "completed", reason: "Previous error"}), []);
});
