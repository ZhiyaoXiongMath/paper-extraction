import { test } from "node:test";
import assert from "node:assert/strict";
import { batchPaperLabel, canDeleteBatch, metrics, readBlocks, terminal, paperOutcome, paperTone } from "./format.js";

test("single-batch deletion accepts paused papers and waits for active papers to pause", () => {
  for (const counts of [{ paused: 3 }, { completed: 1, paused: 2 }, { paused: 1, failed: 1, cancelled: 1 }]) {
    assert.equal(canDeleteBatch({ total: 3, counts }), true);
    assert.equal(canDeleteBatch({ total: 3, counts }, false), false);
  }
  for (const status of ["queued", "running", "pausing"]) {
    assert.equal(canDeleteBatch({ total: 3, counts: { paused: 2, [status]: 1 } }), false);
  }
  assert.equal(canDeleteBatch({ total: 3, counts: { completed: 3 } }, false), true);
});

test("counts queued, running, accepted and failed independently", () => {
  assert.deepEqual(
    metrics([
      { total: 8, counts: { queued: 2, running: 1, accepted: 4, failed: 1 } },
    ]),
    { total: 8, queued: 2, running: 1, completed: 4, failed: 1 },
  );
  assert.equal(terminal("queued"), false);
  assert.equal(terminal("accepted"), true);
});
test("two actively processing papers do not appear in the queue", () => {
  assert.deepEqual(metrics([{ total: 2, counts: { running: 2 } }]), {
    total: 2,
    queued: 0,
    running: 2,
    completed: 0,
    failed: 0,
  });
});
test("an empty draft is not a mathematical rejection", () => {
  const task = {
    status: "failed",
    draft: {},
    reason: "Insufficient sourced history. The draft entries are empty.",
    candidates: [{}],
    reviews: [],
    events: [{ stage: "evidence" }],
  };
  const result = paperOutcome(task);
  assert.equal(result.title, "未生成成品");
  assert.equal(result.summary, "初稿尚未生成，当前结果不能判断论文是否合格。");
  assert.equal(result.initialTab, "candidates");
  assert.equal(paperOutcome({ ...task, candidates: [] }).initialTab, "log");
});
test("a generated draft names the check that actually stopped", () => {
  const task = {
    status: "failed",
    draft: { paragraph: "Let X be compact." },
    reviews: [{ stage: "compile", passed: false }],
    events: [{ stage: "compile" }],
  };
  assert.equal(paperOutcome(task).title, "编译与版式检查未完成");
  assert.equal(paperOutcome(task).initialTab, "checks");
});
test("delivered papers open the reader regardless of a previous review failure", () => {
  const task = {
    status: "completed",
    draft: { paragraph: "Let X be compact." },
    reviews: [{ stage: "review", passed: false }],
  };
  assert.equal(paperOutcome({ ...task, delivery: { status: "delivered" } }).title, "已交付");
  assert.equal(paperOutcome({ ...task, delivery: { status: "delivered" } }).initialTab, "reader");
  assert.equal(paperOutcome({ ...task, delivery: { status: "reviewed" } }).title, "已交付 · 已通过检查");
});
test("partial check coverage has a plain delivery label in the batch and paper header", () => {
  const task = {
    status: "completed", delivery: {
      status: "delivered", draft_round: 2, compiler_verified: true,
      checks: { content: "not_checked", layout: "failed", visual: "not_checked" },
    },
  };
  for (const kind of ["final", "prior", "readable"]) {
    const paper = { ...task, delivery: { ...task.delivery, kind } };
    assert.equal(paperTone(paper), "neutral");
    assert.equal(batchPaperLabel(paper), "已交付");
    assert.equal(paperOutcome(paper).title, "已交付");
  }
  const reviewed = { ...task, delivery: { ...task.delivery, status: "reviewed" } };
  assert.equal(paperTone(reviewed), "good");
  assert.match(batchPaperLabel(reviewed), /全检查通过/);
  const summary = { ...task, delivery: { status: "delivered", kind: "summary" } };
  assert.equal(paperTone(summary), "good");
  assert.equal(batchPaperLabel(summary), "已交付 · 筛选说明");
  assert.equal(batchPaperLabel({ status: "completed" }), "已完成");
});

test("a Markdown-only result opens the screening note", () => {
  const paper = {status: "completed", delivery: {
    kind: "summary", outcome: "insufficient_evidence", note: "证据不足。"
  }};
  const outcome = paperOutcome(paper);
  assert.equal(outcome.initialTab, "summary");
  assert.equal(outcome.summary, "证据不足。");
});
test("reader block ids preserve the source evidence links", () => {
  const blocks = readBlocks({
    history: ["previous"],
    shared_definitions: ["shared"],
    groups: [{ definitions: ["local"], theorem: "conclusion" }],
  });
  assert.deepEqual(
    blocks.map((x) => [x.id, x.heading]),
    [
      ["history.0", "Theorem A"],
      ["shared_definitions.0", "Definition 1"],
      ["groups.0.definitions.0", "Definition 2"],
      ["groups.0.theorem", "Theorem 1"],
    ],
  );
});
