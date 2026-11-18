import assert from "node:assert/strict";
import { test } from "node:test";
import { evidenceLabel, readableReviewText, reviewDetails, reviewField } from "./review-format.js";

test("localizes historical, current and dedicated blocks in their reviewed draft", () => {
  assert.equal(reviewField("history.2"), "历史定理 C");
  assert.equal(reviewField("groups.0.theorem"), "本文定理 1");
  assert.equal(reviewField("shared_definitions.0"), "共用定义 1");
  assert.equal(reviewField("groups.2.definitions.1"), "定理 3 的专用定义 2");
  assert.equal(reviewField("paragraph"), "引言开场段");
  assert.equal(reviewField("unrecognized.path"), "unrecognized.path");
  assert.equal(evidenceLabel("p2-1"), "原文第 2 页 · p2-1");
  assert.equal(evidenceLabel("t1-1"), "原文依据 t1-1");
});

test("keeps source quotes and shows mixed legacy diagnostics without duplicating the structured edit", () => {
  const edit = {
    draft_field: "history.0", kind: "incorrect_attribution_or_locator", operation: "replace",
    required_change: "Provide the theorem number.", current_excerpt: "p. 2",
    replacement_text: "Theorem 2.1", evidence_id: "p2-1", source_excerpt: "by Theorem 2.1",
  };
  const issues = [
    "history.0: Provide the theorem number. [replace: Theorem 2.1] (draft: p. 2; original p2-1: by Theorem 2.1)",
    "history.0: Provide the theorem number. Also fix the missing hypothesis.",
  ];
  const result = reviewDetails({ defects: [edit], issues });
  assert.deepEqual(result, { defects: [edit], plain: [issues[1]] });
  const legacy = reviewDetails({ issues });
  assert.deepEqual(legacy.plain, [issues[1]]);
  assert.equal(legacy.defects[0].draft_field, "history.0");
  assert.equal(legacy.defects[0].replacement_text, "Theorem 2.1");
  assert.equal(legacy.defects[0].evidence_id, "p2-1");
  assert.deepEqual(reviewDetails({ defects: [{}], issues: ["A previous free-text verdict."] }),
    { defects: [], plain: ["A previous free-text verdict."] });
});

test("readable legacy diagnostics retain inline theorem contents and source locators", () => {
  const raw = String.raw`See \begin{historical}[{\cite[p.~2]{source}}]Every $X$ satisfies the result.\end{historical} for details.`;
  assert.equal(readableReviewText(raw), "See Every $X$ satisfies the result.〔原论文 · 第 2 页〕 for details.");
  const old = "paragraph: Restore the hypotheses. (draft: [missing]; original t2-1: source (with parentheses))";
  const [item] = reviewDetails({ issues: [old] }).defects;
  assert.equal(item.current_excerpt, "");
  assert.equal(item.source_excerpt, "source (with parentheses)");
  assert.equal(item.evidence_id, "t2-1");
});
