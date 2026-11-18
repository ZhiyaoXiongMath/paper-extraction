import { test } from "node:test";
import assert from "node:assert/strict";
import { metrics, readBlocks, terminal } from "./format.js";

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
