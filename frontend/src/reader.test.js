import assert from "node:assert/strict";
import { test } from "node:test";
import { readerFragment } from "./reader.js";

test("a saved historical caption renders its locator and preserves the statement", () => {
  const body = "Every convergent sequence in a metric space has a unique limit.";
  const wrapped = "\\begin{historical}[{\\cite[\\S1.1, p.~2]{source}}]\n" + body + "\n\\end{historical}";
  assert.deepEqual(readerFragment(wrapped, "Theorem B"), { text: body, caption: "原论文 · §1.1，第 2 页" });
  assert.equal(readerFragment("Theorem B.\n" + body, "Theorem B").text, body);
  assert.equal(readerFragment("Theorem B states that curvature stays bounded.", "Theorem B").text,
    "Theorem B states that curvature stays bounded.");
});

test("definition and theorem captions preserve equation numbers and math delimiters", () => {
  const equation = "\\[\\partial_t\\varphi=S(\\omega(t))-\\bar S.\\]";
  assert.deepEqual(readerFragment("\\begin{definition}[{\\cite[\\S1, equation~(1.1)]{source}}]\n" + equation + "\n\\end{definition}"),
    { text: equation, caption: "原论文 · §1，公式 (1.1)" });
  assert.deepEqual(readerFragment("\\begin{theorem}[{\\cite[Corollary~1.2]{source}}]$T_*<\\infty$\\end{theorem}"),
    { text: "$T_*<\\infty$", caption: "原论文 · 推论 1.2" });
});

test("inline citations stay associated with their claims and bibliography entries", () => {
  const raw = "One result \\cite[pp.~3--4]{prior}. Another \\cite[Theorem~1.1]{source}.";
  assert.equal(readerFragment(raw, "", [{ key: "source" }, { key: "prior" }]).text,
    "One result 〔文献 [2] · 第 3–4 页〕. Another 〔原论文 · 定理 1.1〕.");
});

test("file and physical-page locators preserve the path and registered coordinate systems", () => {
  const raw = "\\cite[\\texttt{parts/a\\_b\\symbol{91}x\\symbol{93}.tex} L~7--9; URI path \\texttt{\\%E6\\%B5\\%8B.tex} L~12; PDF physical pp.~1--2]{source}";
  assert.equal(readerFragment(raw).text,
    "〔原论文 · parts/a_b[x].tex 第 7–9 行；测.tex 第 12 行；PDF 物理页 1–2〕");
  assert.equal(readerFragment(raw, "", [], "en").text,
    "[Source paper, parts/a_b[x].tex, lines 7–9; 测.tex, line 12; PDF physical pp. 1–2]");
});

test("the English reader uses numbered references and keeps mathematical content intact", () => {
  const references = [{ key: "source" }, { key: "prior" }];
  const body = "Yano and Kon \\cite[\\S1, Lemma~2]{source}: The result holds for $n\\ge 2$. See also \\cite[pp.~3--4]{prior,source}.";
  assert.equal(readerFragment(body, "", references, "en").text,
    "Yano and Kon [1, §1, Lemma 2]: The result holds for $n\\ge 2$. See also [2, 1, pp. 3–4].");
  const caption = "\\begin{historical}[{\\cite[\\S4, after Theorem~3]{source}}]Let $M^n$ be complete.\\end{historical}";
  assert.deepEqual(readerFragment(caption, "Theorem C", references, "en"),
    { text: "Let $M^n$ be complete.", caption: "[1, §4, after Theorem 3]" });
});

test("the English reader localizes known legacy locator fields only inside citations", () => {
  const body = "Original title «定理» \\cite[\\S1，引理 2；第 8--9 页]{source}.";
  assert.equal(readerFragment(body, "", [], "en").text,
    "Original title «定理» [Source paper, §1, Lemma 2; pp. 8–9].");
});
