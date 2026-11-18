import test from "node:test";
import assert from "node:assert/strict";
import { renderMarkdown } from "./rich-text.js";

const tex = html => [...html.matchAll(/<annotation encoding="application\/x-tex">([\s\S]*?)<\/annotation>/g)].map(m => m[1]);

test("mixed mathematical prose preserves indexed derivatives, scalar-curvature bars and the stated initial value", () => {
  const output = renderMarkdown("For ω(t) = ω0 + i∂∂̄φ(t) > 0, ∂_t φ = S(ω(t)) − S̄, the mean is S̄.");
  const maths = tex(output).join(" ");
  assert.match(maths, /\\omega_\{0\}/);
  assert.match(maths, /\\partial_t/);
  assert.match(maths, /\\partial\\bar\{\\partial\}/);
  assert.match(maths, /\\bar\{S\}/);
  assert.match(output, /<msub>/);
  assert.match(output, /<mover/);
  assert.match(output, /the mean is/);
});

test("ambiguous plain data and explicitly supplied TeX do not acquire indices or averages", () => {
  const output = renderMarkdown("Ht, AH, α0, ∂tφ, φt and S are the recorded symbols. $ω(t)=ω0$.");
  assert.match(output, /Ht, AH/);
  assert.doesNotMatch(output, /<msub>|<mover/);
  assert.equal(tex(renderMarkdown("ω₀, ω_0, α₀")).filter(v => /_/.test(v)).length, 3);
  assert.doesNotMatch(tex(renderMarkdown("U(t) = ω0")).join(""), /_\{0\}/);
});

test("Markdown and TeX accents preserve headings, lists, emphasis and literal code", () => {
  const html = renderMarkdown("## Background\n\n- **First item** and $x_1$\n- " + String.raw`K\"ahler and Chru\'sciel` + "\n\n" + String.raw`\emph{A comparison}` + ". Use `∂_t φ`.");
  assert.match(html, /<h2>Background<\/h2>/);
  assert.match(html, /<ul>/);
  assert.match(html, /<strong>First item<\/strong>/);
  assert.match(html, /Kähler and Chruściel/);
  assert.match(html, /<em>A comparison<\/em>/);
  assert.match(html, /<code>∂_t φ<\/code>/);
  assert.deepEqual(tex(html), ["x_1"]);
});

test("display math spans multiple lines and handled parse errors keep their source", () => {
  const html = renderMarkdown("Before\n\n$$\na+b\n\n=c\n$$\n\nAfter");
  assert.match(html, /class="katex-display"/);
  assert.match(html, /After/);
  const literal = renderMarkdown("$\\unknownMacro{x}$");
  assert.match(literal, /math-fallback/);
  assert.match(literal, /\\unknownMacro/);
});

test("candidate reasons link only registered text IDs while math and code keep their meaning", () => {
  const html = renderMarkdown("Required to state c5, c20, c13. `c5` and $c5$.", new Set(["c5", "c20"]));
  assert.deepEqual([...html.matchAll(/data-candidate-ref="([^"]+)"/g)].map(m => m[1]), ["c5", "c20"]);
  assert.match(html, /c13/);
  assert.doesNotMatch(renderMarkdown("c5"), /data-candidate-ref/);
});

test("raw HTML and active URL schemes cannot become executable paper markup", () => {
  const html = renderMarkdown('<img src=x onerror=alert(1)> [bad](javascript:alert(1)) **OK**');
  assert.doesNotMatch(html, /<img |href="javascript:/i);
  assert.match(html, /&lt;img/);
  assert.match(html, /<strong>OK<\/strong>/);
});


test("candidate dependency fields preserve differences without repeating the reason's list", async () => {
  const { requiredRelations } = await import("./candidate-relations.js");
  assert.deepEqual(requiredRelations("Required to state c5, c20.", ["c20", "c5"]), []);
  assert.deepEqual(requiredRelations("Required to state c5, c20, c13", ["c5", "c20"]), [["仅判定理由列出", ["c13"]]]);
  assert.deepEqual(requiredRelations("Required to state c5", ["c5", "c20"]), [["“用于陈述”另列", ["c20"]]]);
  assert.deepEqual(requiredRelations("A comparison with c5", ["c20"]), [["用于陈述", ["c20"]]]);
});

test("letter-named TeX accents do not consume the start of other TeX commands", () => {
  assert.match(renderMarkdown(String.raw`Fran\c{c}ois`), /François/);
  assert.match(renderMarkdown(String.raw`\cite{source}`), /\\cite\{source\}/);
  assert.match(tex(renderMarkdown(String.raw`\cdot x`)).join(""), /\\cdot/);
});

test("Unicode primes on mathematical symbols use KaTeX prime glyphs without changing quotations or code", () => {
  const warnings = [];
  const warn = console.warn;
  console.warn = (...args) => warnings.push(args.join(" "));
  try {
    const html = renderMarkdown("The symbols Φ′′(t), Φ″(t) and f′(t) occur in “quoted text”; the width is 12″. `Φ″`");
    assert.deepEqual(tex(html).map(value => value.replaceAll("&#x27;", "'")), ["\\Phi''(t)", "\\Phi''(t)", "f'(t)"]);
    assert.match(html, /“quoted text”/);
    assert.match(html, /12″/);
    assert.match(html, /<code>Φ″<\/code>/);
    assert.deepEqual(warnings, []);
  } finally {
    console.warn = warn;
  }
});
