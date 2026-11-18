import MarkdownIt from "markdown-it";
import katex from "katex";

const md = new MarkdownIt({ html: false, linkify: false, typographer: false });
const escape = md.utils.escapeHtml;

const greek = Object.fromEntries(
  [..."αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΥΦΨΩϕϑς"].map((c, i) => [
    c, ["alpha","beta","gamma","delta","varepsilon","zeta","eta","theta","iota","kappa",
      "lambda","mu","nu","xi","o","pi","rho","sigma","tau","upsilon","varphi","chi","psi",
      "omega","Gamma","Delta","Theta","Lambda","Xi","Pi","Sigma","Upsilon","Phi","Psi",
      "Omega","phi","vartheta","varsigma"][i],
  ]),
);
const symbols = { "∂":"\\partial ", "∇":"\\nabla ", "∞":"\\infty ", "∈":"\\in ", "∉":"\\notin ",
  "⊂":"\\subset ", "⊆":"\\subseteq ", "≥":"\\ge ", "≤":"\\le ", "≠":"\\ne ", "≈":"\\approx ",
  "∼":"\\sim ", "×":"\\times ", "→":"\\to ", "↦":"\\mapsto ", "−":"-", "∗":"\\ast ",
  "′":"'", "″":"''", "ℝ":"\\mathbb{R}", "ℂ":"\\mathbb{C}", "ℤ":"\\mathbb{Z}", "ℕ":"\\mathbb{N}", "ℚ":"\\mathbb{Q}" };
const below = "₀₁₂₃₄₅₆₇₈₉₊₋", above = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻";
const digits = "0123456789+-";
const marks = { "\u0304":"bar", "\u0305":"overline", "\u0302":"hat", "\u0303":"tilde", "\u0307":"dot" };
const operators = new Set(["tr", "Ric", "Scal", "det", "dim", "ker", "rank", "log", "exp", "sin", "cos", "lim", "sup", "inf"]);
const commands = new Set(["frac","dfrac","tfrac","sqrt","bar","overline","hat","widehat","tilde","dot",
  "partial","nabla","operatorname","mathrm","mathbb","mathcal","mathbf","left","right",
  "infty","to","mapsto","cdot","times","le","leq","ge","geq","neq","in","subset","subseteq",
  "sum","int","lim","log","sin","cos","det","sim","approx","langle","rangle", ...Object.values(greek)]);
const token = /\\[A-Za-z]+|[A-Za-z]+|[0-9]+(?:\.[0-9]+)?|[ \t]+|./uy;
const mathPunctuation = new Set([..."()[]{}_^+=/<>|,−-′″"]);
const open = { "(": ")", "[": "]", "{": "}" };
const close = new Set(Object.values(open));

const baseTex = (ch) => {
  const command = greek[ch];
  return command ? (command === "o" ? "o" : "\\" + command + " ") : symbols[ch] ?? ch;
};

export function unicodeMathToTex(source, inferInitialValue = false) {
  // Interpret character-level notation. Plain Ht/φt, unmarked S, and ordinary
  // alphanumeric names retain the characters supplied by the source.
  let result = source.replace(/([A-Za-z\p{Script=Greek}∂∇])([\u0304\u0305\u0302\u0303\u0307])/gu,
    (_, ch, accent) => "\\" + marks[accent] + "{" + baseTex(ch).trimEnd() + "}");
  if (inferInitialValue) {
    result = result.replace(/(\p{Script=Greek})(\s*\(\s*t\s*\)\s*=\s*)(\p{Script=Greek})0(?![\p{L}\p{N}])/gu,
      (match, left, equals, right) => left === right ? left + equals + right + "_{0}" : match);
  }
  result = result.replace(/[₀₁₂₃₄₅₆₇₈₉₊₋]+/gu, (s) => "_{" + [...s].map(c => digits[below.indexOf(c)]).join("") + "}")
    .replace(/[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+/gu, (s) => "^{" + [...s].map(c => digits[above.indexOf(c)]).join("") + "}");
  return [...result].map(baseTex).join("").replace(/(\\[A-Za-z]+) (?=[^A-Za-z]|$)/g, "$1");
}

function nextToken(s, pos) {
  token.lastIndex = pos;
  return token.exec(s)?.[0] ?? "";
}

function implicitAt(s, start) {
  let pos = start, end = start, anchor = false;
  const stack = [];
  let tex = "", keep = "";
  while (pos < s.length) {
    const word = nextToken(s, pos);
    if (!word || word[0] === "\n") break;
    let value = word;
    if (/^[A-Za-z]+$/.test(word)) {
      if (word.length > 1) {
        if (!operators.has(word)) break;
        value = "\\operatorname{" + word + "}";
      }
    } else if (word[0] === "\\") {
      if (!commands.has(word.slice(1))) break;
      anchor = true;
    } else if (Object.hasOwn(greek, word) || Object.hasOwn(symbols, word)
        || above.includes(word) || below.includes(word) || marks[word]) {
      anchor = true;
    } else if (!/^\d|^[ \t]/.test(word) && !mathPunctuation.has(word)) {
      break;
    }
    if (pos === start && (/^[ \t\d]/.test(word) || (mathPunctuation.has(word) && !open[word]))) return null;
    if (word === "," && !stack.length) break;
    if (open[word]) stack.push(open[word]);
    if (close.has(word)) {
      if (word !== stack.at(-1)) break;
      stack.pop();
    }
    if (word === "_" || word === "^") anchor = true;
    pos += word.length;
    tex += value;
    if (!stack.length && !/^[ \t,_^+=/<>\-]$/.test(word) && !/^[ \t]+$/.test(word)) {
      keep = tex;
      end = pos;
    }
  }
  if (!anchor || end <= start || /(?:^|[^\\])__|\*\*/.test(keep)) return null;
  return { start, end, type: "formula", tex: unicodeMathToTex(keep, true), raw: s.slice(start, end), display: false };
}

function braceEnd(text, start) {
  let depth = 1;
  for (let i = start; i < text.length; i++) {
    if (text[i] === "\\" && /[{}\\]/.test(text[i + 1] || "")) { i++; continue; }
    if (text[i] === "{") depth++;
    if (text[i] === "}" && --depth === 0) return i;
  }
  return -1;
}

const accents = { '"':"\u0308", "'":"\u0301", "\x60":"\u0300", "^":"\u0302", "~":"\u0303", "=":"\u0304", ".":"\u0307", c:"\u0327" };
const delimiters = [["$$","$$",true], ["\\[","\\]",true], ["\\(","\\)",false], ["$","$",false]];

function pieces(text) {
  const spans = [];
  for (let i = 0; i < text.length;) {
    let part;
    const delim = delimiters.find(([left]) => text.startsWith(left, i));
    if (delim && !(i > 0 && text[i-1] === "\\")) {
      const [left, right, display] = delim;
      let end = text.indexOf(right, i + left.length);
      while (end !== -1 && right === "$" && text[end-1] === "\\") end = text.indexOf(right, end + 1);
      if (end > i + left.length) {
        const finish = end + right.length;
        part = {start:i, end:finish, type:"formula", raw:text.slice(i,finish),
          tex:unicodeMathToTex(text.slice(i + left.length,end)), display};
      }
    }
    if (!part && text[i] === "\\") {
      const markup = /^\\(emph|textit|textbf)\{/.exec(text.slice(i));
      const accent = /^\\(["'\x60^~=.](?!\s)|c(?=\{|\s))(?:\{([A-Za-z])\}|\s*([A-Za-z]))/.exec(text.slice(i));
      if (markup) {
        const from = i + markup[0].length, end = braceEnd(text, from);
        if (end >= 0) part = {start:i,end:end+1,type:markup[1] === "textbf" ? "strong" : "em",content:text.slice(from,end)};
      } else if (accent) {
        part = {start:i,end:i+accent[0].length,type:"accent",
          content:((accent[2] || accent[3]) + accents[accent[1]]).normalize("NFC")};
      }
    }
    if (!part) part = implicitAt(text, i);
    if (part) {
      spans.push(part);
      i = part.end;
    } else {
      i += nextToken(text, i).length || 1;
    }
  }
  return spans;
}

function mathHtml(piece) {
  try {
    return katex.renderToString(piece.tex, { displayMode: piece.display, throwOnError: true, trust: false, strict:"ignore" });
  } catch {
    return '<code class="math-fallback">' + escape(piece.raw) + "</code>";
  }
}

const cached = new WeakMap();
const plainText = md.inline.ruler.getRules("")[0];
md.inline.ruler.at("text", (state, silent) => {
  if (!cached.has(state)) cached.set(state, pieces(state.src));
  const spans = cached.get(state);
  const span = spans.find(p => p.start >= state.pos && p.end <= state.posMax);
  if (!span) return plainText(state, silent);
  if (span.start === state.pos) {
    if (!silent) state.push("paper_span", "", 0).meta = span;
    state.pos = span.end;
    return true;
  }
  const end = state.posMax;
  state.posMax = span.start;
  try { return plainText(state, silent); } finally { state.posMax = end; }
});
md.renderer.rules.paper_span = (tokens, i, options, env) => {
  const s = tokens[i].meta;
  if (s.type === "formula") return mathHtml(s);
  if (s.type === "accent") return escape(s.content);
  return "<"+s.type+">"+md.renderInline(s.content,env)+"</"+s.type+">";
};

md.renderer.rules.text = (tokens, index, options, env) => {
  const s = tokens[index].content;
  if (!env?.candidateIds?.size) return escape(s);
  const parts = s.split(/\b(c[1-9]\d*)\b/g);
  return parts.map((part, i) => i % 2 && env.candidateIds.has(part)
    ? '<button type="button" class="candidate-ref" data-candidate-ref="'+part+'">'+part+'</button>'
    : escape(part)).join("");
};

export function renderMarkdown(text, candidateIds) {
  return md.render(String(text ?? ""), {candidateIds});
}


md.block.ruler.before("fence", "paper_display", (state, first, last, silent) => {
  const line = n => state.src.slice(state.bMarks[n] + state.tShift[n], state.eMarks[n]);
  const opening = line(first).trim();
  if (opening !== "$$" && opening !== "\\[") return false;
  const closing = opening === "$$" ? "$$" : "\\]";
  let n = first + 1;
  while (n < last && line(n).trim() !== closing) n++;
  if (n === last) return false;
  if (!silent) {
    const inner = state.getLines(first + 1, n, state.blkIndent, false);
    const block = state.push("paper_span", "", 0);
    block.block = true;
    block.map = [first, n + 1];
    block.meta = {type:"formula", display:true, tex:unicodeMathToTex(inner), raw:opening+"\n"+inner+"\n"+closing};
    state.line = n + 1;
  }
  return true;
});
