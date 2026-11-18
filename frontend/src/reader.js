// A v13 caption can itself contain bracketed source locators, e.g. [{\cite[p.~2]{source}}].
function groupAt(text, start) {
  const open = text[start];
  if (open !== "{" && open !== "[") return null;
  const stack = [open];
  for (let i = start + 1; i < text.length; i++) {
    const ch = text[i];
    if (ch === "\\") {
      i++;
      continue;
    }
    const top = stack[stack.length - 1];
    if (ch === "{" || (ch === "[" && top === "[")) stack.push(ch);
    else if ((ch === "}" && top === "{") || (ch === "]" && top === "[")) {
      stack.pop();
      if (!stack.length) return { value: text.slice(start + 1, i), end: i + 1 };
    }
  }
  return null;
}

function ungroup(text) {
  const trimmed = text.trim();
  if (!trimmed.startsWith("{")) return trimmed;
  const group = groupAt(trimmed, 0);
  return group?.end === trimmed.length ? group.value.trim() : trimmed;
}

function sourcePath(path, isUri) {
  const name = path.replace(/\\symbol\{(91|93)\}/g, (_, n) => String.fromCharCode(Number(n)))
    .replace(/\\textbackslash\{\}/g, "\\").replace(/\\textasciitilde\{\}/g, "~")
    .replace(/\\textasciicircum\{\}/g, "^").replace(/\\([#$%&_{}])/g, "$1");
  if (isUri) {
    try { return decodeURIComponent(name); } catch { /* Preserve an unrecognized encoding. */ }
  }
  return name;
}

function readableLocator(text) {
  const files = [];
  const filePattern = /(URI path )?\\texttt\s*\{/g;
  let rest = "", from = 0;
  for (let match; (match = filePattern.exec(text));) {
    const group = groupAt(text, filePattern.lastIndex - 1);
    if (!group) continue;
    const line = text.slice(group.end).match(/^\s+L[~\s]*(\d+(?:--?\d+)?(?:,\s*\d+(?:--?\d+)?)*)/);
    const label = sourcePath(group.value, Boolean(match[1])) +
      (line ? " 第 " + line[1].replace(/--/g, "–") + " 行" : "");
    rest += text.slice(from, match.index) + "\u0000" + files.length + "\u0000";
    files.push(label);
    filePattern.lastIndex = from = group.end + (line?.[0].length || 0);
  }
  return (rest + text.slice(from)).replace(/~/g, " ").replace(/--/g, "–")
    .replace(/\\S\s*/g, "§")
    .replace(/\b(PDF physical )?pp?\.\s*(\d+(?:[–-]\d+)?(?:,\s*\d+(?:[–-]\d+)?)*)/gi,
      (_, physical, pages) => (physical ? "PDF 物理页 " + pages : "第 " + pages + " 页"))
    .replace(/\b(Theorem|Corollary|Proposition|Definition|Lemma|Equation)\s+(?=[(\d])/gi,
      (word) => ({ theorem: "定理", corollary: "推论", proposition: "命题", definition: "定义",
        lemma: "引理", equation: "公式" })[word.trim().toLowerCase()] + " ")
    .replace(/,\s*/g, "，").replace(/;\s*/g, "；")
    .replace(/\u0000(\d+)\u0000/g, (_, index) => files[index]);
}

function formatCitations(text, references, inline) {
  const cite = /\\cite\b\s*/g;
  let result = "", from = 0;
  for (let match; (match = cite.exec(text));) {
    let i = cite.lastIndex;
    const locator = text[i] === "[" ? groupAt(text, i) : null;
    if (locator) i = locator.end;
    while (/\s/.test(text[i] || "")) i++;
    const keys = text[i] === "{" ? groupAt(text, i) : null;
    if (!keys || !keys.value.trim()) continue;
    const cited = keys.value.split(",").map((item) => {
      const key = item.trim();
      if (key === "source") return "原论文";
      const index = references.findIndex((ref) => ref.key === key);
      return index >= 0 ? "文献 [" + (index + 1) + "]" : "文献 " + key;
    }).join("、");
    const label = cited + (locator?.value.trim() ? " · " + readableLocator(locator.value) : "");
    result += text.slice(from, match.index) + (inline ? "〔" + label + "〕" : label);
    cite.lastIndex = from = keys.end;
  }
  return result + text.slice(from);
}

export function readerFragment(input, heading = "", references = []) {
  const source = String(input ?? "").trim();
  const wrapped = /^\\begin\s*\{(historical|theorem|definition|introparagraph)\}\s*([\s\S]*?)\s*\\end\s*\{\1\}$/.exec(source);
  let body = wrapped ? wrapped[2].trim() : source;
  let caption = "";
  if (wrapped && body.startsWith("[")) {
    const group = groupAt(body, 0);
    if (group) {
      caption = formatCitations(ungroup(group.value), references, false);
      body = body.slice(group.end).trimStart();
    }
  }
  // A standalone title is markup. A sentence beginning “Theorem B states…” is content.
  if (heading) {
    const literal = heading.replace(/[^\w\s]/g, "\\$&");
    const title = new RegExp("^(?:" + literal + "[.:](?=\\s|$)|" + literal +
      "(?=[ \\t]*\\n)|\\*\\*" + literal + "[.:]?\\*\\*[.:]?|\\\\textbf\\{" + literal + "[.:]?\\}[.:]?)(?:\\s+|$)");
    body = body.replace(title, "");
  }
  return { text: formatCitations(body, references, true), caption };
}
