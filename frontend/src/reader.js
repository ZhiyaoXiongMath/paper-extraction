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

function readableLocator(text, english) {
  const files = [];
  const filePattern = /(URI path )?\\texttt\s*\{/g;
  let rest = "", from = 0;
  for (let match; (match = filePattern.exec(text));) {
    const group = groupAt(text, filePattern.lastIndex - 1);
    if (!group) continue;
    const line = text.slice(group.end).match(/^\s+L[~\s]*(\d+(?:--?\d+)?(?:,\s*\d+(?:--?\d+)?)*)/);
    let label = sourcePath(group.value, Boolean(match[1]));
    if (line) {
      const range = line[1].replace(/--/g, "–");
      label += english
        ? (/[-–,]/.test(range) ? ", lines " : ", line ") + range
        : " 第 " + range + " 行";
    }
    rest += text.slice(from, match.index) + "\u0000" + files.length + "\u0000";
    files.push(label);
    filePattern.lastIndex = from = group.end + (line?.[0].length || 0);
  }
  const locator = (rest + text.slice(from)).replace(/~/g, " ").replace(/--/g, "–")
    .replace(/\\S\s*/g, "§");
  const localized = english ? locator
    .replace(/(定理|推论|命题|定义|引理|公式)\s*(?=[(\d])/g,
      (word) => ({ 定理: "Theorem", 推论: "Corollary", 命题: "Proposition", 定义: "Definition",
        引理: "Lemma", 公式: "Equation" })[word.trim()] + " ")
    .replace(/PDF\s*物理页\s*(\d+(?:[–-]\d+)?(?:,\s*\d+(?:[–-]\d+)?)*)/g,
      (_, pages) => (/[-–,]/.test(pages) ? "PDF physical pp. " : "PDF physical p. ") + pages)
    .replace(/第\s*(\d+(?:[–-]\d+)?(?:,\s*\d+(?:[–-]\d+)?)*)\s*(页|行)/g,
      (_, range, kind) => (kind === "页"
        ? (/[-–,]/.test(range) ? "pp. " : "p. ")
        : (/[-–,]/.test(range) ? "lines " : "line ")) + range)
    .replace(/，\s*/g, ", ").replace(/；\s*/g, "; ")
    : locator
    .replace(/\b(PDF physical )?pp?\.\s*(\d+(?:[–-]\d+)?(?:,\s*\d+(?:[–-]\d+)?)*)/gi,
      (_, physical, pages) => (physical ? "PDF 物理页 " + pages : "第 " + pages + " 页"))
    .replace(/\b(Theorem|Corollary|Proposition|Definition|Lemma|Equation)\s+(?=[(\d])/gi,
      (word) => ({ theorem: "定理", corollary: "推论", proposition: "命题", definition: "定义",
        lemma: "引理", equation: "公式" })[word.trim().toLowerCase()] + " ")
    .replace(/,\s*/g, "，").replace(/;\s*/g, "；");
  return localized.replace(/\u0000(\d+)\u0000/g, (_, index) => files[index]);
}

function formatCitations(text, references, inline, locale) {
  const english = locale === "en";
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
      const index = references.findIndex((ref) => ref.key === key);
      if (english) return index >= 0 ? String(index + 1) : key === "source" ? "Source paper" : "Reference " + key;
      if (key === "source") return "原论文";
      return index >= 0 ? "文献 [" + (index + 1) + "]" : "文献 " + key;
    }).join(english ? ", " : "、");
    const label = cited + (locator?.value.trim()
      ? (english ? ", " : " · ") + readableLocator(locator.value, english)
      : "");
    result += text.slice(from, match.index) + (english ? "[" + label + "]" : inline ? "〔" + label + "〕" : label);
    cite.lastIndex = from = keys.end;
  }
  return result + text.slice(from);
}

export function readerFragment(input, heading = "", references = [], locale = "zh") {
  const source = String(input ?? "").trim();
  const wrapped = /^\\begin\s*\{(historical|theorem|definition|introparagraph)\}\s*([\s\S]*?)\s*\\end\s*\{\1\}$/.exec(source);
  let body = wrapped ? wrapped[2].trim() : source;
  let caption = "";
  if (wrapped && body.startsWith("[")) {
    const group = groupAt(body, 0);
    if (group) {
      caption = formatCitations(ungroup(group.value), references, false, locale);
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
  return { text: formatCitations(body, references, true, locale), caption };
}
