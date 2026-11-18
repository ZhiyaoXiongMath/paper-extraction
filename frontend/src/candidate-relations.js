/** Display the structured dependency list once while preserving disagreements in stored records. */
export function requiredRelations(reason, neededBy) {
  if (!Array.isArray(neededBy)) return [];
  const ids = [...new Set(neededBy)];
  const listed = String(reason ?? "").trim().match(/^Required to state\s+(c\d+(?:(?:,\s*(?:and\s+)?|\s+and\s+)c\d+)*)\.?$/i);
  if (!listed) return ids.length ? [["用于陈述", ids]] : [];
  const textIds = [...new Set(listed[1].match(/c\d+/gi))];
  const onlyText = textIds.filter(id => !ids.includes(id));
  const onlyField = ids.filter(id => !textIds.includes(id));
  return [
    ...(onlyText.length ? [["仅判定理由列出", onlyText]] : []),
    ...(onlyField.length ? [["“用于陈述”另列", onlyField]] : []),
  ];
}
