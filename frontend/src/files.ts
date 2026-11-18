export interface UploadFile {
  file: File;
  path: string;
}
export const pickedFiles = (files?: FileList | null): UploadFile[] =>
  Array.from(files || []).map((file) => ({
    file,
    path: file.webkitRelativePath || file.name,
  }));

async function entries(
  reader: FileSystemDirectoryReader,
): Promise<FileSystemEntry[]> {
  const all: FileSystemEntry[] = [];
  while (true) {
    const page = await new Promise<FileSystemEntry[]>((resolve, reject) =>
      reader.readEntries(resolve, reject),
    );
    if (!page.length) return all;
    all.push(...page);
  }
}
async function walk(
  entry: FileSystemEntry,
  parent: string,
  files: UploadFile[],
): Promise<void> {
  if (entry.name.startsWith(".")) return;
  const path = parent ? parent + "/" + entry.name : entry.name;
  if (entry.isDirectory) {
    const reader = (entry as FileSystemDirectoryEntry).createReader();
    for (const child of await entries(reader)) await walk(child, path, files);
  } else if (
    entry.isFile &&
    /\.(pdf|tex|bib|bbl|sty|cls|bst|def|png|jp[e]?g|eps)$/i.test(entry.name)
  ) {
    const file = await new Promise<File>((resolve, reject) =>
      (entry as FileSystemFileEntry).file(resolve, reject),
    );
    files.push({ file, path });
  }
}
export async function droppedFiles(
  transfer: DataTransfer,
): Promise<UploadFile[]> {
  // Drag data is readable only during the event. Capture entry handles before awaiting.
  const roots = Array.from(transfer.items)
    .filter((item) => item.kind === "file")
    .map((item) => ({
      entry: item.webkitGetAsEntry?.(),
      file: item.getAsFile(),
    }));
  if (!roots.length) return pickedFiles(transfer.files);
  const result: UploadFile[] = [];
  for (const { entry, file } of roots) {
    if (entry) await walk(entry, "", result);
    else if (file)
      result.push({ file, path: file.webkitRelativePath || file.name });
  }
  return result;
}
