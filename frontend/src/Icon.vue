<script setup lang="ts">
defineProps({ name: String, size: { type: Number, default: 18 } });
const paths: Record<string, string> = {
  book: "M3 3.5h7l2 2 2-2h7v15h-7l-2 2-2-2H3z M12 5.5v15",
  grid: "M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z",
  file: "M6 2h9l4 4v16H6z M14 2v5h5 M9 12h7 M9 16h6",
  stack: "M3 7l9-4 9 4-9 4z M3 12l9 4 9-4 M3 17l9 4 9-4",
  plus: "M12 5v14 M5 12h14",
  pencil: "M4 20l4.5-1 11-11a2.5 2.5 0 0 0-3.5-3.5l-11 11L4 20z M14.5 6l3.5 3.5",
  settings:
    "M3 6h7 M14 6h7 M3 12h12 M19 12h2 M3 18h2 M9 18h12 M10 3v6 M15 9v6 M8 15v6",
  arrow: "M4 12h16 M14 6l6 6-6 6",
  back: "M20 12H4 M10 6l-6 6 6 6",
  chevron: "M9 5l7 7-7 7",
  clock: "M12 6v6l3.5 2 M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0z",
  check: "m5 12 4 4L19 6",
  fail: "M12 7v6 M12 17h.01 M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0z",
  upload: "M12 15V3 M7 8l5-5 5 5 M4 15v5h16v-5",
  folder: "M3 5h7l2 3h9v12H3z",
  download: "M12 3v12 M7 10l5 5 5-5 M4 15v5h16v-5",
  link: "M10 13a4 4 0 0 0 6 0l3-3a4 4 0 0 0-6-6l-2 2 M14 11a4 4 0 0 0-6 0l-3 3a4 4 0 0 0 6 6l2-2",
  play: "m7 4 13 8-13 8z",
  pause: "M9 5v14 M15 5v14",
  refresh: "M20 10a8 8 0 1 0-1 7 M20 4v6h-6",
  chip: "M6 6h12v12H6z M9 2v4 M15 2v4 M9 18v4 M15 18v4 M2 9h4 M2 15h4 M18 9h4 M18 15h4",
  ext: "M13 4h7v7 M20 4l-10 10 M9 5H4v15h15v-5",
  close: "M6 6l12 12 M18 6L6 18",
  trash: "M3 6h18 M8 6V4h8v2 M5.5 6l1 14h11l1-14 M10 10v6 M14 10v6",
  list: "M8 5h13 M8 12h13 M8 19h13 M3 5h1 M3 12h1 M3 19h1",
  activity: "M2 13h5l3-8 4 14 3-6h5",
  monitor: "M3 4h18v13H3z M7 21h10 M12 17v4",
};
</script>
<template>
  <svg
    aria-hidden="true"
    :width="size"
    :height="size"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="1.6"
    stroke-linecap="round"
    stroke-linejoin="round"
  >
    <path :d="paths[name || '']" />
  </svg>
</template>
