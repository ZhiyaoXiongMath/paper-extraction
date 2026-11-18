import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  base: "/static/",
  plugins: [vue()],
  build: {
    outDir: "../src/paper_extraction/static",
    emptyOutDir: true,
    license: { fileName: "licenses.txt" },
  },
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
