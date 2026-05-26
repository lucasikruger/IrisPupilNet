import { defineConfig } from "astro/config";
import react from "@astrojs/react";
import { fileURLToPath } from "node:url";

export default defineConfig({
  integrations: [react()],
  server: {
    host: "0.0.0.0",
    port: 4323,
  },
  vite: {
    resolve: {
      alias: {
        "@lib": fileURLToPath(new URL("./lib", import.meta.url)),
      },
    },
    optimizeDeps: {
      exclude: ["onnxruntime-web", "@mediapipe/tasks-vision"],
    },
    server: {
      headers: {
        // Required for WebGPU + SharedArrayBuffer (mediapipe/onnxruntime-web).
        // Use credentialless so cross-origin requests (backend API) still work.
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "credentialless",
      },
    },
  },
});
