import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const API_TARGET = process.env.KITCHEN_API || "http://127.0.0.1:8099";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(process.cwd(), "./src"),
    },
  },
  server: {
    host: "127.0.0.1",
    port: 5179,
    proxy: {
      // dev: forward API + SSE chat stream to the Python steering server
      "/api": { target: API_TARGET, changeOrigin: true },
    },
  },
});
