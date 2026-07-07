import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import { TanStackRouterVite } from "@tanstack/router-plugin/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  envPrefix: ["VITE_", "FRONTEND__"],
  plugins: [
    TanStackRouterVite({
      target: "react",
      autoCodeSplitting: true,
      routesDirectory: "./src/routes",
      generatedRouteTree: "./src/routeTree.gen.ts",
      routeFileIgnorePattern: "/-",
    }),
    react(),
    tailwindcss(),
  ],
  resolve: { alias: { "@": path.resolve(import.meta.dirname, "src") } },
  server: { port: Number(process.env.FRONTEND__PORT ?? 3001) },
  build: { sourcemap: false },
});
