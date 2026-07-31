// @ts-check
import { defineConfig } from "astro/config";
import tailwindcss from "@tailwindcss/vite";
import sitemap from "@astrojs/sitemap";

// The repo root is the parent of this site/ directory.
// All companion content (projects/, setup/, README.md) is read from there.
const repoRoot = new URL("../", import.meta.url).pathname;

// https://astro.build/config
export default defineConfig({
  site: "https://under-the-hood.example.com",
  vite: {
    plugins: [tailwindcss()],
    resolve: {
      alias: {
        // Allow importing repo content at build time if ever needed.
        "@repo": repoRoot,
      },
    },
  },
  integrations: [sitemap()],
  markdown: {
    shikiConfig: {
      // A dark, low-contrast theme that fits the minimal aesthetic.
      theme: "github-dark-default",
      wrap: true,
    },
  },
});
