import { defineConfig } from "astro/config";
import sitemap from "@astrojs/sitemap";
import rehypeAutolinkHeadings from "rehype-autolink-headings";
import rehypeKatex from "rehype-katex";
import rehypeSlug from "rehype-slug";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { remarkObsidianCallouts } from "./src/lib/remark-obsidian-callouts.mjs";
import { remarkObsidianHighlights } from "./src/lib/remark-obsidian-highlights.mjs";
import { remarkObsidianWikiLinks } from "./src/lib/remark-obsidian-wiki-links.mjs";
import { siteConfig } from "./src/lib/site-config.mjs";

export default defineConfig({
  site: siteConfig.siteUrl,
  base: siteConfig.base,
  markdown: {
    remarkPlugins: [remarkGfm, remarkMath, remarkObsidianCallouts, remarkObsidianHighlights, remarkObsidianWikiLinks],
    rehypePlugins: [
      rehypeSlug,
      [
        rehypeAutolinkHeadings,
        {
          behavior: "wrap",
          properties: { className: ["heading-anchor"] }
        }
      ],
      rehypeKatex
    ],
    shikiConfig: {
      theme: "github-dark"
    }
  },
  integrations: [sitemap()]
});
