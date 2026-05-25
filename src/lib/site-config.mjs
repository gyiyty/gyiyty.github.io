export const siteConfig = {
  siteTitle: "个人知识库",
  siteDescription: "从 Obsidian 发布的长期笔记、想法和技术记录。",
  authorName: "YPC",
  siteUrl: process.env.SITE_URL || "https://username.github.io",
  base: "/",
  vaultPath: "E:/ypc/mynote/cardbox",
  publishTag: "#publish",
  hiddenTags: ["#publish", "#永久笔记"],
  notesSourceGlob: "**/*.md",
  importedNotesDir: "src/content/notes",
  importedAssetsDir: "public/notes-assets",
  publicAssetPrefix: "/notes-assets",
  wallpaperSourceDir: "wallpaper",
  wallpaperOutputDir: "public/wallpaper"
};
