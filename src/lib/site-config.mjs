export const siteConfig = {
  siteTitle: "个人知识库",
  siteDescription: "从 Obsidian 发布的长期笔记、想法和技术记录。",
  authorName: "YPC",
  about: {
    eyebrow: "About the Author",
    title: "YPC",
    intro:
      "我在这里整理公开笔记、长期想法和学习记录。内容从 Obsidian 中筛选发布，关注技术、知识管理和持续写作。",
    meta: [
      {
        label: "GitHub",
        value: "github.com"
      },
      {
        label: "Email",
        value: "hello@example.com"
      }
    ],
    gallery: [
      {
        src: "/wallpaper/about-band.webp",
        alt: "红色光影背景",
        caption: "写作现场"
      },
      {
        src: "/wallpaper/hero-home.webp",
        alt: "站点首页背景图",
        caption: "知识库"
      },
      {
        src: "/wallpaper/hero-note.webp",
        alt: "文章页背景图",
        caption: "公开笔记"
      }
    ],
    sections: [
      {
        eyebrow: "Focus",
        title: "关注主题",
        body: "技术实践、系统思考、知识管理和那些值得长期复看的问题。"
      },
      {
        eyebrow: "Workflow",
        title: "写作方式",
        body: "先在 Obsidian 中沉淀笔记，再筛选适合公开的内容发布到这个静态网站。"
      }
    ]
  },
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
