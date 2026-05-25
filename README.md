# 个人知识网站

这是一个基于 Astro 的静态个人网站，用于把 Obsidian 中的公开 Markdown 笔记发布到 GitHub Pages。站点支持标题搜索、tag 浏览、LaTeX 渲染、表格、代码块、RSS、sitemap，并使用 `wallpaper` 目录中的图片生成优化后的 WebP 视觉资产。

## 功能

- 只发布带 `#publish` 的 Obsidian 笔记。
- 导入时隐藏系统/类型 tag，默认隐藏 `#publish` 和 `#永久笔记`。
- 支持 Obsidian 常见 Markdown 写法、GFM 表格、任务列表、代码块。
- 支持行内公式 `$...$` 和块级公式 `$$...$$`，通过 KaTeX 渲染。
- 支持常见 Obsidian 附件引用，导入时复制到 `public/notes-assets`。
- 提供文章列表、文章详情、tag 聚合、标题搜索、关于页、404、RSS 和 sitemap。
- GitHub Actions 自动构建并部署到 GitHub Pages。

## 目录

```text
src/content/notes/      导入后的公开笔记
public/notes-assets/    导入后的公开附件
wallpaper/              原始视觉参考图
public/wallpaper/       构建时生成的优化图片
scripts/import-notes.mjs
scripts/prepare-assets.mjs
src/lib/site-config.mjs
```

## 配置

站点配置位于 `src/lib/site-config.mjs`：

```js
export const siteConfig = {
  siteTitle: "个人知识库",
  siteDescription: "从 Obsidian 发布的长期笔记、想法和技术记录。",
  authorName: "YPC",
  siteUrl: process.env.SITE_URL || "https://username.github.io",
  base: "/",
  vaultPath: "E:/ypc/mynote/cardbox",
  publishTag: "#publish",
  hiddenTags: ["#publish", "#永久笔记"]
};
```

如需更换 Obsidian Vault 路径、站点标题、作者名或隐藏 tag，修改这个文件即可。

## 使用

安装依赖：

```powershell
npm.cmd install
```

从 Obsidian 导入公开笔记：

```powershell
npm.cmd run import:notes
```

删除已导入文章及其未被其他文章引用的附件：

```powershell
npm.cmd run delete:note -- paxos算法
```

参数可以是导入后的 slug、文章标题或源 Markdown 文件名。该命令只删除仓库内的导入结果，不会修改 Obsidian Vault。

本地开发预览：

```powershell
npm.cmd run dev
```

构建静态站点：

```powershell
npm.cmd run build
```

## 发布笔记

1. 在 Obsidian 笔记中添加 `#publish`。
2. 确认不想展示的类型 tag 已加入 `hiddenTags`。
3. 运行 `npm.cmd run import:notes`。
4. 运行 `npm.cmd run build` 检查构建。
5. 提交并推送到 GitHub，GitHub Actions 会部署到 GitHub Pages。

没有任何公开笔记时，项目会保留一个 `draft` 占位文件，保证空内容状态也能正常构建。

## GitHub Pages

当前按用户主页配置，目标地址形如：

```text
https://<username>.github.io
```

如果改成项目页，例如 `https://<username>.github.io/<repo>/`，需要同步调整 `src/lib/site-config.mjs` 中的 `base`，并检查 GitHub Actions 中的 `SITE_URL`。

## 注意事项

- `wallpaper` 中保留原图；`npm.cmd run build` 会生成 `public/wallpaper` 下的 WebP 优化图。
- `public/wallpaper`、`dist`、`.astro`、`node_modules` 不进入版本控制。
- `AGENTS.md` 是本地代理协作说明，已被 `.gitignore` 忽略。
