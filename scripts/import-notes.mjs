import fs from "node:fs/promises";
import path from "node:path";
import matter from "gray-matter";
import { siteConfig } from "../src/lib/site-config.mjs";

const root = process.cwd();
const vaultRoot = path.resolve(siteConfig.vaultPath);
const outputRoot = path.resolve(root, siteConfig.importedNotesDir);
const assetsRoot = path.resolve(root, siteConfig.importedAssetsDir);
const hiddenTags = new Set(siteConfig.hiddenTags.map(normalizeTag));
const publishTag = normalizeTag(siteConfig.publishTag);
const markdownExtensions = new Set([".md", ".markdown"]);
const assetReferencePattern = /!\[\[([^\]]+)\]\]|\[\[([^\]|]+\.(?:png|jpe?g|gif|webp|svg|pdf|mp3|mp4|wav))(?:\|[^\]]*)?\]\]|!\[([^\]]*)\]\(([^)]+)\)/gi;

function normalizeSlashes(value) {
  return value.replace(/\\/g, "/");
}

function normalizeTag(tag) {
  return String(tag || "")
    .trim()
    .replace(/^#/, "")
    .replace(/\/$/, "");
}

function ensureHashTag(tag) {
  const normalized = normalizeTag(tag);
  return normalized ? `#${normalized}` : "";
}

function slugify(value) {
  return String(value || "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/\.md$/i, "")
    .replace(/[^\p{Letter}\p{Number}]+/gu, "-")
    .replace(/^-+|-+$/g, "")
    .toLowerCase() || "note";
}

function titleFromBody(body, fallback) {
  const heading = body.match(/^#\s+(.+)$/m);
  return heading?.[1]?.trim() || fallback;
}

function normalizeFrontmatterTags(value) {
  if (!value) return [];
  if (Array.isArray(value)) return value.map(ensureHashTag).filter(Boolean);
  if (typeof value === "string") return value.split(/[,\s]+/).map(ensureHashTag).filter(Boolean);
  return [];
}

function extractInlineTags(markdown) {
  const tags = [];
  const pattern = /(^|[\s([{>])#([\p{Letter}\p{Number}_/-]+)/gu;
  let match;
  while ((match = pattern.exec(markdown))) {
    tags.push(`#${match[2]}`);
  }
  return tags;
}

function cleanTags(tags) {
  const seen = new Set();
  const clean = [];
  for (const tag of tags) {
    const normalized = normalizeTag(tag);
    if (!normalized || hiddenTags.has(normalized)) continue;
    if (seen.has(normalized)) continue;
    seen.add(normalized);
    clean.push(normalized);
  }
  return clean.sort((a, b) => a.localeCompare(b, "zh-CN"));
}

async function walk(dir) {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  const files = [];
  for (const entry of entries) {
    if (entry.name.startsWith(".obsidian") || entry.name === ".trash") continue;
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) files.push(...await walk(fullPath));
    if (entry.isFile() && markdownExtensions.has(path.extname(entry.name).toLowerCase())) files.push(fullPath);
  }
  return files;
}

async function buildAssetIndex(dir) {
  const index = new Map();
  async function visit(current) {
    const entries = await fs.readdir(current, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.name.startsWith(".obsidian") || entry.name === ".trash") continue;
      const fullPath = path.join(current, entry.name);
      if (entry.isDirectory()) {
        await visit(fullPath);
      } else {
        const key = entry.name.toLowerCase();
        if (!index.has(key)) index.set(key, fullPath);
      }
    }
  }
  await visit(dir);
  return index;
}

function parseAssetTarget(rawTarget) {
  return decodeURIComponent(rawTarget.trim().split("|")[0].replace(/^<|>$/g, ""));
}

function isRemoteOrAnchor(target) {
  return /^(?:[a-z][a-z\d+.-]*:|#|\/)/i.test(target);
}

async function resolveAsset(notePath, target, assetIndex) {
  if (isRemoteOrAnchor(target)) return null;
  const cleanTarget = parseAssetTarget(target);
  const direct = path.resolve(path.dirname(notePath), cleanTarget);
  try {
    const stat = await fs.stat(direct);
    if (stat.isFile()) return direct;
  } catch {
    // Try by basename below.
  }
  return assetIndex.get(path.basename(cleanTarget).toLowerCase()) || null;
}

async function copyAsset(assetPath, usedAssetNames) {
  const parsed = path.parse(assetPath);
  const baseSlug = slugify(parsed.name);
  let candidate = `${baseSlug}${parsed.ext.toLowerCase()}`;
  let counter = 2;
  while (usedAssetNames.has(candidate)) {
    candidate = `${baseSlug}-${counter}${parsed.ext.toLowerCase()}`;
    counter += 1;
  }
  usedAssetNames.add(candidate);
  await fs.mkdir(assetsRoot, { recursive: true });
  await fs.copyFile(assetPath, path.join(assetsRoot, candidate));
  return `${siteConfig.publicAssetPrefix}/${candidate}`;
}

async function rewriteAssetReferences(markdown, notePath, assetIndex, usedAssetNames) {
  const replacements = [];
  for (const match of markdown.matchAll(assetReferencePattern)) {
    const [full] = match;
    const obsidianImageTarget = match[1];
    const obsidianFileTarget = match[2];
    const markdownAlt = match[3];
    const markdownTarget = match[4];
    const target = obsidianImageTarget || obsidianFileTarget || markdownTarget;
    const resolved = await resolveAsset(notePath, target, assetIndex);
    if (!resolved) continue;
    const publicUrl = await copyAsset(resolved, usedAssetNames);
    const alt = markdownAlt || path.basename(parseAssetTarget(target), path.extname(target));
    const replacement = `![${alt}](${publicUrl})`;
    replacements.push({ start: match.index, end: match.index + full.length, replacement });
  }

  if (replacements.length === 0) return markdown;

  let output = "";
  let cursor = 0;
  for (const item of replacements) {
    output += markdown.slice(cursor, item.start);
    output += item.replacement;
    cursor = item.end;
  }
  output += markdown.slice(cursor);
  return output;
}

function makeFrontmatter(data, title, tags, sourcePath) {
  const output = {
    title,
    date: data.date || undefined,
    description: data.description || data.summary || undefined,
    tags,
    sourcePath: normalizeSlashes(path.relative(vaultRoot, sourcePath))
  };
  return Object.fromEntries(Object.entries(output).filter(([, value]) => value !== undefined && value !== ""));
}

async function writePlaceholder() {
  const placeholder = matter.stringify(
    "这个文件只用于让空内容集合保持可构建状态，不会出现在网站页面中。\n",
    {
      title: "内容占位",
      date: "2026-05-25",
      draft: true,
      tags: []
    }
  );
  await fs.writeFile(path.join(outputRoot, "placeholder.md"), placeholder, "utf8");
}

async function importNotes() {
  await fs.rm(outputRoot, { recursive: true, force: true });
  await fs.rm(assetsRoot, { recursive: true, force: true });
  await fs.mkdir(outputRoot, { recursive: true });
  await fs.mkdir(assetsRoot, { recursive: true });

  const markdownFiles = await walk(vaultRoot);
  const assetIndex = await buildAssetIndex(vaultRoot);
  const usedSlugs = new Set();
  const usedAssetNames = new Set();
  let imported = 0;

  for (const filePath of markdownFiles) {
    const raw = await fs.readFile(filePath, "utf8");
    const parsed = matter(raw);
    const allTags = [
      ...normalizeFrontmatterTags(parsed.data.tags),
      ...extractInlineTags(parsed.content)
    ];
    const normalizedTags = allTags.map(normalizeTag);
    if (!normalizedTags.includes(publishTag)) continue;

    const title = parsed.data.title || titleFromBody(parsed.content, path.basename(filePath, path.extname(filePath)));
    const publicTags = cleanTags(allTags);
    let noteSlug = slugify(title);
    let suffix = 2;
    while (usedSlugs.has(noteSlug)) {
      noteSlug = `${slugify(title)}-${suffix}`;
      suffix += 1;
    }
    usedSlugs.add(noteSlug);

    const content = await rewriteAssetReferences(parsed.content, filePath, assetIndex, usedAssetNames);
    const frontmatter = makeFrontmatter(parsed.data, title, publicTags, filePath);
    const output = matter.stringify(content.trim() + "\n", frontmatter);
    await fs.writeFile(path.join(outputRoot, `${noteSlug}.md`), output, "utf8");
    imported += 1;
  }

  if (imported === 0) {
    await writePlaceholder();
  }

  console.log(`Imported ${imported} public note${imported === 1 ? "" : "s"} from ${vaultRoot}`);
}

importNotes().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
