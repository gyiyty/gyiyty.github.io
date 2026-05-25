import fs from "node:fs/promises";
import path from "node:path";
import matter from "gray-matter";
import { siteConfig } from "../src/lib/site-config.mjs";

const root = process.cwd();
const notesRoot = path.resolve(root, siteConfig.importedNotesDir);
const assetsRoot = path.resolve(root, siteConfig.importedAssetsDir);
const target = process.argv[2];

function usage() {
  console.error("Usage: npm.cmd run delete:note -- <slug|title|source-file>");
  console.error("Example: npm.cmd run delete:note -- paxos算法");
}

function slugify(value) {
  return String(value || "")
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/\.md$/i, "")
    .replace(/[^\p{Letter}\p{Number}]+/gu, "-")
    .replace(/^-+|-+$/g, "")
    .toLowerCase();
}

function normalizeTarget(value) {
  return slugify(path.basename(String(value || ""), path.extname(String(value || ""))));
}

async function readNotes() {
  const entries = await fs.readdir(notesRoot, { withFileTypes: true });
  const notes = [];
  for (const entry of entries) {
    if (!entry.isFile() || !entry.name.endsWith(".md")) continue;
    const filePath = path.join(notesRoot, entry.name);
    const raw = await fs.readFile(filePath, "utf8");
    const parsed = matter(raw);
    notes.push({
      filePath,
      fileName: entry.name,
      slug: path.basename(entry.name, ".md"),
      raw,
      content: parsed.content,
      data: parsed.data
    });
  }
  return notes;
}

function assetNamesFromContent(content) {
  const names = new Set();
  const escapedPrefix = siteConfig.publicAssetPrefix.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const pattern = new RegExp(`${escapedPrefix}/([^\\s)\\]'"<>]+)`, "g");
  let match;
  while ((match = pattern.exec(content))) {
    names.add(decodeURIComponent(match[1]));
  }
  return names;
}

function matchesTarget(note, normalizedTarget, rawTarget) {
  const sourceBase = note.data.sourcePath
    ? path.basename(note.data.sourcePath, path.extname(note.data.sourcePath))
    : "";
  return [
    note.slug,
    note.data.title,
    sourceBase,
    note.fileName
  ].some((value) => value === rawTarget || normalizeTarget(value) === normalizedTarget);
}

async function deleteIfExists(filePath) {
  try {
    await fs.rm(filePath, { force: true });
    return true;
  } catch (error) {
    if (error.code === "ENOENT") return false;
    throw error;
  }
}

if (!target) {
  usage();
  process.exit(1);
}

const notes = await readNotes();
const normalizedTarget = normalizeTarget(target);
const matches = notes.filter((note) => matchesTarget(note, normalizedTarget, target));

if (matches.length === 0) {
  console.error(`No imported note matched: ${target}`);
  process.exit(1);
}

if (matches.length > 1) {
  console.error(`Multiple imported notes matched: ${target}`);
  for (const note of matches) {
    console.error(`- ${note.slug} (${note.data.sourcePath || note.fileName})`);
  }
  process.exit(1);
}

const note = matches[0];
const noteAssets = assetNamesFromContent(note.content);
const otherAssetRefs = new Set();
for (const other of notes) {
  if (other.filePath === note.filePath) continue;
  for (const assetName of assetNamesFromContent(other.content)) {
    otherAssetRefs.add(assetName);
  }
}

const deletedAssets = [];
const sharedAssets = [];
for (const assetName of noteAssets) {
  if (otherAssetRefs.has(assetName)) {
    sharedAssets.push(assetName);
    continue;
  }
  const assetPath = path.resolve(assetsRoot, assetName);
  if (!assetPath.startsWith(assetsRoot + path.sep)) {
    throw new Error(`Refusing to delete asset outside imported asset directory: ${assetName}`);
  }
  if (await deleteIfExists(assetPath)) {
    deletedAssets.push(assetName);
  }
}

await fs.rm(note.filePath, { force: true });

console.log(`Deleted note: ${path.relative(root, note.filePath)}`);
if (deletedAssets.length > 0) {
  console.log("Deleted assets:");
  for (const asset of deletedAssets) console.log(`- ${siteConfig.publicAssetPrefix}/${asset}`);
}
if (sharedAssets.length > 0) {
  console.log("Skipped shared assets:");
  for (const asset of sharedAssets) console.log(`- ${siteConfig.publicAssetPrefix}/${asset}`);
}
