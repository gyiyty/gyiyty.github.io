import fs from "node:fs/promises";
import path from "node:path";
import sharp from "sharp";
import { siteConfig } from "../src/lib/site-config.mjs";

const root = process.cwd();
const sourceDir = path.resolve(root, siteConfig.wallpaperSourceDir);
const outputDir = path.resolve(root, siteConfig.wallpaperOutputDir);

const jobs = [
  {
    source: "GBC3.png",
    outputs: [
      { file: "hero-home.webp", width: 2200, quality: 82 },
      { file: "hero-home-mobile.webp", width: 1100, quality: 78 }
    ]
  },
  {
    source: "GBC1.jpg",
    outputs: [
      { file: "hero-note.webp", width: 1800, quality: 82 },
      { file: "hero-note-mobile.webp", width: 900, quality: 78 }
    ]
  },
  {
    source: "GBC2.png",
    outputs: [
      { file: "about-band.webp", width: 2000, quality: 82 },
      { file: "about-band-mobile.webp", width: 1000, quality: 78 }
    ]
  }
];

async function exists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

await fs.mkdir(outputDir, { recursive: true });

for (const job of jobs) {
  const sourcePath = path.join(sourceDir, job.source);
  if (!(await exists(sourcePath))) {
    console.warn(`Missing wallpaper source: ${sourcePath}`);
    continue;
  }

  for (const output of job.outputs) {
    const outputPath = path.join(outputDir, output.file);
    await sharp(sourcePath)
      .resize({ width: output.width, withoutEnlargement: true })
      .webp({ quality: output.quality })
      .toFile(outputPath);
    console.log(`Prepared ${path.relative(root, outputPath)}`);
  }
}
