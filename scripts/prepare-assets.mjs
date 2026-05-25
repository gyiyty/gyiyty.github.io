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
  },
  {
    source: "nina.png",
    transparentWhite: true,
    outputs: [{ file: "nina.webp", width: 96, quality: 86 }]
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
    if (job.transparentWhite) {
      await prepareTransparentWhiteAsset(sourcePath, outputPath, output);
    } else {
      await sharp(sourcePath)
        .resize({ width: output.width, withoutEnlargement: true })
        .webp({ quality: output.quality })
        .toFile(outputPath);
    }
    console.log(`Prepared ${path.relative(root, outputPath)}`);
  }
}

async function prepareTransparentWhiteAsset(sourcePath, outputPath, output) {
  const { data, info } = await sharp(sourcePath)
    .resize({ width: output.width, withoutEnlargement: true })
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const background = findConnectedWhiteBackground(data, info.width, info.height, info.channels);

  for (const index of background) {
    data[index + 3] = 0;
  }

  await sharp(data, {
    raw: {
      width: info.width,
      height: info.height,
      channels: info.channels
    }
  })
    .webp({ quality: output.quality })
    .toFile(outputPath);
}

function findConnectedWhiteBackground(data, width, height, channels) {
  const visited = new Uint8Array(width * height);
  const background = [];
  const queue = [];

  const pushIfBackground = (x, y) => {
    if (x < 0 || y < 0 || x >= width || y >= height) {
      return;
    }

    const pixel = y * width + x;
    if (visited[pixel]) {
      return;
    }

    visited[pixel] = 1;
    const index = pixel * channels;
    if (isNearWhite(data[index], data[index + 1], data[index + 2])) {
      queue.push(pixel);
      background.push(index);
    }
  };

  for (let x = 0; x < width; x += 1) {
    pushIfBackground(x, 0);
    pushIfBackground(x, height - 1);
  }

  for (let y = 1; y < height - 1; y += 1) {
    pushIfBackground(0, y);
    pushIfBackground(width - 1, y);
  }

  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const pixel = queue[cursor];
    const x = pixel % width;
    const y = Math.floor(pixel / width);
    pushIfBackground(x + 1, y);
    pushIfBackground(x - 1, y);
    pushIfBackground(x, y + 1);
    pushIfBackground(x, y - 1);
  }

  return background;
}

function isNearWhite(red, green, blue) {
  const max = Math.max(red, green, blue);
  const min = Math.min(red, green, blue);
  const luma = red * 0.2126 + green * 0.7152 + blue * 0.0722;

  return luma > 225 && max - min < 34;
}
