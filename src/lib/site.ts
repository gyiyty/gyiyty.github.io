import { siteConfig } from "./site-config.mjs";

export { siteConfig };

export function normalizeTag(tag: string): string {
  return tag.trim().replace(/^#/, "");
}

export function displayTag(tag: string): string {
  return normalizeTag(tag);
}

export function makeTagSlug(tag: string): string {
  return encodeURIComponent(normalizeTag(tag));
}

export function tagFromSlug(slug: string): string {
  return decodeURIComponent(slug);
}

export function formatDate(date?: Date): string {
  if (!date) return "";
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit"
  }).format(date);
}

export function sortByDateDesc<T extends { data: { date?: Date }; slug: string }>(items: T[]): T[] {
  return [...items].sort((a, b) => {
    const aTime = a.data.date?.getTime() ?? 0;
    const bTime = b.data.date?.getTime() ?? 0;
    if (aTime !== bTime) return bTime - aTime;
    return a.slug.localeCompare(b.slug, "zh-CN");
  });
}

export function publicNotes<T extends { data: { draft?: boolean } }>(items: T[]): T[] {
  return items.filter((item) => !item.data.draft);
}

export function collectTags(notes: Array<{ data: { tags?: string[] } }>): Array<{ tag: string; count: number }> {
  const counts = new Map<string, number>();
  for (const note of notes) {
    for (const tag of note.data.tags ?? []) {
      const normalized = normalizeTag(tag);
      counts.set(normalized, (counts.get(normalized) ?? 0) + 1);
    }
  }
  return [...counts.entries()]
    .map(([tag, count]) => ({ tag, count }))
    .sort((a, b) => b.count - a.count || a.tag.localeCompare(b.tag, "zh-CN"));
}
