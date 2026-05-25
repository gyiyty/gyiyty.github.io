import rss from "@astrojs/rss";
import { getCollection } from "astro:content";
import { publicNotes, siteConfig, sortByDateDesc } from "../lib/site";

export async function GET(context) {
  const notes = sortByDateDesc(publicNotes(await getCollection("notes")));
  return rss({
    title: siteConfig.siteTitle,
    description: siteConfig.siteDescription,
    site: context.site,
    items: notes.map((note) => ({
      title: note.data.title,
      pubDate: note.data.date,
      description: note.data.description,
      link: `/notes/${note.slug}/`
    }))
  });
}
