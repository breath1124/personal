import type { APIRoute } from "astro";
import { getCollection } from "astro:content";
import { SITE } from "../config/site";
import { escapeXml } from "../utils/xml";

export const GET: APIRoute = async ({ site }) => {
  if (!site) {
    return new Response("Missing `site` config.", { status: 500 });
  }

  const base = import.meta.env.BASE_URL;
  const posts = (await getCollection("blog", ({ data }) => !data.draft)).sort(
    (a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf()
  );

  const feedUrl = new URL(`${base}rss.xml`, site).toString();
  const homeUrl = new URL(base, site).toString();

  const lastBuildDate =
    posts[0]?.data.updatedDate ?? posts[0]?.data.pubDate ?? new Date();

  const items = posts
    .map((post) => {
      const link = new URL(`${base}blog/${post.slug}/`, site).toString();
      return (
        `    <item>\n` +
        `      <title>${escapeXml(post.data.title)}</title>\n` +
        `      <link>${escapeXml(link)}</link>\n` +
        `      <guid isPermaLink="true">${escapeXml(link)}</guid>\n` +
        `      <pubDate>${post.data.pubDate.toUTCString()}</pubDate>\n` +
        `      <description>${escapeXml(post.data.description)}</description>\n` +
        `    </item>\n`
      );
    })
    .join("");

  const xml =
    `<?xml version="1.0" encoding="UTF-8"?>\n` +
    `<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">\n` +
    `  <channel>\n` +
    `    <title>${escapeXml(SITE.title)}</title>\n` +
    `    <link>${escapeXml(homeUrl)}</link>\n` +
    `    <description>${escapeXml(SITE.description)}</description>\n` +
    `    <language>${escapeXml(SITE.locale)}</language>\n` +
    `    <lastBuildDate>${lastBuildDate.toUTCString()}</lastBuildDate>\n` +
    `    <atom:link href="${escapeXml(
      feedUrl
    )}" rel="self" type="application/rss+xml" />\n` +
    items +
    `  </channel>\n` +
    `</rss>\n`;

  return new Response(xml, {
    headers: {
      "Content-Type": "application/rss+xml; charset=utf-8"
    }
  });
};

