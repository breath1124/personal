import type { APIRoute } from "astro";
import { getCollection } from "astro:content";
import { toUrlSlug } from "../utils/slug";
import { getCategories } from "../utils/taxonomy";
import { escapeXml } from "../utils/xml";

export const GET: APIRoute = async ({ site }) => {
  if (!site) {
    return new Response("Missing `site` config.", { status: 500 });
  }

  const base = import.meta.env.BASE_URL;
  const posts = (await getCollection("blog", ({ data }) => !data.draft)).sort(
    (a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf()
  );

  const categories = new Set<string>();
  for (const post of posts) {
    for (const category of getCategories(post.data)) {
      categories.add(category);
    }
  }

  const sortedCategories = [...categories].sort((a, b) =>
    a.localeCompare(b, "zh-CN")
  );

  const urls: Array<{ loc: string; lastmod?: Date }> = [
    {
      loc: new URL(base, site).toString(),
      lastmod: posts[0]?.data.updatedDate ?? posts[0]?.data.pubDate
    },
    { loc: new URL(`${base}blog/`, site).toString() },
    { loc: new URL(`${base}categories/`, site).toString() },
    ...sortedCategories.map((category) => ({
      loc: new URL(`${base}categories/${toUrlSlug(category)}/`, site).toString()
    })),
    ...posts.map((post) => ({
      loc: new URL(`${base}blog/${post.slug}/`, site).toString(),
      lastmod: post.data.updatedDate ?? post.data.pubDate
    }))
  ];

  const body =
    `<?xml version="1.0" encoding="UTF-8"?>\n` +
    `<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n` +
    urls
      .map(({ loc, lastmod }) => {
        const lastmodLine = lastmod
          ? `<lastmod>${escapeXml(lastmod.toISOString())}</lastmod>`
          : "";
        return `  <url><loc>${escapeXml(loc)}</loc>${lastmodLine}</url>`;
      })
      .join("\n") +
    `\n</urlset>\n`;

  return new Response(body, {
    headers: {
      "Content-Type": "application/xml; charset=utf-8"
    }
  });
};
