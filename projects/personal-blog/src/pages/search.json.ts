import type { APIRoute } from "astro";
import { getCollection } from "astro:content";
import { getCategories } from "../utils/taxonomy";

export const prerender = true;

export const GET: APIRoute = async () => {
  const base = import.meta.env.BASE_URL;
  const posts = (await getCollection("blog", ({ data }) => !data.draft)).sort(
    (a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf()
  );

  const items = posts.map((post) => ({
    title: post.data.title,
    description: post.data.description,
    pubDate: post.data.pubDate.toISOString(),
    tags: post.data.tags ?? [],
    categories: getCategories(post.data),
    url: `${base}blog/${post.slug}/`
  }));

  return new Response(JSON.stringify({ items }), {
    headers: {
      "Content-Type": "application/json; charset=utf-8"
    }
  });
};

