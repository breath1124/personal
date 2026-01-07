import type { APIRoute } from "astro";

export const GET: APIRoute = ({ site }) => {
  if (!site) {
    return new Response("Missing `site` config.", { status: 500 });
  }

  const base = import.meta.env.BASE_URL;
  const sitemapUrl = new URL(`${base}sitemap.xml`, site).toString();
  const body = `User-agent: *\nAllow: /\nSitemap: ${sitemapUrl}\n`;

  return new Response(body, {
    headers: {
      "Content-Type": "text/plain; charset=utf-8"
    }
  });
};

