import { defineConfig } from "astro/config";

function normalizeBase(base) {
  if (!base) return "/";
  const withLeadingSlash = base.startsWith("/") ? base : `/${base}`;
  if (withLeadingSlash === "/") return "/";
  return withLeadingSlash.endsWith("/")
    ? withLeadingSlash.slice(0, -1)
    : withLeadingSlash;
}

function defaultSite() {
  const repo = process.env.GITHUB_REPOSITORY;
  if (!repo) return "http://localhost:4321";
  const [owner] = repo.split("/");
  if (!owner) return "http://localhost:4321";
  return `https://${owner}.github.io`;
}

function defaultBase() {
  const repo = process.env.GITHUB_REPOSITORY;
  if (!repo) return "/";
  const [owner, name] = repo.split("/");
  if (!owner || !name) return "/";
  const isUserOrOrgPages = name.toLowerCase() === `${owner.toLowerCase()}.github.io`;
  return isUserOrOrgPages ? "/" : `/${name}`;
}

export default defineConfig({
  output: "static",
  site: process.env.SITE || defaultSite(),
  base: normalizeBase(process.env.BASE_PATH || defaultBase()),
  trailingSlash: "always"
});

