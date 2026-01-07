export function toUrlSlug(value: string): string {
  const normalized = value
    .trim()
    .toLowerCase()
    .replaceAll("&", " and ")
    .replaceAll("+", " plus ")
    .replaceAll("#", " sharp ");

  const collapsed = normalized.replace(/\s+/g, "-");
  const cleaned = collapsed
    .replace(/[^\p{L}\p{N}-]+/gu, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "");
  return cleaned || "untitled";
}
