export function uniqueStrings(values: readonly string[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const value of values) {
    const trimmed = value.trim();
    if (!trimmed) continue;
    if (seen.has(trimmed)) continue;
    seen.add(trimmed);
    result.push(trimmed);
  }
  return result;
}

export function getCategories(data: {
  category?: string;
  categories?: string[];
}): string[] {
  const categories = Array.isArray(data.categories) ? data.categories : [];
  const category = typeof data.category === "string" ? [data.category] : [];
  return uniqueStrings([...category, ...categories]);
}

