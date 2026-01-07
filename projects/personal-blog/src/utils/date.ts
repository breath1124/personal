import { SITE } from "../config/site";

export function formatDate(date: Date): string {
  return new Intl.DateTimeFormat(SITE.locale, {
    year: "numeric",
    month: "2-digit",
    day: "2-digit"
  }).format(date);
}

