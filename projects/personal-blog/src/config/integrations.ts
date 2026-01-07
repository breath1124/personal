export type AnalyticsProvider = "none" | "plausible" | "umami" | "google";

export const ANALYTICS: {
  provider: AnalyticsProvider;
  plausibleDomain?: string;
  plausibleScriptSrc?: string;
  umamiScriptSrc?: string;
  umamiWebsiteId?: string;
  googleMeasurementId?: string;
} = {
  provider: "none"
};

export type GiscusMapping =
  | "pathname"
  | "url"
  | "title"
  | "og:title"
  | "specific"
  | "number";

export type GiscusConfig = {
  enabled: boolean;
  repo: string;
  repoId: string;
  category: string;
  categoryId: string;
  mapping: GiscusMapping;
  strict: "0" | "1";
  reactionsEnabled: "0" | "1";
  emitMetadata: "0" | "1";
  inputPosition: "top" | "bottom";
  theme: string;
  lang: string;
};

export const GISCUS: GiscusConfig = {
  enabled: false,
  repo: "",
  repoId: "",
  category: "General",
  categoryId: "",
  mapping: "pathname",
  strict: "0",
  reactionsEnabled: "1",
  emitMetadata: "0",
  inputPosition: "top",
  theme: "preferred_color_scheme",
  lang: "zh-CN"
};

