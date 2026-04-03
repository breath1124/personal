import type { AiSettings, HoldingRecord } from "@/lib/types";

export const AI_SETTINGS_KEY = "cy-lab-ai-settings";
export const HOLDINGS_KEY = "cy-lab-holdings";
export const ORACLE_PROFILE_KEY = "cy-lab-oracle-profile";

export const DEFAULT_AI_SETTINGS: AiSettings = {
  providerName: "OpenAI Compatible",
  baseUrl: "https://api.openai.com/v1",
  model: "gpt-4.1-mini",
  apiKey: "",
  temperature: 0.4
};

export const DEFAULT_HOLDING_DRAFT: Omit<HoldingRecord, "id" | "addedAt"> = {
  fundCode: "",
  units: 1000,
  averageCost: 1,
  thesis: "",
  targetWeight: 20,
  note: ""
};
