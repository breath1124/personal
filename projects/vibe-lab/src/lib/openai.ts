"use client";

import type { AiSettings } from "@/lib/types";

type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

function normalizeBaseUrl(baseUrl: string) {
  return baseUrl.replace(/\/+$/, "");
}

export async function requestModelAnalysis(
  settings: AiSettings,
  messages: ChatMessage[]
) {
  const endpoint = `${normalizeBaseUrl(settings.baseUrl)}/chat/completions`;

  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${settings.apiKey}`
    },
    body: JSON.stringify({
      model: settings.model,
      temperature: settings.temperature,
      messages
    })
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || "模型请求失败");
  }

  const payload = (await response.json()) as {
    choices?: Array<{ message?: { content?: string } }>;
  };

  const content = payload.choices?.[0]?.message?.content?.trim();

  if (!content) {
    throw new Error("模型没有返回可用内容");
  }

  return content;
}
