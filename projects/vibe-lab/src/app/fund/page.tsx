import marketBrief from "@/generated/market-brief.json";
import { AppShell } from "@/components/app-shell";
import { FundAssistant } from "@/components/fund-assistant";
import type { MarketBrief } from "@/lib/types";

const brief = marketBrief as MarketBrief;

export default function FundPage() {
  return (
    <AppShell
      activePath="/fund"
      title="基金助手"
      description="录入你的真实持仓，拉基金画像和重仓股公告，再让模型给出更像投研助手而不是玩具 Demo 的判断。"
    >
      <FundAssistant marketBrief={brief} />
    </AppShell>
  );
}
