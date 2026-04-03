import type { AnalysisSection, FundSnapshot, HoldingRecord, NoticeItem } from "@/lib/types";

function maxDrawdown(values: number[]) {
  let peak = values[0] ?? 0;
  let max = 0;

  for (const value of values) {
    if (value > peak) peak = value;
    if (peak === 0) continue;
    const drawdown = ((peak - value) / peak) * 100;
    if (drawdown > max) max = drawdown;
  }

  return Number(max.toFixed(2));
}

function latestScaleMomentum(snapshot: FundSnapshot) {
  const latest = snapshot.fluctuationScale.at(-1);
  return latest?.mom ?? null;
}

function noticeSignals(notices: NoticeItem[]) {
  const positives = notices.filter((item) => /(回购|分红|增持|业绩预增|中标)/.test(item.title));
  const warnings = notices.filter(
    (item) => /(减持|处罚|诉讼|留置|亏损|下修|风险提示)/.test(item.title)
  );

  return { positives, warnings };
}

export function buildHoldingSections(
  holding: HoldingRecord,
  snapshot: FundSnapshot,
  notices: NoticeItem[]
): AnalysisSection[] {
  const values = snapshot.navTrend.slice(-120).map((item) => item.value);
  const drawdown = values.length > 2 ? maxDrawdown(values) : null;
  const signals = noticeSignals(notices);
  const marketValue = snapshot.latestNav ? snapshot.latestNav * holding.units : null;
  const costValue = holding.averageCost * holding.units;
  const pnlPct =
    marketValue && costValue
      ? Number((((marketValue - costValue) / costValue) * 100).toFixed(2))
      : null;

  const overview = [
    `${snapshot.name} 当前单位净值 ${snapshot.latestNav ?? "-"}，近 1 月 ${snapshot.returns.month ?? "-"}%，近 3 月 ${snapshot.returns.quarter ?? "-"}%。`,
    pnlPct === null
      ? "当前缺少完整估值，无法计算你的持仓盈亏。"
      : `按你录入的成本测算，当前浮动收益约 ${pnlPct}%。`,
    drawdown === null
      ? "近 120 个交易日回撤数据不足。"
      : `近 120 个交易日最大回撤约 ${drawdown}%，能反映这只基金最近的波动压力。`
  ];

  const structure = [
    snapshot.manager
      ? `当前基金经理为 ${snapshot.manager.name}，任职时长 ${snapshot.manager.workTime}，管理规模 ${snapshot.manager.fundSize}。`
      : "当前未拿到基金经理信息。",
    snapshot.assetAllocation.at(-1)?.stockRatio
      ? `最近一期股票仓位约 ${snapshot.assetAllocation.at(-1)?.stockRatio}% 。`
      : "最近一期资产配置数据暂不可用。",
    latestScaleMomentum(snapshot)
      ? `最近一期基金规模变化为 ${latestScaleMomentum(snapshot)}。`
      : "最近一期规模变化暂不可用。"
  ];

  const action = [
    pnlPct !== null && pnlPct > 15
      ? "如果你的仓位已经偏重，可以考虑先做一次再平衡，而不是继续无条件追高。"
      : "如果你还在建立仓位，优先判断这只基金是否仍符合你的原始配置角色。",
    snapshot.returns.quarter !== null && snapshot.returns.quarter < -12
      ? "最近一个季度回撤较深，任何加仓都更适合分批，而不是一次性补仓。"
      : "近期波动尚可控，更重要的是看主题和持仓逻辑是否仍清晰。",
    signals.warnings.length > 0
      ? `最近公告里有 ${signals.warnings.length} 条偏风险信号，建议重点复核对应重仓股。`
      : "最近公告没有出现明显高风险关键词，可以把重点放在估值和仓位纪律上。"
  ];

  const noticeSummary = [
    ...signals.warnings.slice(0, 2).map((item) => `风险提示: ${item.shortName} ${item.title}`),
    ...signals.positives.slice(0, 2).map((item) => `积极信号: ${item.shortName} ${item.title}`)
  ];

  return [
    { title: "组合概览", items: overview },
    { title: "结构判断", items: structure },
    { title: "动作建议", items: action },
    {
      title: "公告观察",
      items: noticeSummary.length > 0 ? noticeSummary : ["最近没有抓到明显的公告信号。"]
    }
  ];
}

export function buildPortfolioBrief(
  holdings: HoldingRecord[],
  snapshots: FundSnapshot[],
  noticeMap: Record<string, NoticeItem[]>
) {
  const totalCost = holdings.reduce((sum, item) => sum + item.averageCost * item.units, 0);
  const totalValue = holdings.reduce((sum, holding) => {
    const snapshot = snapshots.find((item) => item.code === holding.fundCode);
    return sum + (snapshot?.latestNav ?? 0) * holding.units;
  }, 0);

  const totalPnL =
    totalCost > 0 ? Number((((totalValue - totalCost) / totalCost) * 100).toFixed(2)) : 0;

  const highlights = holdings.map((holding) => {
    const snapshot = snapshots.find((item) => item.code === holding.fundCode);
    const notices = noticeMap[holding.fundCode] ?? [];
    const riskHits = noticeSignals(notices).warnings.length;
    return {
      code: holding.fundCode,
      name: snapshot?.name ?? holding.fundCode,
      monthReturn: snapshot?.returns.month ?? null,
      quarterReturn: snapshot?.returns.quarter ?? null,
      riskHits
    };
  });

  return {
    totalCost: Number(totalCost.toFixed(2)),
    totalValue: Number(totalValue.toFixed(2)),
    totalPnL,
    highlights
  };
}
