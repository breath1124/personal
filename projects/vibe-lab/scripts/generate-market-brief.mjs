import fs from "node:fs/promises";
import path from "node:path";
import vm from "node:vm";

const FEATURED_FUNDS = ["161725", "110022", "005827"];

function absolutePath(relativePath) {
  return path.join(process.cwd(), relativePath);
}

async function fetchText(url) {
  const response = await fetch(url, {
    headers: {
      "User-Agent": "Mozilla/5.0 Codex Market Brief"
    }
  });

  if (!response.ok) {
    throw new Error(`Request failed: ${url} (${response.status})`);
  }

  return response.text();
}

function parseFundScript(script) {
  const context = {};
  vm.createContext(context);
  vm.runInContext(script, context);
  return context;
}

function toNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizeNoticePayload(text) {
  return text.replace(/^[^(]+\(/, "").replace(/\)\s*$/, "");
}

async function loadFundDigest(code) {
  const script = await fetchText(
    `https://fund.eastmoney.com/pingzhongdata/${code}.js?v=${Date.now()}`
  );
  const data = parseFundScript(script);
  const navTrend = data.Data_netWorthTrend ?? [];
  const latestPoint = navTrend.at(-1);
  const fluctuationScale = data.Data_fluctuationScale?.series ?? [];
  const latestScale = fluctuationScale.at(-1);
  const manager = data.Data_currentFundManager?.[0];

  return {
    code,
    name: data.fS_name ?? code,
    latestNav: latestPoint?.y ?? null,
    navDate: latestPoint ? new Date(latestPoint.x).toISOString() : null,
    monthReturn: toNumber(data.syl_1y),
    quarterReturn: toNumber(data.syl_3y),
    yearReturn: toNumber(data.syl_1n),
    manager: manager?.name ?? null,
    scale: latestScale ? `${latestScale.y} 亿` : null,
    stockCodes: (data.stockCodesNew ?? []).map((item) => item.replace(/^[01]\./, ""))
  };
}

async function loadNotices(stockCode, limit = 2) {
  const text = await fetchText(
    "https://np-anotice-stock.eastmoney.com/api/security/ann" +
      `?cb=callback&sr=-1&page_size=${limit}&page_index=1&ann_type=A&client_source=web&stock_list=${stockCode}`
  );

  const payload = JSON.parse(normalizeNoticePayload(text));
  const list = payload?.data?.list ?? [];
  return list.map((item) => ({
    title: item.title_ch ?? item.title ?? "",
    stockCode: item.codes?.[0]?.stock_code ?? stockCode,
    shortName: item.codes?.[0]?.short_name ?? "",
    noticeDate: item.notice_date ?? "",
    columnName: item.columns?.map((column) => column.column_name).join(" / ") ?? ""
  }));
}

function buildPulse(featuredFunds, notices) {
  const strongestFund = [...featuredFunds]
    .filter((item) => typeof item.monthReturn === "number")
    .sort((a, b) => (b.monthReturn ?? 0) - (a.monthReturn ?? 0))[0];

  const weakestFund = [...featuredFunds]
    .filter((item) => typeof item.monthReturn === "number")
    .sort((a, b) => (a.monthReturn ?? 0) - (b.monthReturn ?? 0))[0];

  const riskNoticeCount = notices.filter((item) =>
    /(留置|处罚|诉讼|亏损|减持|风险提示)/.test(item.title)
  ).length;

  return [
    strongestFund
      ? {
          title: "相对强势基金",
          detail: `${strongestFund.name} 最近 1 个月 ${strongestFund.monthReturn}% 。强势不代表立刻追高，更适合拿来观察风格是否继续占优。`,
          tone: "positive"
        }
      : {
          title: "基金走势",
          detail: "当前没有拿到足够的收益率数据。",
          tone: "calm"
        },
    weakestFund
      ? {
          title: "相对承压基金",
          detail: `${weakestFund.name} 最近 1 个月 ${weakestFund.monthReturn}% 。如果这也是你的持仓，需要结合仓位和主题逻辑重新判断。`,
          tone: "watch"
        }
      : {
          title: "回撤观察",
          detail: "当前没有拿到足够的回撤数据。",
          tone: "calm"
        },
    {
      title: "公告信号",
      detail:
        riskNoticeCount > 0
          ? `最近抓到 ${riskNoticeCount} 条偏风险公告，建议在做加仓前先复核对应重仓股的事件性质。`
          : "最近公告面没有明显高风险关键词，市场情绪相对平稳。",
      tone: riskNoticeCount > 0 ? "watch" : "calm"
    }
  ];
}

async function main() {
  const featuredFunds = [];
  const noticeDigest = [];

  for (const code of FEATURED_FUNDS) {
    const digest = await loadFundDigest(code);
    featuredFunds.push(digest);

    const topStockCodes = digest.stockCodes.slice(0, 2);
    for (const stockCode of topStockCodes) {
      const notices = await loadNotices(stockCode, 2);
      noticeDigest.push(...notices);
    }
  }

  const uniqueNoticeDigest = Array.from(
    new Map(
      noticeDigest.map((item) => [`${item.stockCode}-${item.title}`, item])
    ).values()
  )
    .sort((a, b) => String(b.noticeDate).localeCompare(String(a.noticeDate)))
    .slice(0, 8);

  const output = {
    generatedAt: new Date().toISOString(),
    featuredFunds: featuredFunds.map(({ stockCodes, ...rest }) => rest),
    noticeDigest: uniqueNoticeDigest,
    marketPulse: buildPulse(featuredFunds, uniqueNoticeDigest)
  };

  await fs.mkdir(absolutePath("src/generated"), { recursive: true });
  await fs.writeFile(
    absolutePath("src/generated/market-brief.json"),
    `${JSON.stringify(output, null, 2)}\n`
  );
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
