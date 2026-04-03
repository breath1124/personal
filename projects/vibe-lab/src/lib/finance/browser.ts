"use client";

import type { FundSnapshot, NoticeItem } from "@/lib/types";

type EastMoneyGlobals = {
  fS_name?: string;
  fS_code?: string;
  syl_1n?: string;
  syl_6y?: string;
  syl_3y?: string;
  syl_1y?: string;
  Data_netWorthTrend?: Array<{
    x: number;
    y: number;
    equityReturn?: number;
  }>;
  stockCodesNew?: string[];
  Data_assetAllocation?: {
    categories?: string[];
    series?: Array<{ name: string; data: Array<number | null> }>;
  };
  Data_fluctuationScale?: {
    categories?: string[];
    series?: Array<{ y: number; mom?: string }>;
  };
  Data_holderStructure?: {
    categories?: string[];
    series?: Array<{ name: string; data: Array<number | null> }>;
  };
  Data_currentFundManager?: Array<{
    name?: string;
    workTime?: string;
    fundSize?: string;
  }>;
  Data_performanceEvaluation?: {
    categories?: string[];
    data?: number[];
  };
};

function cleanupEastMoneyGlobals() {
  const globalKeys = [
    "fS_name",
    "fS_code",
    "syl_1n",
    "syl_6y",
    "syl_3y",
    "syl_1y",
    "Data_netWorthTrend",
    "stockCodesNew",
    "Data_assetAllocation",
    "Data_fluctuationScale",
    "Data_holderStructure",
    "Data_currentFundManager",
    "Data_performanceEvaluation"
  ];

  for (const key of globalKeys) {
    Reflect.deleteProperty(window, key);
  }
}

function loadScript(src: string) {
  return new Promise<void>((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.async = true;
    script.onload = () => {
      script.remove();
      resolve();
    };
    script.onerror = () => {
      script.remove();
      reject(new Error(`加载脚本失败: ${src}`));
    };
    document.body.appendChild(script);
  });
}

function toNumber(input: string | undefined) {
  if (!input) return null;
  const parsed = Number(input);
  return Number.isFinite(parsed) ? parsed : null;
}

function normalizeStockCode(rawCode: string) {
  return rawCode.replace(/^[01]\./, "");
}

function buildNoticeDetailUrl(stockCode: string, artCode?: string) {
  if (!artCode) return undefined;
  return `https://data.eastmoney.com/notices/detail/${stockCode}/${artCode}.html`;
}

export async function loadFundSnapshot(code: string): Promise<FundSnapshot> {
  cleanupEastMoneyGlobals();
  await loadScript(
    `https://fund.eastmoney.com/pingzhongdata/${code}.js?v=${Date.now()}`
  );

  const globals = window as typeof window & EastMoneyGlobals;
  const navTrend = (globals.Data_netWorthTrend ?? [])
    .filter((item) => typeof item?.x === "number" && typeof item?.y === "number")
    .map((item) => ({
      timestamp: item.x,
      value: item.y
    }));

  const latestPoint = navTrend.at(-1) ?? null;
  const previousPoint = navTrend.at(-2) ?? null;
  const latestNav = latestPoint?.value ?? null;
  const dailyChangePct =
    latestPoint && previousPoint && previousPoint.value !== 0
      ? Number(
          (((latestPoint.value - previousPoint.value) / previousPoint.value) * 100).toFixed(2)
        )
      : null;

  const allocationCategories = globals.Data_assetAllocation?.categories ?? [];
  const allocationSeries = globals.Data_assetAllocation?.series ?? [];
  const stockRatio = allocationSeries.find((item) => item.name.includes("股票"));
  const bondRatio = allocationSeries.find((item) => item.name.includes("债券"));
  const cashRatio = allocationSeries.find((item) => item.name.includes("现金"));
  const netAsset = allocationSeries.find((item) => item.name.includes("净资产"));

  const assetAllocation = allocationCategories.map((date, index) => ({
    date,
    stockRatio: stockRatio?.data?.[index] ?? null,
    bondRatio: bondRatio?.data?.[index] ?? null,
    cashRatio: cashRatio?.data?.[index] ?? null,
    netAsset: netAsset?.data?.[index] ?? null
  }));

  const fluctuationScaleCategories = globals.Data_fluctuationScale?.categories ?? [];
  const fluctuationScaleSeries = globals.Data_fluctuationScale?.series ?? [];
  const fluctuationScale = fluctuationScaleCategories.map((date, index) => ({
    date,
    scale: fluctuationScaleSeries[index]?.y ?? 0,
    mom: fluctuationScaleSeries[index]?.mom ?? null
  }));

  const holderCategories = globals.Data_holderStructure?.categories ?? [];
  const institutionSeries = globals.Data_holderStructure?.series?.find((item) =>
    item.name.includes("机构")
  );
  const individualSeries = globals.Data_holderStructure?.series?.find((item) =>
    item.name.includes("个人")
  );
  const internalSeries = globals.Data_holderStructure?.series?.find((item) =>
    item.name.includes("内部")
  );
  const holderStructure = holderCategories.map((date, index) => ({
    date,
    institution: institutionSeries?.data?.[index] ?? null,
    individual: individualSeries?.data?.[index] ?? null,
    internal: internalSeries?.data?.[index] ?? null
  }));

  const manager = globals.Data_currentFundManager?.[0]
    ? {
        name: globals.Data_currentFundManager[0].name ?? "",
        workTime: globals.Data_currentFundManager[0].workTime ?? "",
        fundSize: globals.Data_currentFundManager[0].fundSize ?? ""
      }
    : null;

  const performanceLabels = globals.Data_performanceEvaluation?.categories ?? [];
  const performanceData = globals.Data_performanceEvaluation?.data ?? [];
  const performanceRadar = performanceLabels.map((label, index) => ({
    label,
    value: performanceData[index] ?? 0
  }));

  const snapshot: FundSnapshot = {
    code,
    name: globals.fS_name ?? `基金 ${code}`,
    latestNav,
    navDate: latestPoint ? new Date(latestPoint.timestamp).toISOString() : null,
    dailyChangePct,
    returns: {
      month: toNumber(globals.syl_1y),
      quarter: toNumber(globals.syl_3y),
      halfYear: toNumber(globals.syl_6y),
      year: toNumber(globals.syl_1n)
    },
    navTrend,
    stockCodes: (globals.stockCodesNew ?? []).map(normalizeStockCode),
    assetAllocation,
    fluctuationScale,
    holderStructure,
    manager,
    performanceRadar
  };

  cleanupEastMoneyGlobals();
  return snapshot;
}

export function loadStockNotices(stockCode: string, limit = 4) {
  return new Promise<NoticeItem[]>((resolve, reject) => {
    const callbackName = `__fund_notice_${stockCode}_${Date.now()}`;

    const timer = window.setTimeout(() => {
      cleanup();
      reject(new Error("公告数据请求超时"));
    }, 12000);

    function cleanup() {
      window.clearTimeout(timer);
      Reflect.deleteProperty(window, callbackName);
      script.remove();
    }

    (window as typeof window & Record<string, (payload: any) => void>)[callbackName] = (
      payload
    ) => {
      cleanup();
      const list = payload?.data?.list ?? [];
      resolve(
        list.slice(0, limit).map((item: any) => ({
          title: item.title_ch ?? item.title ?? "",
          stockCode: item.codes?.[0]?.stock_code ?? stockCode,
          shortName: item.codes?.[0]?.short_name ?? "",
          noticeDate: item.notice_date ?? "",
          columnName: item.columns?.map((column: any) => column.column_name).join(" / ") ?? "",
          detailUrl: buildNoticeDetailUrl(
            item.codes?.[0]?.stock_code ?? stockCode,
            item.art_code
          )
        }))
      );
    };

    const script = document.createElement("script");
    script.src =
      "https://np-anotice-stock.eastmoney.com/api/security/ann" +
      `?cb=${callbackName}` +
      "&sr=-1&page_size=" +
      limit +
      "&page_index=1&ann_type=A&client_source=web&stock_list=" +
      stockCode;
    script.async = true;
    script.onerror = () => {
      cleanup();
      reject(new Error("公告数据加载失败"));
    };
    document.body.appendChild(script);
  });
}
