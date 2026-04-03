export type AiSettings = {
  providerName: string;
  baseUrl: string;
  model: string;
  apiKey: string;
  temperature: number;
};

export type HoldingRecord = {
  id: string;
  fundCode: string;
  units: number;
  averageCost: number;
  thesis: string;
  targetWeight: number;
  note: string;
  addedAt: string;
};

export type ReturnMetrics = {
  month: number | null;
  quarter: number | null;
  halfYear: number | null;
  year: number | null;
};

export type FundManager = {
  name: string;
  workTime: string;
  fundSize: string;
};

export type AssetAllocationPoint = {
  date: string;
  stockRatio: number | null;
  bondRatio: number | null;
  cashRatio: number | null;
  netAsset: number | null;
};

export type FundSnapshot = {
  code: string;
  name: string;
  latestNav: number | null;
  navDate: string | null;
  dailyChangePct: number | null;
  returns: ReturnMetrics;
  navTrend: Array<{ timestamp: number; value: number }>;
  stockCodes: string[];
  assetAllocation: AssetAllocationPoint[];
  fluctuationScale: Array<{ date: string; scale: number; mom: string | null }>;
  holderStructure: Array<{
    date: string;
    institution: number | null;
    individual: number | null;
    internal: number | null;
  }>;
  manager: FundManager | null;
  performanceRadar: Array<{ label: string; value: number }>;
};

export type NoticeItem = {
  title: string;
  stockCode: string;
  shortName: string;
  noticeDate: string;
  columnName: string;
};

export type MarketFundDigest = {
  code: string;
  name: string;
  latestNav: number | null;
  navDate: string | null;
  monthReturn: number | null;
  quarterReturn: number | null;
  yearReturn: number | null;
  manager: string | null;
  scale: string | null;
};

export type MarketBrief = {
  generatedAt: string;
  featuredFunds: MarketFundDigest[];
  noticeDigest: NoticeItem[];
  marketPulse: Array<{
    title: string;
    detail: string;
    tone: "calm" | "watch" | "positive";
  }>;
};

export type AnalysisSection = {
  title: string;
  items: string[];
};
