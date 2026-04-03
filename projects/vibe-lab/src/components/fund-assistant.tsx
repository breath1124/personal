"use client";

import { startTransition, useEffect, useMemo, useState } from "react";
import { DEFAULT_AI_SETTINGS, DEFAULT_HOLDING_DRAFT, AI_SETTINGS_KEY, HOLDINGS_KEY } from "@/lib/config";
import { buildHoldingSections, buildPortfolioBrief } from "@/lib/finance/rules";
import { loadFundSnapshot, loadStockNotices } from "@/lib/finance/browser";
import { formatCurrency, formatDate, formatDateTime, formatNumber, formatPct } from "@/lib/format";
import { requestModelAnalysis } from "@/lib/openai";
import { readStorage, usePersistentState } from "@/lib/storage";
import type {
  FundSnapshot,
  HoldingRecord,
  MarketBrief,
  NoticeItem
} from "@/lib/types";
import { Sparkline } from "@/components/sparkline";

type DraftState = typeof DEFAULT_HOLDING_DRAFT;

export function FundAssistant({ marketBrief }: { marketBrief: MarketBrief }) {
  const toneLabel = {
    positive: "偏强",
    watch: "留意",
    calm: "平稳"
  } as const;
  const [holdings, setHoldings, hydrated] = usePersistentState<HoldingRecord[]>(HOLDINGS_KEY, []);
  const [draft, setDraft] = useState<DraftState>(DEFAULT_HOLDING_DRAFT);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [snapshots, setSnapshots] = useState<Record<string, FundSnapshot>>({});
  const [notices, setNotices] = useState<Record<string, NoticeItem[]>>({});
  const [loadingCodes, setLoadingCodes] = useState<string[]>([]);
  const [error, setError] = useState<string>("");
  const [aiQuestion, setAiQuestion] = useState(
    "这组持仓现在更适合继续拿、逐步加仓还是先控制仓位？请给我一个行动框架。"
  );
  const [aiLoading, setAiLoading] = useState(false);
  const [aiResponse, setAiResponse] = useState("");

  useEffect(() => {
    if (!hydrated) return;
    if (holdings.length === 0) return;
    if (!selectedId) {
      setSelectedId(holdings[0].id);
    }
  }, [holdings, hydrated, selectedId]);

  async function refreshHolding(holding: HoldingRecord) {
    setLoadingCodes((current) => [...new Set([...current, holding.fundCode])]);
    setError("");

    try {
      const snapshot = await loadFundSnapshot(holding.fundCode);
      setSnapshots((current) => ({
        ...current,
        [holding.fundCode]: snapshot
      }));

      const topStockCodes = snapshot.stockCodes.slice(0, 2);
      const noticeGroups = await Promise.all(
        topStockCodes.map((stockCode) => loadStockNotices(stockCode, 3).catch(() => []))
      );
      const mergedNotices = noticeGroups
        .flat()
        .sort((a, b) => b.noticeDate.localeCompare(a.noticeDate));

      setNotices((current) => ({
        ...current,
        [holding.fundCode]: mergedNotices
      }));
    } catch (refreshError) {
      setError(
        refreshError instanceof Error
          ? refreshError.message
          : `基金 ${holding.fundCode} 数据加载失败`
      );
    } finally {
      setLoadingCodes((current) => current.filter((item) => item !== holding.fundCode));
    }
  }

  useEffect(() => {
    if (!hydrated) return;
    if (holdings.length === 0) return;

    startTransition(() => {
      void (async () => {
        for (const holding of holdings) {
          if (!snapshots[holding.fundCode]) {
            await refreshHolding(holding);
          }
        }
      })();
    });
  }, [holdings, hydrated, snapshots]);

  const selectedHolding = holdings.find((item) => item.id === selectedId) ?? null;
  const selectedSnapshot = selectedHolding ? snapshots[selectedHolding.fundCode] ?? null : null;
  const selectedNotices = selectedHolding ? notices[selectedHolding.fundCode] ?? [] : [];

  const portfolioBrief = useMemo(
    () => buildPortfolioBrief(holdings, Object.values(snapshots), notices),
    [holdings, notices, snapshots]
  );

  const selectedSections =
    selectedHolding && selectedSnapshot
      ? buildHoldingSections(selectedHolding, selectedSnapshot, selectedNotices)
      : [];

  function updateDraft<K extends keyof DraftState>(key: K, value: DraftState[K]) {
    setDraft((current) => ({
      ...current,
      [key]: value
    }));
  }

  function addHolding() {
    const normalizedCode = draft.fundCode.trim();
    if (!/^\d{6}$/.test(normalizedCode)) {
      setError("基金代码需要是 6 位数字。");
      return;
    }

    const nextHolding: HoldingRecord = {
      id: `${normalizedCode}-${Date.now()}`,
      fundCode: normalizedCode,
      units: draft.units,
      averageCost: draft.averageCost,
      thesis: draft.thesis.trim(),
      targetWeight: draft.targetWeight,
      note: draft.note.trim(),
      addedAt: new Date().toISOString()
    };

    setHoldings((current) => [nextHolding, ...current]);
    setSelectedId(nextHolding.id);
    setDraft(DEFAULT_HOLDING_DRAFT);
    setError("");
    startTransition(() => {
      void refreshHolding(nextHolding);
    });
  }

  function removeHolding(holdingId: string) {
    setHoldings((current) => current.filter((item) => item.id !== holdingId));
    if (selectedId === holdingId) {
      setSelectedId(null);
    }
  }

  async function generateAiReport() {
    if (holdings.length === 0) {
      setError("先录入至少一只持仓，再生成 AI 研判。");
      return;
    }

    const settings = readStorage(AI_SETTINGS_KEY, DEFAULT_AI_SETTINGS);
    if (!settings.apiKey) {
      setError("先去设置页填写 API Key，再生成 AI 研判。");
      return;
    }

    setAiLoading(true);
    setError("");

    try {
      const payload = {
        portfolioBrief,
        holdings: holdings.map((holding) => ({
          ...holding,
          snapshot: snapshots[holding.fundCode] ?? null,
          notices: notices[holding.fundCode] ?? []
        })),
        marketBrief,
        question: aiQuestion
      };

      const content = await requestModelAnalysis(settings, [
        {
          role: "system",
          content:
            "你是一位克制、专业、注重风险边界的基金投研助手。请用简体中文输出 Markdown，严格使用这四个标题：## 一句话判断、## 核心依据、## 风险提醒、## 接下来怎么做。不要给绝对收益承诺，不要假装知道实时行情。"
        },
        {
          role: "user",
          content: JSON.stringify(payload)
        }
      ]);

      setAiResponse(content);
    } catch (analysisError) {
      setError(
        analysisError instanceof Error ? analysisError.message : "AI 研判生成失败"
      );
    } finally {
      setAiLoading(false);
    }
  }

  return (
    <div className="section-stack">
      <section className="stats-grid">
        <article className="panel-card stat-card">
          <span>总成本</span>
          <strong>{formatCurrency(portfolioBrief.totalCost)}</strong>
          <p>按录入份额和成本计算。</p>
        </article>
        <article className="panel-card stat-card">
          <span>估算市值</span>
          <strong>{formatCurrency(portfolioBrief.totalValue)}</strong>
          <p>按最近抓到的单位净值估算。</p>
        </article>
        <article className="panel-card stat-card">
          <span>浮动收益</span>
          <strong className={portfolioBrief.totalPnL >= 0 ? "rise" : "fall"}>
            {formatPct(portfolioBrief.totalPnL)}
          </strong>
          <p>这不是交易建议，只是持仓状态快照。</p>
        </article>
        <article className="panel-card stat-card">
          <span>持仓数量</span>
          <strong>{holdings.length}</strong>
          <p>录入后会同步基金画像和重仓股公告。</p>
        </article>
      </section>

      <section className="grid fund-layout">
        <article className="panel-card">
          <div className="section-heading section-heading--tight">
            <div>
              <p className="eyebrow">Portfolio</p>
              <h2>录入持仓</h2>
            </div>
            <button className="button button--secondary" onClick={() => holdings.forEach((item) => startTransition(() => void refreshHolding(item)))}>
              刷新全部
            </button>
          </div>

          <div className="form-grid">
            <label className="field">
              <span>基金代码</span>
              <input
                className="control"
                inputMode="numeric"
                placeholder="161725"
                value={draft.fundCode}
                onChange={(event) => updateDraft("fundCode", event.target.value)}
              />
            </label>
            <label className="field">
              <span>持有份额</span>
              <input
                className="control"
                min="0"
                step="100"
                type="number"
                value={draft.units}
                onChange={(event) => updateDraft("units", Number(event.target.value))}
              />
            </label>
            <label className="field">
              <span>平均成本</span>
              <input
                className="control"
                min="0"
                step="0.0001"
                type="number"
                value={draft.averageCost}
                onChange={(event) => updateDraft("averageCost", Number(event.target.value))}
              />
            </label>
            <label className="field">
              <span>目标仓位 (%)</span>
              <input
                className="control"
                max="100"
                min="0"
                step="1"
                type="number"
                value={draft.targetWeight}
                onChange={(event) => updateDraft("targetWeight", Number(event.target.value))}
              />
            </label>
          </div>

          <label className="field field--full">
            <span>最初买入逻辑</span>
            <textarea
              className="control control--textarea"
              placeholder="比如：作为消费风格仓位，计划拿 12-18 个月，接受中等波动。"
              value={draft.thesis}
              onChange={(event) => updateDraft("thesis", event.target.value)}
            />
          </label>

          <label className="field field--full">
            <span>备注</span>
            <textarea
              className="control control--textarea"
              placeholder="比如：下一次只在回撤到某个区间时考虑补仓。"
              value={draft.note}
              onChange={(event) => updateDraft("note", event.target.value)}
            />
          </label>

          <div className="button-row">
            <button className="button" onClick={addHolding}>
              添加持仓
            </button>
            <span className="muted">支持直接输入 6 位基金代码。</span>
          </div>

          {error && <p className="callout callout--warn">{error}</p>}

          <div className="holdings-list">
            {holdings.map((holding) => {
              const snapshot = snapshots[holding.fundCode];
              const isLoading = loadingCodes.includes(holding.fundCode);
              const marketValue = snapshot?.latestNav ? snapshot.latestNav * holding.units : null;
              const costValue = holding.averageCost * holding.units;
              const pnl =
                marketValue && costValue
                  ? Number((((marketValue - costValue) / costValue) * 100).toFixed(2))
                  : null;

              return (
                <button
                  className={`holding-row${selectedId === holding.id ? " is-active" : ""}`}
                  key={holding.id}
                  onClick={() => setSelectedId(holding.id)}
                  type="button"
                >
                  <div>
                    <div className="holding-title">
                      <strong>{snapshot?.name ?? holding.fundCode}</strong>
                      <span>{holding.fundCode}</span>
                    </div>
                    <p className="muted">
                      持有 {holding.units} 份 · 成本 {holding.averageCost}
                    </p>
                  </div>
                  <div className="holding-actions">
                    <span className={pnl !== null && pnl >= 0 ? "rise" : "fall"}>
                      {pnl === null ? "同步中" : formatPct(pnl)}
                    </span>
                    <div className="micro-actions">
                      <span className="status-note">{isLoading ? "更新中" : formatDate(snapshot?.navDate ?? null)}</span>
                      <button
                        className="micro-button"
                        onClick={(event) => {
                          event.stopPropagation();
                          startTransition(() => {
                            void refreshHolding(holding);
                          });
                        }}
                        type="button"
                      >
                        刷新
                      </button>
                      <button
                        className="micro-button micro-button--danger"
                        onClick={(event) => {
                          event.stopPropagation();
                          removeHolding(holding.id);
                        }}
                        type="button"
                      >
                        删除
                      </button>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>
        </article>

        <div className="section-stack">
          <article className="panel-card">
            <div className="section-heading section-heading--tight">
              <div>
                <p className="eyebrow">Insight</p>
                <h2>{selectedSnapshot?.name ?? "选择一个持仓"}</h2>
              </div>
              {selectedSnapshot && <span className="muted">更新于 {formatDate(selectedSnapshot.navDate)}</span>}
            </div>

            {selectedSnapshot && selectedHolding ? (
              <div className="section-stack">
                <div className="stats-grid stats-grid--mini">
                  <article className="mini-stat">
                    <span>单位净值</span>
                    <strong>{formatNumber(selectedSnapshot.latestNav, 4)}</strong>
                    <p>日变动 {formatPct(selectedSnapshot.dailyChangePct)}</p>
                  </article>
                  <article className="mini-stat">
                    <span>近 1 月</span>
                    <strong>{formatPct(selectedSnapshot.returns.month)}</strong>
                    <p>近 3 月 {formatPct(selectedSnapshot.returns.quarter)}</p>
                  </article>
                  <article className="mini-stat">
                    <span>近 1 年</span>
                    <strong>{formatPct(selectedSnapshot.returns.year)}</strong>
                    <p>目标仓位 {selectedHolding.targetWeight}%</p>
                  </article>
                </div>

                <Sparkline values={selectedSnapshot.navTrend.slice(-90).map((item) => item.value)} />

                <div className="grid two-up">
                  {selectedSections.map((section) => (
                    <article className="subtle-card" key={section.title}>
                      <h3>{section.title}</h3>
                      <ul className="bullet-list">
                        {section.items.map((item) => (
                          <li key={item}>{item}</li>
                        ))}
                      </ul>
                    </article>
                  ))}
                </div>

                <article className="subtle-card">
                  <h3>最近公告与财报信号</h3>
                  <div className="notice-list notice-list--compact">
                    {selectedNotices.length > 0 ? (
                      selectedNotices.map((item) => (
                        <article className="notice-item notice-item--compact" key={`${item.stockCode}-${item.title}`}>
                          <div>
                            <p className="notice-meta">
                              {item.shortName} · {item.stockCode} · {item.noticeDate.slice(0, 10)}
                            </p>
                            <h3>{item.title}</h3>
                          </div>
                          <span className="notice-tag">{item.columnName || "公告"}</span>
                        </article>
                      ))
                    ) : (
                      <p className="muted">当前没有抓到这只基金重仓股的公告。</p>
                    )}
                  </div>
                </article>
              </div>
            ) : (
              <p className="muted">先录入并选择一只基金，右侧会展示完整画像、公告和建议。</p>
            )}
          </article>

          <article className="panel-card">
            <div className="section-heading section-heading--tight">
              <div>
                <p className="eyebrow">AI Copilot</p>
                <h2>模型研判</h2>
              </div>
              <span className="muted">读取本地 API 配置</span>
            </div>

            <label className="field field--full">
              <span>你想让模型重点回答什么？</span>
              <textarea
                className="control control--textarea"
                value={aiQuestion}
                onChange={(event) => setAiQuestion(event.target.value)}
              />
            </label>

            <div className="button-row">
              <button className="button" disabled={aiLoading} onClick={generateAiReport}>
                {aiLoading ? "生成中..." : "生成 AI 研判"}
              </button>
              <a className="inline-link inline-link--subtle" href="/lab/settings/">
                去设置模型
              </a>
            </div>

            {aiResponse ? (
              <article className="markdown-card">
                {aiResponse.split("\n").map((line, index) =>
                  line.startsWith("## ") ? (
                    <h3 key={`${line}-${index}`}>{line.replace(/^##\s*/, "")}</h3>
                  ) : line.trim() ? (
                    <p key={`${line}-${index}`}>{line}</p>
                  ) : null
                )}
              </article>
            ) : (
              <p className="muted">模型会结合你的持仓、基金数据、公告摘要和今日市场简报输出更深的判断。</p>
            )}
          </article>

          <article className="panel-card">
            <div className="section-heading section-heading--tight">
              <div>
                <p className="eyebrow">Daily Brief</p>
                <h2>今日市场简报</h2>
              </div>
              <span className="muted">{formatDateTime(marketBrief.generatedAt)}</span>
            </div>
            <div className="grid">
              {marketBrief.marketPulse.map((item) => (
                <article className="subtle-card" key={item.title}>
                  <div className="stat-row">
                    <strong>{item.title}</strong>
                    <span className={`tone-pill tone-pill--${item.tone}`}>{toneLabel[item.tone]}</span>
                  </div>
                  <p>{item.detail}</p>
                </article>
              ))}
            </div>
          </article>
        </div>
      </section>
    </div>
  );
}
