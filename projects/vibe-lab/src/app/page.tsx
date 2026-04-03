import Link from "next/link";
import marketBrief from "@/generated/market-brief.json";
import { AppShell } from "@/components/app-shell";
import type { MarketBrief } from "@/lib/types";

const brief = marketBrief as MarketBrief;
const toneLabel = {
  positive: "偏强",
  watch: "留意",
  calm: "平稳"
} as const;

export default function HomePage() {
  return (
    <AppShell
      activePath="/"
      title="把实验项目做成真正可用的界面"
      description="基金助手提供持仓、数据和 AI 研判；MBTI 助手提供完整测评和更有用的结果报告。"
    >
      <section className="grid two-up">
        <article className="hero-card">
          <p className="eyebrow">Fund Copilot</p>
          <h2>基金助手</h2>
          <p>录入持仓、同步基金画像、查看公告信号，再决定下一步该继续拿、分批加还是先降仓。</p>
          <Link className="inline-link" href="/fund">
            打开助手
          </Link>
        </article>
        <article className="hero-card">
          <p className="eyebrow">MBTI Report</p>
          <h2>MBTI 助手</h2>
          <p>不是随手玩玩的 12 题测试，而是更完整的自评和工作沟通报告。</p>
          <Link className="inline-link" href="/mbti">
            开始测评
          </Link>
        </article>
      </section>

      <section className="section-stack">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Daily Brief</p>
            <h2>今日市场简报</h2>
          </div>
          <p>{new Date(brief.generatedAt).toLocaleString("zh-CN")}</p>
        </div>
        <div className="grid three-up">
          {brief.marketPulse.map((item) => (
            <article className="panel-card" key={item.title}>
              <span className={`tone-pill tone-pill--${item.tone}`}>{toneLabel[item.tone]}</span>
              <h3>{item.title}</h3>
              <p>{item.detail}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="section-stack">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Funds</p>
            <h2>关注中的基金</h2>
          </div>
        </div>
        <div className="grid three-up">
          {brief.featuredFunds.map((item) => (
            <article className="panel-card" key={item.code}>
              <div className="stat-row">
                <span>{item.code}</span>
                <strong>{item.latestNav ?? "-"}</strong>
              </div>
              <h3>{item.name}</h3>
              <p>
                近一月 {item.monthReturn ?? "-"}% · 近一季 {item.quarterReturn ?? "-"}% · 近一年{" "}
                {item.yearReturn ?? "-"}%
              </p>
              <p className="muted">
                {item.manager ?? "暂缺经理信息"} · {item.scale ?? "暂缺规模信息"}
              </p>
            </article>
          ))}
        </div>
      </section>

      <section className="section-stack">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Notices</p>
            <h2>最新公告</h2>
          </div>
        </div>
        <div className="notice-list">
          {brief.noticeDigest.map((item) => (
            <article className="notice-item" key={`${item.stockCode}-${item.title}`}>
              <div>
                <p className="notice-meta">
                  {item.shortName} · {item.stockCode} · {item.noticeDate.slice(0, 10)}
                </p>
                <h3>
                  {item.detailUrl ? (
                    <a
                      className="notice-link"
                      href={item.detailUrl}
                      rel="noreferrer"
                      target="_blank"
                    >
                      {item.title}
                    </a>
                  ) : (
                    item.title
                  )}
                </h3>
              </div>
              <span className="notice-tag">{item.columnName || "公告"}</span>
            </article>
          ))}
        </div>
      </section>
    </AppShell>
  );
}
