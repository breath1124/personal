"use client";

import Link from "next/link";
import { type FormEvent, useEffect, useMemo, useState } from "react";
import {
  ORACLE_AI_SKILLS,
  type OracleAiSkillId,
  getOracleAiSkill
} from "@/lib/ai-skills";
import { DEFAULT_AI_SETTINGS, AI_SETTINGS_KEY, ORACLE_PROFILE_KEY } from "@/lib/config";
import { requestModelAnalysis } from "@/lib/openai";
import {
  type BirthPrecision,
  type OracleDraft,
  type OracleFocus,
  type OraclePhase,
  DEFAULT_ORACLE_DRAFT,
  buildOracleReading
} from "@/lib/oracle";
import { readStorage, usePersistentState } from "@/lib/storage";

const focusOptions: Array<{ value: OracleFocus; label: string }> = [
  { value: "career", label: "事业推进" },
  { value: "wealth", label: "财务与资源" },
  { value: "relationship", label: "关系与亲密" },
  { value: "transition", label: "换挡与迁移" },
  { value: "growth", label: "自我成长" }
];

const phaseOptions: Array<{ value: OraclePhase; label: string }> = [
  { value: "steady", label: "稳定经营期" },
  { value: "pressure", label: "高压拉扯期" },
  { value: "reset", label: "重整校准期" },
  { value: "expansion", label: "扩张打开期" }
];

const precisionOptions: Array<{ value: BirthPrecision; label: string }> = [
  { value: "exact", label: "出生时刻明确" },
  { value: "approx", label: "只知道大概时段" },
  { value: "unknown", label: "时刻不确定" }
];

const toneLabel = {
  positive: "偏强",
  calm: "平衡",
  watch: "需补位"
} as const;

export function OracleStudio() {
  const [draft, setDraft, hydrated] = usePersistentState<OracleDraft>(
    ORACLE_PROFILE_KEY,
    DEFAULT_ORACLE_DRAFT
  );
  const [activated, setActivated] = useState(false);
  const [error, setError] = useState("");
  const [aiError, setAiError] = useState("");
  const [aiLoading, setAiLoading] = useState(false);
  const [aiResponse, setAiResponse] = useState("");
  const [aiSkillId, setAiSkillId] = useState<OracleAiSkillId>(ORACLE_AI_SKILLS[0].id);
  const [aiQuestion, setAiQuestion] = useState<string>(ORACLE_AI_SKILLS[0].suggestedQuestion);

  useEffect(() => {
    if (!hydrated) return;
    if (draft.birthDate) {
      setActivated(true);
    }
  }, [draft.birthDate, hydrated]);

  const result = useMemo(() => {
    if (!activated || !draft.birthDate) return null;
    return buildOracleReading(draft);
  }, [activated, draft]);
  const selectedAiSkill = getOracleAiSkill(aiSkillId);

  function updateDraft<K extends keyof OracleDraft>(key: K, value: OracleDraft[K]) {
    setDraft((current) => ({
      ...current,
      [key]: value
    }));
  }

  function generateReading(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (!draft.birthDate) {
      setError("先输入出生日期，实验盘才能生成。");
      return;
    }

    setActivated(true);
    setError("");
  }

  function resetDraft() {
    setDraft(DEFAULT_ORACLE_DRAFT);
    setActivated(false);
    setError("");
    setAiError("");
    setAiResponse("");
    setAiQuestion(ORACLE_AI_SKILLS[0].suggestedQuestion);
    setAiSkillId(ORACLE_AI_SKILLS[0].id);
  }

  async function generateAiReadout() {
    if (!result) {
      setAiError("先生成实验盘，再让模型解读。");
      return;
    }

    const settings = readStorage(AI_SETTINGS_KEY, DEFAULT_AI_SETTINGS);
    if (!settings.apiKey) {
      setAiError("当前公开站点不会内置平台 API Key；如需直接可用的 AI 解读，需要接服务端代理。");
      return;
    }

    setAiLoading(true);
    setAiError("");

    try {
      const content = await requestModelAnalysis(settings, [
        {
          role: "system",
          content: selectedAiSkill.systemPrompt
        },
        {
          role: "user",
          content: JSON.stringify({
            skill: {
              id: selectedAiSkill.id,
              label: selectedAiSkill.label,
              description: selectedAiSkill.description
            },
            question: aiQuestion.trim() || draft.question.trim() || selectedAiSkill.suggestedQuestion,
            draft,
            result
          })
        }
      ]);
      setAiResponse(content);
    } catch (analysisError) {
      setAiError(analysisError instanceof Error ? analysisError.message : "AI 解读失败");
    } finally {
      setAiLoading(false);
    }
  }

  return (
    <div className="section-stack">
      <section className="grid oracle-layout">
        <article className="panel-card">
          <div className="section-heading section-heading--tight">
            <div>
              <p className="eyebrow">Oracle Input</p>
              <h2>生成你的实验盘</h2>
            </div>
            <span className="tone-pill">仅保存在本地</span>
          </div>

          <p className="muted">
            这不是照搬测测的情绪话术流，而是把八字实验盘、紫微斗数镜像和现实校验拆开，让结果能落回真实生活。
          </p>

          <form className="section-stack" onSubmit={generateReading}>
            <div className="form-grid">
              <label className="field">
                <span>出生日期</span>
                <input
                  className="control"
                  type="date"
                  value={draft.birthDate}
                  onChange={(event) => updateDraft("birthDate", event.target.value)}
                />
              </label>
              <label className="field">
                <span>出生时间</span>
                <input
                  className="control"
                  type="time"
                  value={draft.birthTime}
                  onChange={(event) => updateDraft("birthTime", event.target.value)}
                />
              </label>
              <label className="field">
                <span>时间精度</span>
                <select
                  className="control"
                  value={draft.precision}
                  onChange={(event) => updateDraft("precision", event.target.value as BirthPrecision)}
                >
                  {precisionOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>当前最想看什么</span>
                <select
                  className="control"
                  value={draft.focus}
                  onChange={(event) => updateDraft("focus", event.target.value as OracleFocus)}
                >
                  {focusOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field field--full">
                <span>你当前处在什么阶段</span>
                <select
                  className="control"
                  value={draft.phase}
                  onChange={(event) => updateDraft("phase", event.target.value as OraclePhase)}
                >
                  {phaseOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <label className="field">
              <span>这次最想被帮助看清的问题</span>
              <textarea
                className="control control--textarea"
                placeholder="比如：最近适不适合换工作，或者这段关系里到底该怎么做。"
                value={draft.question}
                onChange={(event) => updateDraft("question", event.target.value)}
              />
            </label>

            <div className="button-row">
              <button className="button" type="submit">
                生成实验盘
              </button>
              <button className="button button--secondary" onClick={resetDraft} type="button">
                清空
              </button>
              {error && <span className="callout callout--warn">{error}</span>}
            </div>
          </form>
        </article>

        <article className="panel-card oracle-summary-card">
          <div className="section-heading section-heading--tight">
            <div>
              <p className="eyebrow">Oracle Snapshot</p>
              <h2>{result ? result.chartTitle : "等待输入"}</h2>
            </div>
            {result && (
              <span className={`tone-pill tone-pill--${result.elementBalance[0]?.tone ?? "calm"}`}>
                {result.focusLabel}
              </span>
            )}
          </div>

          {result ? (
            <div className="section-stack">
              <p className="oracle-summary">{result.focusSummary}</p>

              <div className="app-card__chips">
                <span className="chip">日主 {result.dayMaster.stem}</span>
                <span className="chip">主导 {result.dominantElement}</span>
                <span className="chip">命宫 {result.palace.name}</span>
                <span className="chip">{result.confidenceLabel}</span>
              </div>

              <article className="subtle-card">
                <h3>一句话读盘</h3>
                <p>
                  {result.dayMaster.stem}{result.dayMaster.element}日主，整体更像
                  {result.rhythmLabel}。当前流年是 {result.yearlyTheme.pillar}，
                  主题偏向“{result.yearlyTheme.title}”。
                </p>
              </article>

              <article className="subtle-card">
                <h3>为什么这版和测测不一样</h3>
                <ul className="bullet-list">
                  <li>先给盘面，再给现实转译，不用一堆玄学语言把人包住。</li>
                  <li>每次解读都配现实校验和行动实验，避免只获得“被说中”的快感。</li>
                  <li>把时辰精度直接暴露出来，不装成毫无误差的绝对结论。</li>
                </ul>
              </article>
            </div>
          ) : (
            <div className="oracle-placeholder">
              <p className="oracle-summary">
                输入出生信息后，这里会先给你一张压缩版结论：日主、主导五行、命宫落点和流年主题。
              </p>
              <ul className="bullet-list">
                <li>八字实验盘：四柱、五行结构和主导节奏。</li>
                <li>紫微斗数镜像：命宫、身宫、叙事星和六项现实领域评分。</li>
                <li>现实转译：校验问题、行动实验和使用边界。</li>
              </ul>
            </div>
          )}
        </article>
      </section>

      {result && (
        <>
          <section className="stats-grid">
            <article className="panel-card stat-card">
              <span>日主</span>
              <strong>
                {result.dayMaster.stem} {result.dayMaster.element}
              </strong>
              <p>{result.dayMaster.title}</p>
            </article>
            <article className="panel-card stat-card">
              <span>主导五行</span>
              <strong>{result.dominantElement}</strong>
              <p>辅助能量 {result.supportingElement}</p>
            </article>
            <article className="panel-card stat-card">
              <span>命宫 / 身宫</span>
              <strong>{result.palace.name}</strong>
              <p>{result.bodyPalace.name}</p>
            </article>
            <article className="panel-card stat-card">
              <span>参考度</span>
              <strong>{result.confidenceLabel}</strong>
              <p>{result.phaseLabel}</p>
            </article>
          </section>

          <section className="panel-card">
            <div className="section-heading section-heading--tight">
              <div>
                <p className="eyebrow">AI Oracle</p>
                <h2>模型命理解读</h2>
              </div>
              <Link className="inline-link inline-link--subtle" href="/settings">
                设置模型
              </Link>
            </div>

            <div className="form-grid">
              <label className="field">
                <span>算命技能</span>
                <select
                  className="control"
                  value={aiSkillId}
                  onChange={(event) => setAiSkillId(event.target.value as OracleAiSkillId)}
                >
                  {ORACLE_AI_SKILLS.map((skill) => (
                    <option key={skill.id} value={skill.id}>
                      {skill.label}
                    </option>
                  ))}
                </select>
              </label>
              <article className="subtle-card">
                <h3>{selectedAiSkill.label}</h3>
                <p>{selectedAiSkill.description}</p>
              </article>
            </div>

            <label className="field field--full">
              <span>你希望模型重点解读什么？</span>
              <textarea
                className="control control--textarea"
                value={aiQuestion}
                onChange={(event) => setAiQuestion(event.target.value)}
              />
            </label>
            <p className="muted">默认提问：{selectedAiSkill.suggestedQuestion}</p>

            <div className="button-row">
              <button className="button" disabled={aiLoading} onClick={generateAiReadout} type="button">
                {aiLoading ? "生成中..." : "生成 AI 命理解读"}
              </button>
              {aiError && <span className="callout callout--warn">{aiError}</span>}
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
              <p className="muted">
                模型会结合你的实验盘、本地规则结果和当前问题，按所选技能给出更细的命理解读。
              </p>
            )}
          </section>

          <section className="grid two-up">
            <article className="panel-card">
              <div className="section-heading section-heading--tight">
                <div>
                  <p className="eyebrow">Bazi Frame</p>
                  <h2>八字实验盘</h2>
                </div>
              </div>

              <div className="pillar-grid">
                {result.pillars.map((pillar) => (
                  <article className="pillar-card" key={pillar.label}>
                    <span>{pillar.label}</span>
                    <strong>
                      {pillar.stem}
                      {pillar.branch}
                    </strong>
                    <p>
                      {pillar.stemElement} / {pillar.branchElement}
                    </p>
                  </article>
                ))}
              </div>

              <article className="subtle-card">
                <h3>日主底色</h3>
                <p>{result.dayMaster.brief}</p>
              </article>
            </article>

            <article className="panel-card">
              <div className="section-heading section-heading--tight">
                <div>
                  <p className="eyebrow">Element Balance</p>
                  <h2>五行结构</h2>
                </div>
                <span className="muted">{result.rhythmLabel}</span>
              </div>

              <div className="section-stack">
                {result.elementBalance.map((item) => (
                  <article className="element-row" key={item.element}>
                    <div className="element-row__head">
                      <strong>{item.element}</strong>
                      <span className={`tone-pill tone-pill--${item.tone}`}>
                        {toneLabel[item.tone]}
                      </span>
                    </div>
                    <div className="element-track" aria-hidden="true">
                      <span
                        className={`element-fill element-fill--${item.tone}`}
                        style={{ width: `${item.ratio * 100}%` }}
                      />
                    </div>
                    <p>{item.summary}</p>
                  </article>
                ))}
              </div>

              <article className="subtle-card">
                <h3>驱动力分布</h3>
                <div className="app-card__chips">
                  {result.relationMix.map((item) => (
                    <span className="chip" key={item.relation}>
                      {item.label} {item.count}
                    </span>
                  ))}
                </div>
              </article>
            </article>
          </section>

          <section className="grid two-up">
            <article className="panel-card">
              <div className="section-heading section-heading--tight">
                <div>
                  <p className="eyebrow">Ziwei Mirror</p>
                  <h2>紫微斗数镜像</h2>
                </div>
              </div>

              <div className="oracle-meta-grid">
                <article className="subtle-card">
                  <h3>{result.palace.name}</h3>
                  <p>{result.palace.note}</p>
                </article>
                <article className="subtle-card">
                  <h3>{result.bodyPalace.name}</h3>
                  <p>{result.bodyPalace.note}</p>
                </article>
              </div>

              <div className="oracle-meta-grid">
                {result.stars.map((star) => (
                  <article className="subtle-card" key={star.name}>
                    <div className="stat-row">
                      <h3>{star.name}</h3>
                      <span className="tone-pill">{star.tag}</span>
                    </div>
                    <p>{star.note}</p>
                  </article>
                ))}
              </div>
            </article>

            <article className="panel-card">
              <div className="section-heading section-heading--tight">
                <div>
                  <p className="eyebrow">Reality Translation</p>
                  <h2>六项现实评分</h2>
                </div>
                <span className="muted">{result.yearlyTheme.relationLabel}</span>
              </div>

              <div className="oracle-domain-list">
                {result.domainScores.map((item) => (
                  <article className="oracle-domain-card" key={item.name}>
                    <div className="stat-row">
                      <h3>{item.name}</h3>
                      <strong>{item.score}</strong>
                    </div>
                    <span className={`tone-pill tone-pill--${item.tone}`}>{toneLabel[item.tone]}</span>
                    <p>{item.note}</p>
                  </article>
                ))}
              </div>
            </article>
          </section>

          <section className="grid two-up">
            {result.sections.map((section) => (
              <article className="panel-card" key={section.title}>
                <h2>{section.title}</h2>
                <ul className="bullet-list">
                  {section.items.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </article>
            ))}
          </section>

          <section className="grid two-up">
            <article className="panel-card">
              <div className="section-heading section-heading--tight">
                <div>
                  <p className="eyebrow">Reality Check</p>
                  <h2>反向校验</h2>
                </div>
              </div>
              <ul className="bullet-list">
                {result.realityChecks.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </article>

            <article className="panel-card">
              <div className="section-heading section-heading--tight">
                <div>
                  <p className="eyebrow">Action Experiments</p>
                  <h2>行动实验</h2>
                </div>
              </div>
              <ul className="bullet-list">
                {result.actionExperiments.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </article>
          </section>

          <section className="panel-card">
            <div className="section-heading section-heading--tight">
              <div>
                <p className="eyebrow">Boundary</p>
                <h2>使用边界</h2>
              </div>
            </div>
            <p className="muted">{result.confidenceNote}</p>
            <ul className="bullet-list">
              {result.calibrationQuestions.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </section>
        </>
      )}
    </div>
  );
}
