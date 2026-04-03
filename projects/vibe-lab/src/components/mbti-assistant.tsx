"use client";

import Link from "next/link";
import { useState } from "react";
import { DEFAULT_AI_SETTINGS, AI_SETTINGS_KEY } from "@/lib/config";
import { MBTI_QUESTIONS, buildMbtiReport, scoreMbti } from "@/lib/mbti";
import { requestModelAnalysis } from "@/lib/openai";
import { readStorage } from "@/lib/storage";

type FocusMode = "work" | "team" | "communication" | "growth";

export function MbtiAssistant() {
  const [answers, setAnswers] = useState<Record<string, number>>({});
  const [focus, setFocus] = useState<FocusMode>("work");
  const [context, setContext] = useState("");
  const [result, setResult] = useState<ReturnType<typeof scoreMbti> | null>(null);
  const [error, setError] = useState("");
  const [aiLoading, setAiLoading] = useState(false);
  const [aiResponse, setAiResponse] = useState("");

  const completedCount = Object.keys(answers).length;

  function updateAnswer(questionId: string, score: number) {
    setAnswers((current) => ({
      ...current,
      [questionId]: score
    }));
  }

  function submitAssessment() {
    if (completedCount < MBTI_QUESTIONS.length) {
      setError(`还有 ${MBTI_QUESTIONS.length - completedCount} 道题未完成。`);
      return;
    }

    const scored = scoreMbti(answers);
    setResult(scored);
    setError("");
  }

  async function generateAiReadout() {
    if (!result) {
      setError("先完成测评，再生成 AI 解读。");
      return;
    }

    const settings = readStorage(AI_SETTINGS_KEY, DEFAULT_AI_SETTINGS);
    if (!settings.apiKey) {
      setError("当前公开站点不会内置平台 API Key；如需直接可用的 AI 解读，需要接服务端代理。");
      return;
    }

    setAiLoading(true);
    setError("");

    try {
      const content = await requestModelAnalysis(settings, [
        {
          role: "system",
          content:
            "你是一位冷静、清楚、不过度神化人格类型的职业沟通教练。请用简体中文输出 Markdown，使用这四个标题：## 类型解读、## 当前场景的优势、## 可能的盲区、## 实际建议。"
        },
        {
          role: "user",
          content: JSON.stringify({
            focus,
            context,
            result
          })
        }
      ]);
      setAiResponse(content);
    } catch (analysisError) {
      setError(analysisError instanceof Error ? analysisError.message : "AI 解读失败");
    } finally {
      setAiLoading(false);
    }
  }

  const sections = result ? buildMbtiReport(result, focus) : [];

  return (
    <div className="section-stack">
      <section className="panel-card">
        <div className="section-heading section-heading--tight">
          <div>
            <p className="eyebrow">Assessment</p>
            <h2>完整测评</h2>
          </div>
          <span className="muted">
            已完成 {completedCount}/{MBTI_QUESTIONS.length}
          </span>
        </div>

        <div className="form-grid">
          <label className="field">
            <span>这次更关注什么场景？</span>
            <select
              className="control"
              value={focus}
              onChange={(event) => setFocus(event.target.value as FocusMode)}
            >
              <option value="work">工作方式</option>
              <option value="team">团队协作</option>
              <option value="communication">沟通表达</option>
              <option value="growth">成长策略</option>
            </select>
          </label>
          <label className="field">
            <span>当前你最想解决的问题</span>
            <input
              className="control"
              placeholder="比如：带团队时怎么沟通更有效。"
              value={context}
              onChange={(event) => setContext(event.target.value)}
            />
          </label>
        </div>

        <div className="question-stack">
          {MBTI_QUESTIONS.map((question, index) => (
            <article className="question-card" key={question.id}>
              <div className="question-head">
                <span className="question-index">Q{index + 1}</span>
                <div>
                  <h3>{question.prompt}</h3>
                  <p>
                    左侧：{question.left}
                    <br />
                    右侧：{question.right}
                  </p>
                </div>
              </div>
              <div className="answer-row">
                {[
                  { label: "明显左侧", score: -2 },
                  { label: "略偏左侧", score: -1 },
                  { label: "差不多", score: 0 },
                  { label: "略偏右侧", score: 1 },
                  { label: "明显右侧", score: 2 }
                ].map((option) => (
                  <button
                    className={`answer-pill${answers[question.id] === option.score ? " is-active" : ""}`}
                    key={option.label}
                    onClick={() => updateAnswer(question.id, option.score)}
                    type="button"
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </article>
          ))}
        </div>

        <div className="button-row">
          <button className="button" onClick={submitAssessment} type="button">
            生成结果报告
          </button>
          {error && <span className="callout callout--warn">{error}</span>}
        </div>
      </section>

      {result && (
        <>
          <section className="stats-grid">
            <article className="panel-card stat-card">
              <span>结果类型</span>
              <strong>{result.type}</strong>
              <p>{result.label}</p>
            </article>
            {result.dimensions.map((dimension) => (
              <article className="panel-card stat-card" key={dimension.axis}>
                <span>{dimension.label}</span>
                <strong>
                  {dimension.letter} {dimension.confidence}%
                </strong>
                <p>
                  {dimension.left} / {dimension.right}
                </p>
              </article>
            ))}
          </section>

          <section className="grid two-up">
            {sections.map((section) => (
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

          <section className="panel-card">
            <div className="section-heading section-heading--tight">
              <div>
                <p className="eyebrow">AI Readout</p>
                <h2>AI 个性化解读</h2>
              </div>
              <Link className="inline-link inline-link--subtle" href="/settings">
                设置模型
              </Link>
            </div>

            <div className="button-row">
              <button className="button" disabled={aiLoading} onClick={generateAiReadout} type="button">
                {aiLoading ? "生成中..." : "生成 AI 解读"}
              </button>
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
                AI 会结合你的类型结果、当前场景和关注问题，输出一版更具体的沟通与工作建议。
              </p>
            )}
          </section>
        </>
      )}
    </div>
  );
}
