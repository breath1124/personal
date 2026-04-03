"use client";

import { useEffect, useEffectEvent, useMemo, useRef, useState } from "react";
import { Sparkline } from "@/components/sparkline";
import {
  REACTION_TASKS,
  analyzeReactionSuite,
  type ReactionInput,
  type ReactionOutcome,
  type ReactionTaskDefinition,
  type ReactionTaskId,
  type ReactionTrial
} from "@/lib/reaction";

type Phase = "intro" | "waiting" | "active" | "feedback" | "task-summary" | "finished";

type RuntimeStimulus = {
  taskId: ReactionTaskId;
  prompt: string;
  waitingCopy: string;
  activeCopy: string;
  expectedInput: ReactionInput | null;
  delayMs: number;
  responseWindowMs: number;
  visualLabel: string;
  visualSubLabel: string;
  visualKind: "orb" | "arrow" | "square";
  visualTone: "neutral" | "active" | "warning";
  directionKey?: ReactionInput;
};

const DIRECTION_STIMULI: Array<{ key: ReactionInput; symbol: string; label: string }> = [
  { key: "ArrowUp", symbol: "↑", label: "上" },
  { key: "ArrowRight", symbol: "→", label: "右" },
  { key: "ArrowDown", symbol: "↓", label: "下" },
  { key: "ArrowLeft", symbol: "←", label: "左" }
];

function randomBetween(min: number, max: number) {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function pickOne<T>(items: T[]) {
  return items[Math.floor(Math.random() * items.length)];
}

function formatMs(value: number | null) {
  return value === null ? "-" : `${Math.round(value)} ms`;
}

function outcomeLabel(outcome: ReactionOutcome) {
  switch (outcome) {
    case "success":
      return "命中";
    case "anticipation":
      return "抢拍";
    case "wrong":
      return "按错";
    case "commission":
      return "误击";
    case "omission":
      return "漏击";
    case "withhold":
      return "克制成功";
    default:
      return outcome;
  }
}

function buildStimulus(task: ReactionTaskDefinition): RuntimeStimulus {
  if (task.id === "flash") {
    return {
      taskId: task.id,
      prompt: "看到蓝色脉冲后再出手。",
      waitingCopy: "等待信号，不要抢拍。",
      activeCopy: "现在，立刻点击或按空格。",
      expectedInput: "tap",
      delayMs: randomBetween(1200, 2500),
      responseWindowMs: 2200,
      visualLabel: "准备",
      visualSubLabel: "等到变亮再点击",
      visualKind: "orb",
      visualTone: "neutral"
    };
  }

  if (task.id === "choice") {
    const cue = pickOne(DIRECTION_STIMULI);
    return {
      taskId: task.id,
      prompt: "看到方向后，按对应方向键或按钮。",
      waitingCopy: "方向还没出现，先别动。",
      activeCopy: `立刻做出 ${cue.label} 方向反应。`,
      expectedInput: cue.key,
      delayMs: randomBetween(900, 1900),
      responseWindowMs: 2400,
      visualLabel: cue.symbol,
      visualSubLabel: `目标方向：${cue.label}`,
      visualKind: "arrow",
      visualTone: "active",
      directionKey: cue.key
    };
  }

  const isGo = Math.random() < 0.72;
  return {
    taskId: task.id,
    prompt: "只对蓝色圆点响应，琥珀方块保持不动。",
    waitingCopy: "先等信号完全出现。",
    activeCopy: isGo ? "蓝色圆点，立即响应。" : "琥珀方块，保持不动。",
    expectedInput: isGo ? "tap" : null,
    delayMs: randomBetween(800, 1700),
    responseWindowMs: 1500,
    visualLabel: isGo ? "GO" : "HOLD",
    visualSubLabel: isGo ? "蓝色圆点" : "不要点",
    visualKind: isGo ? "orb" : "square",
    visualTone: isGo ? "active" : "warning"
  };
}

export function ReactionLab() {
  const [phase, setPhase] = useState<Phase>("intro");
  const [taskIndex, setTaskIndex] = useState(0);
  const [roundIndex, setRoundIndex] = useState(0);
  const [stimulus, setStimulus] = useState<RuntimeStimulus | null>(null);
  const [results, setResults] = useState<ReactionTrial[]>([]);
  const [status, setStatus] = useState("准备好以后再开始。");
  const [feedback, setFeedback] = useState("三段测试会连续完成，最终给你一份完整分析。");

  const triggerTimerRef = useRef<number | null>(null);
  const responseTimerRef = useRef<number | null>(null);
  const nextTimerRef = useRef<number | null>(null);
  const activatedAtRef = useRef<number | null>(null);
  const runtimeStimulusRef = useRef<RuntimeStimulus | null>(null);

  const currentTask = REACTION_TASKS[taskIndex];
  const completedCount = results.length;
  const totalRounds = REACTION_TASKS.reduce((sum, task) => sum + task.rounds, 0);
  const progressPct = Math.round((completedCount / totalRounds) * 100);
  const analysis = useMemo(() => analyzeReactionSuite(results), [results]);
  const currentTaskAnalysis =
    analysis.tasks.find((task) => task.taskId === currentTask.id) ?? analysis.tasks[taskIndex];

  function clearTimers() {
    if (triggerTimerRef.current) {
      window.clearTimeout(triggerTimerRef.current);
      triggerTimerRef.current = null;
    }
    if (responseTimerRef.current) {
      window.clearTimeout(responseTimerRef.current);
      responseTimerRef.current = null;
    }
    if (nextTimerRef.current) {
      window.clearTimeout(nextTimerRef.current);
      nextTimerRef.current = null;
    }
  }

  function resetSuite() {
    clearTimers();
    activatedAtRef.current = null;
    runtimeStimulusRef.current = null;
    setResults([]);
    setTaskIndex(0);
    setRoundIndex(0);
    setStimulus(null);
    setPhase("intro");
    setStatus("准备好以后再开始。");
    setFeedback("三段测试会连续完成，最终给你一份完整分析。");
  }

  function prepareTask(nextTaskIndex: number) {
    clearTimers();
    activatedAtRef.current = null;
    runtimeStimulusRef.current = null;
    setTaskIndex(nextTaskIndex);
    setRoundIndex(0);
    setStimulus(null);
    setPhase("intro");
    setStatus("阅读规则后开始。");
    setFeedback(REACTION_TASKS[nextTaskIndex].description);
  }

  function scheduleRound(nextTaskIndex: number, nextRoundIndex: number) {
    clearTimers();
    const task = REACTION_TASKS[nextTaskIndex];
    const nextStimulus = buildStimulus(task);

    runtimeStimulusRef.current = nextStimulus;
    activatedAtRef.current = null;
    setTaskIndex(nextTaskIndex);
    setRoundIndex(nextRoundIndex);
    setStimulus(nextStimulus);
    setPhase("waiting");
    setStatus(nextStimulus.waitingCopy);
    setFeedback(nextStimulus.prompt);

    triggerTimerRef.current = window.setTimeout(() => {
      activatedAtRef.current = performance.now();
      setPhase("active");
      setStatus(nextStimulus.activeCopy);

      responseTimerRef.current = window.setTimeout(() => {
        finalizeTrial(null, nextStimulus.expectedInput === null ? "withhold" : "omission");
      }, nextStimulus.responseWindowMs);
    }, nextStimulus.delayMs);
  }

  function advanceAfterTrial(currentTaskIndex: number, currentRoundIndex: number) {
    const task = REACTION_TASKS[currentTaskIndex];

    nextTimerRef.current = window.setTimeout(() => {
      if (currentRoundIndex + 1 < task.rounds) {
        scheduleRound(currentTaskIndex, currentRoundIndex + 1);
        return;
      }

      if (currentTaskIndex + 1 < REACTION_TASKS.length) {
        setStimulus(null);
        setPhase("task-summary");
        setStatus(`${task.title} 已完成。`);
        setFeedback("看一眼这一项的表现，再进入下一项。");
        return;
      }

      setStimulus(null);
      setPhase("finished");
      setStatus("整套测试完成。");
      setFeedback("下面是你的完整反应分析。");
    }, 860);
  }

  function finalizeTrial(input: ReactionInput | null, forcedOutcome?: ReactionOutcome) {
    const activeStimulus = runtimeStimulusRef.current;
    if (!activeStimulus) return;

    clearTimers();

    const task = REACTION_TASKS[taskIndex];
    const activatedAt = activatedAtRef.current;
    const latencyMs =
      activatedAt && input ? Math.max(0, performance.now() - activatedAt) : null;

    let outcome: ReactionOutcome = forcedOutcome ?? "success";
    let correct = false;

    if (forcedOutcome) {
      correct = forcedOutcome === "withhold";
    } else if (phase === "waiting") {
      outcome = "anticipation";
      correct = false;
    } else if (task.id === "choice") {
      outcome = input === activeStimulus.expectedInput ? "success" : "wrong";
      correct = outcome === "success";
    } else if (task.id === "inhibition") {
      if (activeStimulus.expectedInput === null) {
        outcome = "commission";
        correct = false;
      } else {
        outcome = "success";
        correct = true;
      }
    } else {
      outcome = "success";
      correct = true;
    }

    const trial: ReactionTrial = {
      taskId: task.id,
      round: roundIndex + 1,
      prompt: activeStimulus.visualSubLabel,
      expectedInput: activeStimulus.expectedInput,
      responseInput: input,
      latencyMs: latencyMs === null ? null : Math.round(latencyMs),
      correct,
      outcome,
      recordedAt: new Date().toISOString()
    };

    setResults((current) => [...current, trial]);
    setPhase("feedback");

    if (outcome === "success") {
      setStatus(`${outcomeLabel(outcome)} · ${formatMs(trial.latencyMs)}`);
      setFeedback("这轮处理很干净。");
    } else if (outcome === "withhold") {
      setStatus("克制成功");
      setFeedback("这类题关键不是快，而是稳住不误击。");
    } else if (outcome === "anticipation") {
      setStatus("太快了，算抢拍");
      setFeedback("这一轮不是反应慢，而是信号还没出现就出手了。");
    } else if (outcome === "wrong") {
      setStatus("方向按错");
      setFeedback("选择反应不只看速度，更看识别后映射得准不准。");
    } else if (outcome === "commission") {
      setStatus("误击了不该点的信号");
      setFeedback("抑制控制的核心是该停就停。");
    } else {
      setStatus("这一轮漏掉了");
      setFeedback("后半程容易漏击，通常意味着注意维持在下降。");
    }

    advanceAfterTrial(taskIndex, roundIndex);
  }

  const onKeyDown = useEffectEvent((event: KeyboardEvent) => {
    if (event.repeat) return;

    const key =
      event.key === " "
        ? "Space"
        : (event.key as ReactionInput);

    if (!["Space", "ArrowUp", "ArrowRight", "ArrowDown", "ArrowLeft"].includes(key)) {
      return;
    }

    if (phase === "waiting" || phase === "active") {
      event.preventDefault();
    }

    const task = REACTION_TASKS[taskIndex];
    if (task.id === "choice" && key.startsWith("Arrow")) {
      finalizeTrial(key);
      return;
    }

    if (key === "Space") {
      finalizeTrial("Space");
    }
  });

  useEffect(() => {
    const listener = (event: KeyboardEvent) => onKeyDown(event);
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, [onKeyDown]);

  useEffect(() => () => clearTimers(), []);

  const currentTaskResults = results.filter((trial) => trial.taskId === currentTask.id);
  const recentTrials = [...results].slice(-12).reverse();

  return (
    <div className="section-stack">
      <section className="reaction-module-grid">
        {REACTION_TASKS.map((task, index) => (
          <article
            className={`panel-card reaction-module-card${index === taskIndex ? " is-active" : ""}`}
            key={task.id}
          >
            <p className="eyebrow">{task.eyebrow}</p>
            <div className="stat-row">
              <h2>{task.title}</h2>
              <span className="tone-pill">{task.duration}</span>
            </div>
            <p>{task.description}</p>
            <div className="app-card__chips">
              <span className="chip">{task.shortLabel}</span>
              <span className="chip">{task.focus}</span>
              <span className="chip">{task.rounds} 轮</span>
            </div>
          </article>
        ))}
      </section>

      <section className="grid reaction-layout">
        <article className="panel-card reaction-stage-card">
          <div className="section-heading section-heading--tight">
            <div>
              <p className="eyebrow">Reaction Run</p>
              <h2>{currentTask.title}</h2>
            </div>
            <span className="muted">
              第 {taskIndex + 1}/{REACTION_TASKS.length} 项 · 第 {Math.min(roundIndex + 1, currentTask.rounds)}/
              {currentTask.rounds} 轮
            </span>
          </div>

          <div className="progress-track" aria-hidden="true">
            <span className="progress-fill" style={{ width: `${progressPct}%` }} />
          </div>

          <div className={`reaction-stage reaction-stage--${phase}`}>
            {phase === "intro" && (
              <div className="reaction-callout">
                <p className="eyebrow">Rule</p>
                <h3>{currentTask.shortLabel}</h3>
                <p>{currentTask.responseHint}</p>
                <button
                  className="button"
                  onClick={() => scheduleRound(taskIndex, 0)}
                  type="button"
                >
                  {taskIndex === 0 ? "开始整套测试" : "开始这一项"}
                </button>
              </div>
            )}

            {(phase === "waiting" || phase === "active" || phase === "feedback") && stimulus && (
              <div className="reaction-active-stack">
                <button
                  className={`reaction-target reaction-target--${stimulus.visualKind} reaction-target--${phase} reaction-target--${stimulus.visualTone}`}
                  onClick={() => finalizeTrial("tap")}
                  type="button"
                >
                  <span className="reaction-target__label">{stimulus.visualLabel}</span>
                  <span className="reaction-target__sub">{stimulus.visualSubLabel}</span>
                </button>

                {currentTask.id === "choice" && (
                  <div className="reaction-pad">
                    {DIRECTION_STIMULI.map((item) => (
                      <button
                        className={`reaction-pad__key${stimulus.directionKey === item.key ? " is-highlight" : ""}`}
                        key={item.key}
                        onClick={() => finalizeTrial(item.key)}
                        type="button"
                      >
                        <span>{item.symbol}</span>
                        <small>{item.label}</small>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}

            {phase === "task-summary" && (
              <div className="reaction-callout">
                <p className="eyebrow">Task Snapshot</p>
                <h3>{currentTaskAnalysis?.headline ?? "这一项完成了"}</h3>
                <p>{currentTaskAnalysis?.insight ?? "继续下一项，完整报告会在最后给出。"}</p>
                <button
                  className="button"
                  onClick={() => prepareTask(taskIndex + 1)}
                  type="button"
                >
                  进入下一项
                </button>
              </div>
            )}

            {phase === "finished" && (
              <div className="reaction-callout">
                <p className="eyebrow">Finished</p>
                <h3>{analysis.profileLabel}</h3>
                <p>{analysis.summary}</p>
                <button className="button" onClick={resetSuite} type="button">
                  重新测试
                </button>
              </div>
            )}
          </div>
        </article>

        <div className="section-stack">
          <article className="panel-card">
            <div className="section-heading section-heading--tight">
              <div>
                <p className="eyebrow">Live Signal</p>
                <h2>当前状态</h2>
              </div>
              <span className="tone-pill">{progressPct}%</span>
            </div>
            <p className="reaction-status">{status}</p>
            <p className="muted">{feedback}</p>
          </article>

          <article className="panel-card">
            <div className="section-heading section-heading--tight">
              <div>
                <p className="eyebrow">Current Task</p>
                <h2>实时指标</h2>
              </div>
            </div>
            <section className="stats-grid stats-grid--mini">
              <article className="mini-stat">
                <span>已完成</span>
                <strong>{currentTaskResults.length}</strong>
                <p>{currentTask.rounds} 轮</p>
              </article>
              <article className="mini-stat">
                <span>准确率</span>
                <strong>{currentTaskAnalysis?.accuracyPct ?? 0}%</strong>
                <p>{currentTask.shortLabel}</p>
              </article>
              <article className="mini-stat">
                <span>中位反应</span>
                <strong>{currentTaskAnalysis?.medianLatencyMs ?? "-"}</strong>
                <p>ms</p>
              </article>
            </section>

            <article className="subtle-card">
              <h3>这一项在测什么</h3>
              <ul className="bullet-list">
                <li>{currentTask.responseHint}</li>
                <li>{currentTask.description}</li>
                <li>测试过程中会自动记录抢拍、误击、漏击和后半程变化。</li>
              </ul>
            </article>
          </article>

          <article className="panel-card">
            <div className="section-heading section-heading--tight">
              <div>
                <p className="eyebrow">Recent Trials</p>
                <h2>逐轮回看</h2>
              </div>
            </div>
            {recentTrials.length > 0 ? (
              <div className="trial-list">
                {recentTrials.map((trial) => (
                  <article className="trial-row" key={`${trial.taskId}-${trial.round}-${trial.recordedAt}`}>
                    <div>
                      <strong>
                        {REACTION_TASKS.find((task) => task.id === trial.taskId)?.shortLabel} · 第 {trial.round} 轮
                      </strong>
                      <p className="muted">{trial.prompt}</p>
                    </div>
                    <div className="trial-row__meta">
                      <span className={`tone-pill tone-pill--${trial.correct ? "positive" : "watch"}`}>
                        {outcomeLabel(trial.outcome)}
                      </span>
                      <strong>{formatMs(trial.latencyMs)}</strong>
                    </div>
                  </article>
                ))}
              </div>
            ) : (
              <p className="muted">开始测试后，这里会记录每一轮的结果。</p>
            )}
          </article>
        </div>
      </section>

      {phase === "finished" && (
        <>
          <section className="stats-grid">
            <article className="panel-card stat-card">
              <span>综合分</span>
              <strong>{analysis.totalScore}</strong>
              <p>{analysis.profileLabel}</p>
            </article>
            <article className="panel-card stat-card">
              <span>速度</span>
              <strong>{analysis.speedScore}</strong>
              <p>启动与切换综合</p>
            </article>
            <article className="panel-card stat-card">
              <span>稳定性</span>
              <strong>{analysis.stabilityScore}</strong>
              <p>波动与后半程维持</p>
            </article>
            <article className="panel-card stat-card">
              <span>控制力</span>
              <strong>{analysis.controlScore}</strong>
              <p>抢拍、误击与漏击</p>
            </article>
          </section>

          <section className="grid two-up">
            <article className="panel-card">
              <div className="section-heading section-heading--tight">
                <div>
                  <p className="eyebrow">Summary</p>
                  <h2>综合判断</h2>
                </div>
                <span className="muted">
                  切换成本 {analysis.switchingCostMs === null ? "-" : `${analysis.switchingCostMs}ms`}
                </span>
              </div>
              <p>{analysis.summary}</p>
              <h3>当前优势</h3>
              <ul className="bullet-list">
                {analysis.strengths.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
              <h3>下一步更值得练的点</h3>
              <ul className="bullet-list">
                {analysis.nextFocus.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </article>

            <article className="panel-card">
              <div className="section-heading section-heading--tight">
                <div>
                  <p className="eyebrow">Raw Data</p>
                  <h2>逐轮数据</h2>
                </div>
              </div>
              <div className="trial-table">
                <div className="trial-table__head">
                  <span>模块</span>
                  <span>轮次</span>
                  <span>结果</span>
                  <span>时延</span>
                </div>
                {results.map((trial) => (
                  <div className="trial-table__row" key={`${trial.taskId}-${trial.round}-${trial.recordedAt}`}>
                    <span>{REACTION_TASKS.find((task) => task.id === trial.taskId)?.shortLabel}</span>
                    <span>{trial.round}</span>
                    <span>{outcomeLabel(trial.outcome)}</span>
                    <span>{formatMs(trial.latencyMs)}</span>
                  </div>
                ))}
              </div>
            </article>
          </section>

          <section className="grid three-up">
            {analysis.tasks.map((task) => (
              <article className="panel-card" key={task.taskId}>
                <div className="section-heading section-heading--tight">
                  <div>
                    <p className="eyebrow">{task.shortLabel}</p>
                    <h2>{task.title}</h2>
                  </div>
                  <span className="tone-pill">{task.score} 分</span>
                </div>
                <div className="stats-grid stats-grid--mini">
                  <article className="mini-stat">
                    <span>准确率</span>
                    <strong>{task.accuracyPct}%</strong>
                    <p>{task.completedRounds} 轮</p>
                  </article>
                  <article className="mini-stat">
                    <span>中位反应</span>
                    <strong>{task.medianLatencyMs ?? "-"}</strong>
                    <p>ms</p>
                  </article>
                  <article className="mini-stat">
                    <span>波动系数</span>
                    <strong>{task.consistencyPct ?? "-"}</strong>
                    <p>%</p>
                  </article>
                </div>
                <p className="reaction-headline">{task.headline}</p>
                <p>{task.insight}</p>
                <Sparkline values={task.latencySeries} stroke={REACTION_TASKS.find((item) => item.id === task.taskId)?.accent} />
                <div className="app-card__chips">
                  {task.anticipationCount > 0 && <span className="chip">抢拍 {task.anticipationCount}</span>}
                  {task.wrongCount > 0 && <span className="chip">按错 {task.wrongCount}</span>}
                  {task.commissionCount > 0 && <span className="chip">误击 {task.commissionCount}</span>}
                  {task.omissionCount > 0 && <span className="chip">漏击 {task.omissionCount}</span>}
                  {task.fatigueDeltaMs !== null && (
                    <span className="chip">后半程 {task.fatigueDeltaMs > 0 ? "+" : ""}{task.fatigueDeltaMs}ms</span>
                  )}
                </div>
              </article>
            ))}
          </section>
        </>
      )}
    </div>
  );
}
