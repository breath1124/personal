export type ReactionTaskId = "flash" | "choice" | "inhibition";

export type ReactionOutcome =
  | "success"
  | "anticipation"
  | "wrong"
  | "commission"
  | "omission"
  | "withhold";

export type ReactionInput = "tap" | "Space" | "ArrowUp" | "ArrowRight" | "ArrowDown" | "ArrowLeft";

export type ReactionTaskDefinition = {
  id: ReactionTaskId;
  title: string;
  shortLabel: string;
  eyebrow: string;
  rounds: number;
  duration: string;
  description: string;
  responseHint: string;
  focus: string;
  accent: string;
};

export type ReactionTrial = {
  taskId: ReactionTaskId;
  round: number;
  prompt: string;
  expectedInput: ReactionInput | null;
  responseInput: ReactionInput | null;
  latencyMs: number | null;
  correct: boolean;
  outcome: ReactionOutcome;
  recordedAt: string;
};

export type ReactionTaskAnalysis = {
  taskId: ReactionTaskId;
  title: string;
  shortLabel: string;
  score: number;
  accuracyPct: number;
  medianLatencyMs: number | null;
  bestLatencyMs: number | null;
  slowestLatencyMs: number | null;
  consistencyPct: number | null;
  fatigueDeltaMs: number | null;
  anticipationCount: number;
  omissionCount: number;
  wrongCount: number;
  commissionCount: number;
  completedRounds: number;
  latencySeries: number[];
  headline: string;
  insight: string;
};

export type ReactionSuiteAnalysis = {
  totalScore: number;
  speedScore: number;
  stabilityScore: number;
  controlScore: number;
  accuracyScore: number;
  switchingCostMs: number | null;
  profileLabel: string;
  summary: string;
  strengths: string[];
  nextFocus: string[];
  tasks: ReactionTaskAnalysis[];
};

export const REACTION_TASKS: ReactionTaskDefinition[] = [
  {
    id: "flash",
    title: "闪光捕捉",
    shortLabel: "简单反应",
    eyebrow: "Simple Reaction",
    rounds: 7,
    duration: "约 1 分钟",
    description: "看到信号出现后立刻出手，测启动速度和抢拍倾向。",
    responseHint: "看到亮起后点击中央区域，或按空格。",
    focus: "启动速度",
    accent: "var(--accent)"
  },
  {
    id: "choice",
    title: "方向切换",
    shortLabel: "选择反应",
    eyebrow: "Choice Reaction",
    rounds: 10,
    duration: "约 90 秒",
    description: "根据信号方向做出对应动作，测识别与切换成本。",
    responseHint: "按方向键，或点击对应的方向按钮。",
    focus: "选择速度",
    accent: "#7c3aed"
  },
  {
    id: "inhibition",
    title: "抑制控制",
    shortLabel: "Go / No-Go",
    eyebrow: "Inhibitory Control",
    rounds: 12,
    duration: "约 90 秒",
    description: "该出手时要快，不该出手时要忍住，测冲动控制和注意维持。",
    responseHint: "只对蓝色圆点响应，看到琥珀方块保持不动。",
    focus: "冲动控制",
    accent: "#0f766e"
  }
];

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function round(value: number, digits = 0) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function average(values: number[]) {
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function median(values: number[]) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1] + sorted[middle]) / 2
    : sorted[middle];
}

function stdev(values: number[]) {
  if (values.length < 2) return null;
  const mean = average(values);
  if (mean === null) return null;
  const variance =
    values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (values.length - 1);
  return Math.sqrt(variance);
}

function coefficientOfVariation(values: number[]) {
  const mean = average(values);
  const deviation = stdev(values);
  if (!mean || !deviation) return null;
  return (deviation / mean) * 100;
}

function fatigueDelta(values: number[]) {
  if (values.length < 4) return null;
  const middle = Math.ceil(values.length / 2);
  const firstHalf = values.slice(0, middle);
  const secondHalf = values.slice(middle);
  const firstMean = average(firstHalf);
  const secondMean = average(secondHalf);

  if (firstMean === null || secondMean === null) return null;
  return secondMean - firstMean;
}

function latencyScore(medianLatencyMs: number | null, idealMs: number, windowMs: number) {
  if (medianLatencyMs === null) return 20;
  const raw = 100 - ((medianLatencyMs - idealMs) / windowMs) * 35;
  return clamp(round(raw), 15, 100);
}

function stabilityScore(consistencyPct: number | null, fatigueMs: number | null) {
  const consistencyPart =
    consistencyPct === null ? 55 : clamp(100 - consistencyPct * 2.2, 20, 100);
  const fatiguePart =
    fatigueMs === null ? 70 : clamp(100 - Math.max(fatigueMs, 0) * 0.35, 15, 100);
  return round(consistencyPart * 0.7 + fatiguePart * 0.3);
}

function controlScore(
  anticipationCount: number,
  commissionCount: number,
  wrongCount: number,
  omissionCount: number
) {
  return clamp(
    round(100 - anticipationCount * 12 - commissionCount * 14 - wrongCount * 8 - omissionCount * 10),
    10,
    100
  );
}

function buildHeadline(taskId: ReactionTaskId, score: number) {
  if (taskId === "flash") {
    if (score >= 82) return "起动很干净";
    if (score >= 65) return "速度在线";
    return "启动还有余量";
  }

  if (taskId === "choice") {
    if (score >= 82) return "切换很顺";
    if (score >= 65) return "判断稳定";
    return "切换成本偏高";
  }

  if (score >= 82) return "收放自如";
  if (score >= 65) return "控制尚稳";
  return "抑制容易松";
}

function buildTaskInsight(
  taskId: ReactionTaskId,
  medianLatencyMs: number | null,
  accuracyPct: number,
  consistencyPct: number | null,
  anticipationCount: number,
  omissionCount: number,
  wrongCount: number,
  commissionCount: number,
  fatigueMs: number | null
) {
  const speedText =
    medianLatencyMs === null
      ? "有效反应样本还不够。"
      : `中位反应时约 ${round(medianLatencyMs)}ms，`;

  const consistencyText =
    consistencyPct === null
      ? "波动样本较少。"
      : consistencyPct <= 12
        ? "波动控制得比较稳。"
        : consistencyPct <= 20
          ? "波动还算可控。"
          : "波动偏大，容易一轮很快一轮很慢。";

  if (taskId === "flash") {
    if (anticipationCount > 1) {
      return `${speedText}但抢拍偏多，说明你会倾向于“提前出手”而不是等信号完全落地。${consistencyText}`;
    }

    return `${speedText}准确率 ${round(accuracyPct)}%。${consistencyText}`;
  }

  if (taskId === "choice") {
    if (wrongCount > 1) {
      return `${speedText}方向判断有 ${wrongCount} 次偏差，说明在切换任务规则时还有一点犹豫。${consistencyText}`;
    }

    return `${speedText}方向判断准确率 ${round(accuracyPct)}%，说明视觉识别到动作映射整体比较顺。${consistencyText}`;
  }

  if (commissionCount > 1) {
    return `${speedText}但误击了 ${commissionCount} 次不该出手的信号，抑制控制是当前更值得补的那一项。${consistencyText}`;
  }

  if (omissionCount > 1) {
    return `${speedText}不过漏掉了 ${omissionCount} 次该出手的目标，说明注意力在后半程会松。${consistencyText}`;
  }

  if (fatigueMs !== null && fatigueMs > 60) {
    return `${speedText}后半程明显慢了 ${round(fatigueMs)}ms，说明耐受和维持力还有提升空间。${consistencyText}`;
  }

  return `${speedText}控制和收手都比较克制，说明你不是单纯“快”，而是能在速度和规则之间保持平衡。${consistencyText}`;
}

function analyzeTask(task: ReactionTaskDefinition, trials: ReactionTrial[]): ReactionTaskAnalysis {
  const correctTrials = trials.filter((trial) => trial.correct);
  const latencySeries = trials
    .filter((trial) => trial.outcome === "success" && typeof trial.latencyMs === "number")
    .map((trial) => trial.latencyMs as number);

  const anticipationCount = trials.filter((trial) => trial.outcome === "anticipation").length;
  const omissionCount = trials.filter((trial) => trial.outcome === "omission").length;
  const wrongCount = trials.filter((trial) => trial.outcome === "wrong").length;
  const commissionCount = trials.filter((trial) => trial.outcome === "commission").length;

  const accuracyPct =
    task.rounds > 0 ? (correctTrials.length / task.rounds) * 100 : 0;
  const medianLatencyMs = median(latencySeries);
  const bestLatencyMs = latencySeries.length > 0 ? Math.min(...latencySeries) : null;
  const slowestLatencyMs = latencySeries.length > 0 ? Math.max(...latencySeries) : null;
  const consistencyPct = coefficientOfVariation(latencySeries);
  const fatigueMs = fatigueDelta(latencySeries);

  const speed =
    task.id === "flash"
      ? latencyScore(medianLatencyMs, 250, 180)
      : task.id === "choice"
        ? latencyScore(medianLatencyMs, 520, 220)
        : latencyScore(medianLatencyMs, 460, 180);
  const stability = stabilityScore(consistencyPct, fatigueMs);
  const control = controlScore(anticipationCount, commissionCount, wrongCount, omissionCount);
  const score = round(speed * 0.35 + stability * 0.25 + control * 0.2 + accuracyPct * 0.2);

  return {
    taskId: task.id,
    title: task.title,
    shortLabel: task.shortLabel,
    score,
    accuracyPct: round(accuracyPct, 1),
    medianLatencyMs: medianLatencyMs === null ? null : round(medianLatencyMs),
    bestLatencyMs: bestLatencyMs === null ? null : round(bestLatencyMs),
    slowestLatencyMs: slowestLatencyMs === null ? null : round(slowestLatencyMs),
    consistencyPct: consistencyPct === null ? null : round(consistencyPct, 1),
    fatigueDeltaMs: fatigueMs === null ? null : round(fatigueMs),
    anticipationCount,
    omissionCount,
    wrongCount,
    commissionCount,
    completedRounds: trials.length,
    latencySeries: latencySeries.map((value) => round(value)),
    headline: buildHeadline(task.id, score),
    insight: buildTaskInsight(
      task.id,
      medianLatencyMs,
      accuracyPct,
      consistencyPct,
      anticipationCount,
      omissionCount,
      wrongCount,
      commissionCount,
      fatigueMs
    )
  };
}

export function analyzeReactionSuite(trials: ReactionTrial[]): ReactionSuiteAnalysis {
  const taskAnalyses = REACTION_TASKS.map((task) =>
    analyzeTask(
      task,
      trials
        .filter((trial) => trial.taskId === task.id)
        .sort((a, b) => a.round - b.round)
    )
  );

  const speedScore = round(average(taskAnalyses.map((task) => task.score)) ?? 0);
  const stability = round(
    average(
      taskAnalyses.map((task) => (task.consistencyPct === null ? 55 : clamp(100 - task.consistencyPct * 2, 20, 100)))
    ) ?? 0
  );
  const control = round(
    average(
      taskAnalyses.map((task) =>
        controlScore(task.anticipationCount, task.commissionCount, task.wrongCount, task.omissionCount)
      )
    ) ?? 0
  );
  const accuracy = round(average(taskAnalyses.map((task) => task.accuracyPct)) ?? 0);
  const totalScore = round(speedScore * 0.35 + stability * 0.2 + control * 0.25 + accuracy * 0.2);

  const flashMedian = taskAnalyses.find((task) => task.taskId === "flash")?.medianLatencyMs ?? null;
  const choiceMedian = taskAnalyses.find((task) => task.taskId === "choice")?.medianLatencyMs ?? null;
  const switchingCostMs =
    flashMedian !== null && choiceMedian !== null ? choiceMedian - flashMedian : null;

  let profileLabel = "稳定执行型";
  if (control < 62 && speedScore >= 72) {
    profileLabel = "冲得快但容易抢拍";
  } else if (speedScore >= 80 && stability >= 75 && control >= 72) {
    profileLabel = "敏捷且稳";
  } else if (accuracy >= 84 && speedScore < 70) {
    profileLabel = "判断稳但还有提速空间";
  } else if (stability < 60) {
    profileLabel = "速度波动偏大";
  }

  const strengths: string[] = [];
  const nextFocus: string[] = [];

  if (speedScore >= 78) {
    strengths.push("启动和切换都比较快，不需要靠“慢下来”换稳定。");
  }
  if (control >= 78) {
    strengths.push("在该收手的时候能收住，不是单纯靠冒进拿速度。");
  }
  if (stability >= 75) {
    strengths.push("同一组测试里的波动较小，说明输出比较稳定。");
  }
  if (switchingCostMs !== null && switchingCostMs <= 180) {
    strengths.push("从看见信号到选动作的切换成本不高。");
  }

  if (strengths.length === 0) {
    strengths.push("基础反应链条已经成形，继续打磨更像是“优化”而不是从零开始。");
  }

  if (switchingCostMs !== null && switchingCostMs > 230) {
    nextFocus.push("选择反应比简单反应慢得明显，说明识别到动作映射这一段还有压缩空间。");
  }
  if (control < 68) {
    nextFocus.push("先把抢拍、误击和漏击压下来，再追求更极端的速度会更划算。");
  }
  if (stability < 65) {
    nextFocus.push("建议用更短的多组训练，把每轮波动先压小，再拉极限速度。");
  }
  if (taskAnalyses.some((task) => (task.fatigueDeltaMs ?? 0) > 60)) {
    nextFocus.push("后半程有明显变慢，说明注意维持和节奏恢复值得补。");
  }

  if (nextFocus.length === 0) {
    nextFocus.push("下一步更适合做更复杂的双任务或节奏干扰训练，而不是继续刷同样难度。");
  }

  return {
    totalScore,
    speedScore,
    stabilityScore: stability,
    controlScore: control,
    accuracyScore: accuracy,
    switchingCostMs: switchingCostMs === null ? null : round(switchingCostMs),
    profileLabel,
    summary:
      profileLabel === "敏捷且稳"
        ? "你的速度、稳定性和抑制控制比较平衡，属于已经能把快和稳同时拿住的类型。"
        : profileLabel === "冲得快但容易抢拍"
          ? "你有明显的启动冲劲，但更值得补的是“等信号完全落地再出手”的控制力。"
          : profileLabel === "判断稳但还有提速空间"
            ? "你更像稳健型选手，判断正确率不错，提速空间主要在动作启动和切换。"
            : "你已经有可用的反应基础，下一步重点是把波动压小，让每轮输出更接近。 ",
    strengths,
    nextFocus,
    tasks: taskAnalyses
  };
}
