export type MbtiAxis = "EI" | "SN" | "TF" | "JP";

export type MbtiQuestion = {
  id: string;
  axis: MbtiAxis;
  prompt: string;
  left: string;
  right: string;
};

export const MBTI_QUESTIONS: MbtiQuestion[] = [
  {
    id: "q1",
    axis: "EI",
    prompt: "进入一个全新团队时，你通常会怎么打开局面？",
    left: "先与人互动，通过交流进入状态",
    right: "先观察环境，想清楚之后再加入"
  },
  {
    id: "q2",
    axis: "SN",
    prompt: "面对一个新问题时，你先抓哪一层信息？",
    left: "事实、边界和已经发生的细节",
    right: "模式、趋势和可能延展的方向"
  },
  {
    id: "q3",
    axis: "TF",
    prompt: "当别人来请教难题时，你更自然的起手式是？",
    left: "先拆问题和可执行方案",
    right: "先理解对方感受和真实诉求"
  },
  {
    id: "q4",
    axis: "JP",
    prompt: "面对一周工作安排，你更偏哪种节奏？",
    left: "先排好优先级和完成节点",
    right: "保留弹性，边推进边调整"
  },
  {
    id: "q5",
    axis: "EI",
    prompt: "累了一天后，什么最能让你恢复？",
    left: "和人聊天、出门走走、切换场景",
    right: "独处、整理思绪、沉浸兴趣"
  },
  {
    id: "q6",
    axis: "SN",
    prompt: "读一篇文章时，你更容易记住什么？",
    left: "定义、例子、结构和数据",
    right: "核心观点、隐含逻辑和趋势"
  },
  {
    id: "q7",
    axis: "TF",
    prompt: "做重要选择时，你更容易被什么说服？",
    left: "一致的逻辑和明确的收益",
    right: "价值感、关系感和长期感受"
  },
  {
    id: "q8",
    axis: "JP",
    prompt: "当项目中途冒出新机会时，你通常会？",
    left: "先看是否影响原计划，再决定要不要改",
    right: "只要更优，就愿意临时换轨"
  },
  {
    id: "q9",
    axis: "EI",
    prompt: "讨论会上你更接近哪种状态？",
    left: "想到就说，边说边澄清",
    right: "先想清楚，再挑重点表达"
  },
  {
    id: "q10",
    axis: "SN",
    prompt: "别人描述一个问题时，你更关心什么？",
    left: "具体发生了什么，细节是否一致",
    right: "这说明了什么，更深层模式是什么"
  },
  {
    id: "q11",
    axis: "TF",
    prompt: "团队出现分歧时，你本能会先守住什么？",
    left: "标准、事实和最优解",
    right: "关系、氛围和后续合作"
  },
  {
    id: "q12",
    axis: "JP",
    prompt: "对于待办事项，你更像哪种人？",
    left: "喜欢收口，一个一个完成",
    right: "喜欢并行探索，最后集中收束"
  },
  {
    id: "q13",
    axis: "EI",
    prompt: "需要做重大决定时，你更容易借助什么？",
    left: "和别人聊一轮，听反馈后定",
    right: "自己独处推演，想透再说"
  },
  {
    id: "q14",
    axis: "SN",
    prompt: "在做方案时，你更先注意什么？",
    left: "落地环节、执行成本和现实约束",
    right: "方向感、创意空间和长期潜力"
  },
  {
    id: "q15",
    axis: "TF",
    prompt: "给人反馈时，你更在意什么？",
    left: "说清问题和提升路径",
    right: "让对方能被接受并愿意继续合作"
  },
  {
    id: "q16",
    axis: "JP",
    prompt: "旅行时你更偏好哪种状态？",
    left: "路线大体确定，关键节点先排好",
    right: "留白更多，边走边决定"
  },
  {
    id: "q17",
    axis: "EI",
    prompt: "陌生社交局里，你更常见的状态是？",
    left: "比较快就能进入互动",
    right: "要先找到舒适感再参与"
  },
  {
    id: "q18",
    axis: "SN",
    prompt: "做复盘时，你更自然会复盘什么？",
    left: "过程细节、执行偏差和证据",
    right: "隐含模式、策略变化和方向判断"
  },
  {
    id: "q19",
    axis: "TF",
    prompt: "别人夸你时，更像是在夸你哪一点？",
    left: "理性、清楚、判断稳",
    right: "细腻、理解人、能共情"
  },
  {
    id: "q20",
    axis: "JP",
    prompt: "面对截止日期，你的默认策略更像？",
    left: "尽早推进，提前锁定结果",
    right: "前期探索，临近节点快速收束"
  },
  {
    id: "q21",
    axis: "EI",
    prompt: "遇到复杂任务时，你更想先怎么做？",
    left: "找人碰一轮，快速获得外部反馈",
    right: "先独立拆解，理顺后再沟通"
  },
  {
    id: "q22",
    axis: "SN",
    prompt: "你更容易被哪种表达打动？",
    left: "具体案例和清楚数据",
    right: "强洞察和高概括"
  },
  {
    id: "q23",
    axis: "TF",
    prompt: "当你不同意别人时，你更先处理什么？",
    left: "观点错在哪里",
    right: "怎样说更不伤害关系"
  },
  {
    id: "q24",
    axis: "JP",
    prompt: "面对多种选择时，你更喜欢？",
    left: "先排除法，尽快定下来",
    right: "先多看几种可能，再慢慢收敛"
  }
];

export const MBTI_LABELS = {
  INTJ: "系统策划者",
  INTP: "逻辑架构师",
  ENTJ: "推进指挥者",
  ENTP: "点子实验者",
  INFJ: "洞察协调者",
  INFP: "价值探索者",
  ENFJ: "关系驱动者",
  ENFP: "灵感连接者",
  ISTJ: "秩序执行者",
  ISFJ: "稳定支持者",
  ESTJ: "结果推进者",
  ESFJ: "氛围组织者",
  ISTP: "问题拆解者",
  ISFP: "体验感知者",
  ESTP: "行动试错者",
  ESFP: "现场调动者"
} as const;

const FOCUS_NOTES = {
  work: {
    E: "在工作里，外向倾向意味着你更适合把想法外显出来，通过同步推动事情前进。",
    I: "在工作里，内向倾向意味着你更适合先独立推演，再带着清晰结论参与讨论。",
    S: "你在执行与落地环节会更稳，更容易守住边界和现实约束。",
    N: "你在方向、概念和机会识别上会更有优势，适合负责前瞻判断。",
    T: "你更容易在优先级、标准和取舍上维持清晰度。",
    F: "你更容易在跨团队协同和用户感受上建立更好的接受度。",
    J: "你通常更适合搭建节奏和里程碑，推动事情收口。",
    P: "你更擅长在不确定环境里保留弹性和试错空间。"
  },
  team: {
    E: "你可以主动带起氛围和讨论，帮团队把隐性问题提到台面。",
    I: "你更适合输出深思后的判断，给团队提供稳定的思路锚点。",
    S: "你能帮团队守住事实、进度和细节质量。",
    N: "你能帮团队看到长期方向与潜在机会。",
    T: "你能把分歧重新拉回到标准与目标本身。",
    F: "你能减少合作摩擦，帮助团队维持信任感。",
    J: "你能让合作更可预期，形成明确的节奏感。",
    P: "你能让团队在变化中保持灵活，不会过早锁死路径。"
  },
  communication: {
    E: "你的表达优势在于即时互动，适合通过来回确认加速达成共识。",
    I: "你的表达优势在于整理后输出，适合先写要点再开口。",
    S: "先给事实、例子和细节，会让你的表达更有说服力。",
    N: "先给观点、结构和方向，会让别人更快抓住你的重点。",
    T: "你适合把判断标准直接讲清楚，减少模糊空间。",
    F: "你适合先照顾对方接受度，再推进结论。",
    J: "把时间点、边界和下一步说透，会让你更舒服。",
    P: "给出多个备选路径，比一次定死更适合你的表达风格。"
  },
  growth: {
    E: "你的成长关键在于筛选外部刺激，而不是被所有输入拉走。",
    I: "你的成长关键在于更早暴露想法，不要等到完全想透才行动。",
    S: "你的成长关键在于别只守住眼前细节，也要定期抬头看趋势。",
    N: "你的成长关键在于别只讲愿景，也要补足落地与证据。",
    T: "你的成长关键在于加入更多共情和关系判断，不要只追求逻辑正确。",
    F: "你的成长关键在于加入更多标准和边界，不要只追求关系和谐。",
    J: "你的成长关键在于允许探索阶段存在不确定，而不是过早收口。",
    P: "你的成长关键在于建立最基本的收束机制，避免一直开放式探索。"
  }
} as const;

export function scoreMbti(answers: Record<string, number>) {
  const axisBuckets: Record<MbtiAxis, number[]> = {
    EI: [],
    SN: [],
    TF: [],
    JP: []
  };

  for (const question of MBTI_QUESTIONS) {
    const score = answers[question.id];
    if (typeof score === "number") {
      axisBuckets[question.axis].push(score);
    }
  }

  const axisMeta = {
    EI: { left: "E", right: "I", label: "能量来源" },
    SN: { left: "S", right: "N", label: "信息加工" },
    TF: { left: "T", right: "F", label: "决策取向" },
    JP: { left: "J", right: "P", label: "行动节奏" }
  } as const;

  const dimensions = Object.entries(axisBuckets).map(([axis, values]) => {
    const total = values.reduce((sum, value) => sum + value, 0);
    const max = values.length * 2 || 1;
    const meta = axisMeta[axis as MbtiAxis];
    const letter = total > 0 ? meta.right : meta.left;
    const confidence = Math.round(50 + (Math.abs(total) / max) * 50);
    return {
      axis: axis as MbtiAxis,
      label: meta.label,
      left: meta.left,
      right: meta.right,
      letter,
      confidence
    };
  });

  const type = dimensions.map((item) => item.letter).join("");
  return {
    type,
    label: MBTI_LABELS[type as keyof typeof MBTI_LABELS] ?? "混合型探索者",
    dimensions
  };
}

export function buildMbtiReport(
  result: ReturnType<typeof scoreMbti>,
  focus: keyof typeof FOCUS_NOTES
) {
  const letters = Object.fromEntries(
    result.dimensions.map((dimension) => [dimension.axis, dimension.letter])
  ) as Record<MbtiAxis, string>;

  const insights = [
    FOCUS_NOTES[focus][letters.EI as keyof typeof FOCUS_NOTES.work],
    FOCUS_NOTES[focus][letters.SN as keyof typeof FOCUS_NOTES.work],
    FOCUS_NOTES[focus][letters.TF as keyof typeof FOCUS_NOTES.work],
    FOCUS_NOTES[focus][letters.JP as keyof typeof FOCUS_NOTES.work]
  ];

  return [
    {
      title: "你的工作底色",
      items: [
        `${result.type} 更接近“${result.label}”，说明你的偏好组合相对清晰。`,
        insights[0],
        insights[1]
      ]
    },
    {
      title: "你处理问题的方式",
      items: [insights[2], insights[3], "这类偏好不是能力上限，而是你默认最顺手的路径。"]
    },
    {
      title: "需要有意识补的部分",
      items: [
        "当你在熟悉场景里很顺时，往往也是盲区最容易被忽略的时候。",
        `当前最值得有意识补齐的，是和 ${result.type} 相反倾向那一侧的策略。`,
        "把结果当成观察自己习惯的镜子，而不是把自己锁死在某个标签里。"
      ]
    }
  ];
}
