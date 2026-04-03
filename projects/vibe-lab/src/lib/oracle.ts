import type { AnalysisSection } from "@/lib/types";

export type OracleFocus = "career" | "wealth" | "relationship" | "transition" | "growth";
export type OraclePhase = "steady" | "pressure" | "reset" | "expansion";
export type BirthPrecision = "exact" | "approx" | "unknown";
export type FiveElement = "木" | "火" | "土" | "金" | "水";
export type HeavenlyStem = "甲" | "乙" | "丙" | "丁" | "戊" | "己" | "庚" | "辛" | "壬" | "癸";
export type EarthlyBranch = "子" | "丑" | "寅" | "卯" | "辰" | "巳" | "午" | "未" | "申" | "酉" | "戌" | "亥";
export type EnergyRelation = "peer" | "output" | "wealth" | "authority" | "resource";

export type OracleDraft = {
  birthDate: string;
  birthTime: string;
  precision: BirthPrecision;
  focus: OracleFocus;
  phase: OraclePhase;
  question: string;
};

export type OraclePillar = {
  label: string;
  stem: HeavenlyStem;
  branch: EarthlyBranch;
  stemElement: FiveElement;
  branchElement: FiveElement;
};

export type OracleResult = {
  chartTitle: string;
  confidenceLabel: string;
  confidenceNote: string;
  focusLabel: string;
  phaseLabel: string;
  focusSummary: string;
  dayMaster: {
    stem: HeavenlyStem;
    element: FiveElement;
    title: string;
    brief: string;
  };
  pillars: OraclePillar[];
  elementBalance: Array<{
    element: FiveElement;
    count: number;
    ratio: number;
    tone: "positive" | "watch" | "calm";
    summary: string;
  }>;
  dominantElement: FiveElement;
  supportingElement: FiveElement;
  missingElement: FiveElement;
  rhythmLabel: string;
  relationMix: Array<{
    relation: EnergyRelation;
    label: string;
    count: number;
  }>;
  yearlyTheme: {
    pillar: string;
    relation: EnergyRelation;
    relationLabel: string;
    title: string;
    advice: string;
  };
  palace: {
    name: string;
    note: string;
  };
  bodyPalace: {
    name: string;
    note: string;
  };
  stars: Array<{
    name: string;
    tag: string;
    note: string;
  }>;
  domainScores: Array<{
    name: string;
    score: number;
    tone: "positive" | "watch" | "calm";
    note: string;
  }>;
  sections: AnalysisSection[];
  realityChecks: string[];
  actionExperiments: string[];
  calibrationQuestions: string[];
};

export const DEFAULT_ORACLE_DRAFT: OracleDraft = {
  birthDate: "",
  birthTime: "09:30",
  precision: "exact",
  focus: "career",
  phase: "steady",
  question: ""
};

const STEMS: HeavenlyStem[] = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"];
const BRANCHES: EarthlyBranch[] = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"];
const ELEMENTS: FiveElement[] = ["木", "火", "土", "金", "水"];

const STEM_ELEMENTS: Record<HeavenlyStem, FiveElement> = {
  甲: "木",
  乙: "木",
  丙: "火",
  丁: "火",
  戊: "土",
  己: "土",
  庚: "金",
  辛: "金",
  壬: "水",
  癸: "水"
};

const BRANCH_ELEMENTS: Record<EarthlyBranch, FiveElement> = {
  子: "水",
  丑: "土",
  寅: "木",
  卯: "木",
  辰: "土",
  巳: "火",
  午: "火",
  未: "土",
  申: "金",
  酉: "金",
  戌: "土",
  亥: "水"
};

const RELATION_LABELS: Record<EnergyRelation, string> = {
  peer: "比肩同频",
  output: "输出表达",
  wealth: "资源调度",
  authority: "规则压力",
  resource: "修复补能"
};

const FOCUS_LABELS: Record<OracleFocus, string> = {
  career: "事业推进",
  wealth: "财务与资源",
  relationship: "关系与亲密",
  transition: "换挡与迁移",
  growth: "自我成长"
};

const PHASE_LABELS: Record<OraclePhase, string> = {
  steady: "稳定经营期",
  pressure: "高压拉扯期",
  reset: "重整校准期",
  expansion: "扩张打开期"
};

const DAY_MASTER_TITLES: Record<HeavenlyStem, string> = {
  甲: "乔木型",
  乙: "藤木型",
  丙: "烈阳型",
  丁: "灯火型",
  戊: "城墙型",
  己: "田园型",
  庚: "刀锋型",
  辛: "珠玉型",
  壬: "江海型",
  癸: "雨露型"
};

const DAY_MASTER_BRIEFS: Record<HeavenlyStem, string> = {
  甲: "你更像先立主轴再发力的人，怕的是方向散，不怕的是工作硬。",
  乙: "你不是靠正面硬顶取胜，而是靠连接、渗透和持续经营拿结果。",
  丙: "你更适合看见全局后带头点火，优势是鼓舞与启动，盲点是容易过热。",
  丁: "你的强项是把细节照亮，把复杂情境处理得更有人味，也更有分寸。",
  戊: "你天然会往承托、稳定和搭框架那边走，适合做中盘与定盘的人。",
  己: "你擅长照料现实问题和具体过程，很多优势都藏在长期复利里。",
  庚: "你对边界、判断和切分很敏感，适合在混乱里快速做出清晰取舍。",
  辛: "你更重质感、标准和精修，很多价值来自把粗糙的东西磨成成品。",
  壬: "你适合在大信息流里穿梭，见势快，转身也快，讨厌被过死的结构困住。",
  癸: "你更擅长感知情境、捕捉细微变化，很多判断来自长时间的观察。"
};

const RHYTHM_LABELS: Record<string, string> = {
  木火: "生发点火型",
  木水: "感知生长型",
  火土: "推动落地型",
  火木: "外放破局型",
  土金: "结构掌控型",
  土火: "承压收束型",
  金水: "冷静洞察型",
  金土: "秩序建构型",
  水木: "流动策动型",
  水金: "观察判断型"
};

const PALACES = [
  {
    name: "命宫",
    note: "底层动机强，很多选择都会先回到“我到底要成为什么样的人”。"
  },
  {
    name: "兄弟宫",
    note: "合作与同伴关系对你影响大，环境里的人会明显改写你的节奏。"
  },
  {
    name: "夫妻宫",
    note: "亲密关系和深度连接容易成为你做长期决策时的重要变量。"
  },
  {
    name: "子女宫",
    note: "创造、作品和兴趣表达会成为你识别自我状态的窗口。"
  },
  {
    name: "财帛宫",
    note: "你对资源、回报和安全边界比较敏感，财务心智会牵动很多判断。"
  },
  {
    name: "疾厄宫",
    note: "身心能量管理是关键变量，一旦过载，其他维度会一起失真。"
  },
  {
    name: "迁移宫",
    note: "环境变动、出行、换挡和外部世界的拉力，对你有放大作用。"
  },
  {
    name: "交友宫",
    note: "你很容易被圈层、合作对象和长期共事的人影响叙事方向。"
  },
  {
    name: "官禄宫",
    note: "事业成就感和角色定位会是你当前最容易放大感知的主题。"
  },
  {
    name: "田宅宫",
    note: "稳定空间、归属感和长期布局比表面看起来更重要。"
  },
  {
    name: "福德宫",
    note: "精神恢复、独处质量和内在秩序决定了你的续航上限。"
  },
  {
    name: "父母宫",
    note: "标准、传承和外部评价体系对你的影响比你以为的更深。"
  }
] as const;

const STAR_ARCHETYPES = [
  { name: "紫微", tag: "主轴感", note: "你不太适合一直被别人定义，做事需要拥有自己的中心轴。" },
  { name: "天机", tag: "洞察快", note: "你对变化很敏感，判断往往先于语言成形。" },
  { name: "太阳", tag: "外放驱动", note: "你在看得见的场域里更容易进入高能状态。" },
  { name: "武曲", tag: "结果硬度", note: "你对效率、产出和可兑现性要求比较高。" },
  { name: "天同", tag: "柔性接口", note: "你做得好的时候，往往是让复杂场景先变得不那么刺人。" },
  { name: "廉贞", tag: "边界感", note: "你对秩序、规则和是否值得投入有明显判断。" },
  { name: "天府", tag: "蓄能稳盘", note: "你更像储能型选手，不适合被迫长期高频透支。" },
  { name: "太阴", tag: "内感精度", note: "很多决定不是突然想清楚，而是慢慢沉淀出把握。" },
  { name: "贪狼", tag: "体验张力", note: "你需要新鲜度和感受度，过于机械的生活会拖慢你。" },
  { name: "巨门", tag: "问题识别", note: "你很会发现系统里的噪音和漏洞，但也要防止陷在反复推敲里。" },
  { name: "天相", tag: "平衡能力", note: "你天然会帮人找平衡点，也因此容易替别人多扛一些。" },
  { name: "天梁", tag: "照拂责任", note: "你在很多关系里会不自觉进入顾全和托底的位置。" },
  { name: "七杀", tag: "突进魄力", note: "当机会清楚时，你有能力迅速切换到强推进模式。" },
  { name: "破军", tag: "换挡勇气", note: "你的人生关键节点往往不是小修小补，而是整段换轨。" }
] as const;

const MONTH_STARTS = [
  { code: 106, order: 12, branch: "丑" as EarthlyBranch },
  { code: 204, order: 1, branch: "寅" as EarthlyBranch },
  { code: 306, order: 2, branch: "卯" as EarthlyBranch },
  { code: 405, order: 3, branch: "辰" as EarthlyBranch },
  { code: 506, order: 4, branch: "巳" as EarthlyBranch },
  { code: 606, order: 5, branch: "午" as EarthlyBranch },
  { code: 707, order: 6, branch: "未" as EarthlyBranch },
  { code: 808, order: 7, branch: "申" as EarthlyBranch },
  { code: 908, order: 8, branch: "酉" as EarthlyBranch },
  { code: 1008, order: 9, branch: "戌" as EarthlyBranch },
  { code: 1107, order: 10, branch: "亥" as EarthlyBranch },
  { code: 1207, order: 11, branch: "子" as EarthlyBranch }
] as const;

const GENERATES: Record<FiveElement, FiveElement> = {
  木: "火",
  火: "土",
  土: "金",
  金: "水",
  水: "木"
};

const CONTROLS: Record<FiveElement, FiveElement> = {
  木: "土",
  火: "金",
  土: "水",
  金: "木",
  水: "火"
};

function mod(value: number, divisor: number) {
  return ((value % divisor) + divisor) % divisor;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function parseBirthDate(value: string) {
  const [year, month, day] = value.split("-").map((item) => Number(item));
  return { year, month, day };
}

function parseBirthTime(value: string) {
  if (!value) return { hour: 12, minute: 0 };
  const [hour, minute] = value.split(":").map((item) => Number(item));
  return {
    hour: Number.isFinite(hour) ? hour : 12,
    minute: Number.isFinite(minute) ? minute : 0
  };
}

function solarYearForBazi(year: number, month: number, day: number) {
  if (month < 2) return year - 1;
  if (month === 2 && day < 4) return year - 1;
  return year;
}

function getMonthInfo(month: number, day: number) {
  const code = month * 100 + day;
  let current = { order: 11, branch: "子" as EarthlyBranch };

  for (const item of MONTH_STARTS) {
    if (code >= item.code) {
      current = { order: item.order, branch: item.branch };
    }
  }

  return current;
}

function getYearPillar(year: number, month: number, day: number): OraclePillar {
  const adjustedYear = solarYearForBazi(year, month, day);
  const offset = adjustedYear - 1984;
  const stem = STEMS[mod(offset, 10)];
  const branch = BRANCHES[mod(offset, 12)];

  return {
    label: "年柱",
    stem,
    branch,
    stemElement: STEM_ELEMENTS[stem],
    branchElement: BRANCH_ELEMENTS[branch]
  };
}

function getMonthPillar(yearStem: HeavenlyStem, month: number, day: number): OraclePillar {
  const monthInfo = getMonthInfo(month, day);
  const firstStemMap: Record<HeavenlyStem, HeavenlyStem> = {
    甲: "丙",
    己: "丙",
    乙: "戊",
    庚: "戊",
    丙: "庚",
    辛: "庚",
    丁: "壬",
    壬: "壬",
    戊: "甲",
    癸: "甲"
  };
  const firstStem = firstStemMap[yearStem];
  const stem = STEMS[mod(STEMS.indexOf(firstStem) + monthInfo.order - 1, 10)];

  return {
    label: "月柱",
    stem,
    branch: monthInfo.branch,
    stemElement: STEM_ELEMENTS[stem],
    branchElement: BRANCH_ELEMENTS[monthInfo.branch]
  };
}

function getDayPillar(year: number, month: number, day: number): OraclePillar & { cycleIndex: number } {
  const baseDate = Date.UTC(1984, 1, 2);
  const targetDate = Date.UTC(year, month - 1, day);
  const diffDays = Math.round((targetDate - baseDate) / 86400000);
  const cycleIndex = mod(diffDays, 60);
  const stem = STEMS[mod(cycleIndex, 10)];
  const branch = BRANCHES[mod(cycleIndex, 12)];

  return {
    label: "日柱",
    stem,
    branch,
    stemElement: STEM_ELEMENTS[stem],
    branchElement: BRANCH_ELEMENTS[branch],
    cycleIndex
  };
}

function getHourBranch(hour: number, minute: number): EarthlyBranch {
  const timeValue = hour + minute / 60;
  if (timeValue >= 23 || timeValue < 1) return "子";
  return BRANCHES[Math.floor((timeValue + 1) / 2)];
}

function getHourPillar(dayStem: HeavenlyStem, hour: number, minute: number): OraclePillar {
  const branch = getHourBranch(hour, minute);
  const childStemMap: Record<HeavenlyStem, HeavenlyStem> = {
    甲: "甲",
    己: "甲",
    乙: "丙",
    庚: "丙",
    丙: "戊",
    辛: "戊",
    丁: "庚",
    壬: "庚",
    戊: "壬",
    癸: "壬"
  };
  const baseStem = childStemMap[dayStem];
  const stem = STEMS[mod(STEMS.indexOf(baseStem) + BRANCHES.indexOf(branch), 10)];

  return {
    label: "时柱",
    stem,
    branch,
    stemElement: STEM_ELEMENTS[stem],
    branchElement: BRANCH_ELEMENTS[branch]
  };
}

function elementSummary(element: FiveElement, count: number, dominantElement: FiveElement, missingElement: FiveElement) {
  if (element === dominantElement) {
    return `${element}偏强，代表你在这类能量上出手更自然，也更容易过量。`;
  }

  if (element === missingElement) {
    return `${element}偏弱，不是没有，而是需要靠环境和习惯刻意补位。`;
  }

  return `${element}处在可调区间，平时更像辅助位，关键时刻决定整体顺不顺。`;
}

function relationOf(dayElement: FiveElement, otherElement: FiveElement): EnergyRelation {
  if (dayElement === otherElement) return "peer";
  if (GENERATES[dayElement] === otherElement) return "output";
  if (CONTROLS[dayElement] === otherElement) return "wealth";
  if (CONTROLS[otherElement] === dayElement) return "authority";
  return "resource";
}

function buildRhythmLabel(primary: FiveElement, secondary: FiveElement) {
  return RHYTHM_LABELS[`${primary}${secondary}`] ?? `${primary}${secondary}混合型`;
}

function buildConfidence(precision: BirthPrecision) {
  switch (precision) {
    case "exact":
      return {
        label: "高参考",
        note: "你提供了较明确的出生时刻，时柱和紫微镜像部分可读性更高。"
      };
    case "approx":
      return {
        label: "中参考",
        note: "出生时刻是大概值，年、月、日的判断更稳，时柱和宫位请当作区间参考。"
      };
    default:
      return {
        label: "谨慎使用",
        note: "出生时刻不确定时，紫微镜像和节奏推断会更偏启发式，不适合拿来做绝对结论。"
      };
  }
}

function buildYearlyTheme(
  dayElement: FiveElement,
  focus: OracleFocus,
  phase: OraclePhase
) {
  const today = new Date();
  const currentYearPillar = getYearPillar(today.getFullYear(), today.getMonth() + 1, today.getDate());
  const relation = relationOf(dayElement, currentYearPillar.stemElement);

  const titleMap: Record<EnergyRelation, string> = {
    peer: "同类年份，容易把压力拉回自我要求",
    output: "表达年份，适合把东西做出来给世界看",
    wealth: "资源年份，更适合算账、分配和调仓",
    authority: "责任年份，规则、角色和边界都会变重",
    resource: "修复年份，先把根系养稳比强冲更值"
  };

  const adviceMap: Record<EnergyRelation, Record<OracleFocus, string>> = {
    peer: {
      career: "事业上别只靠硬扛，多把竞争感翻译成更清楚的协作边界。",
      wealth: "财务上要避免凭状态加大赌性，先设规则再出手。",
      relationship: "关系里最怕的是把“我以为”当作“你应该”。",
      transition: "换挡时先区分真想转，还是只是对现状厌倦。",
      growth: "成长上别只卷效率，先看自我要求有没有过度上升。"
    },
    output: {
      career: "事业上更适合公开表达、做作品和主动建立影响力。",
      wealth: "财务上适合把经验、技能或信息差转成可兑现的结果。",
      relationship: "关系里先表达真实需求，比一味懂事更重要。",
      transition: "换挡期不要只想，要把新方向做成看得见的样本。",
      growth: "成长上多做输出式学习，写下来、讲出来会比闷着更有效。"
    },
    wealth: {
      career: "事业上更适合谈资源、定预算和优化投入产出比。",
      wealth: "财务上重点不是赚快钱，而是建立更能长期复用的分配策略。",
      relationship: "关系里要看谁在持续给，谁在稳定拿，别让账失衡太久。",
      transition: "换挡前先备足资源缓冲，别让理想败给现金流。",
      growth: "成长上给自己买时间、买环境、买更高质量的输入。"
    },
    authority: {
      career: "事业上规则感会变重，适合补流程、补标准、补承诺兑现。",
      wealth: "财务上重视合规和风险边界，别让侥幸心吃掉积累。",
      relationship: "关系里需要更清楚的承诺，而不是无限开放式试探。",
      transition: "换挡期要提前处理现实约束，别把阻力都推到最后。",
      growth: "成长上最值得练的是纪律，不是灵感。"
    },
    resource: {
      career: "事业上先补认知、补方法、补底层能力，再谈冲刺。",
      wealth: "财务上以稳为主，先修复结构，再追求扩张。",
      relationship: "关系里适合先养连接质量，不用急着做大动作。",
      transition: "换挡时多给自己留观测期，不必每一步都立刻定性。",
      growth: "成长上重点是恢复、沉淀和建立长期可持续的节律。"
    }
  };

  const phaseTail: Record<OraclePhase, string> = {
    steady: "当前处在稳定经营期，更适合小步快跑、连续校准。",
    pressure: "现在的高压感会放大优缺点，越是这样越要靠结构而不是情绪。",
    reset: "既然在重整，就别急着证明自己，先把顺序排对。",
    expansion: "既然准备放大，就提前设计收束点，别只顾打开不顾回收。"
  };

  return {
    pillar: `${currentYearPillar.stem}${currentYearPillar.branch}`,
    relation,
    relationLabel: RELATION_LABELS[relation],
    title: titleMap[relation],
    advice: `${adviceMap[relation][focus]} ${phaseTail[phase]}`
  };
}

function palaceNote(name: string, focus: OracleFocus) {
  const focusTail: Record<OracleFocus, string> = {
    career: "把它翻译到事业上，就是别只看岗位，更要看你在哪种角色里最能持续发力。",
    wealth: "把它翻译到资源上，就是你对安全感和调度权的需求会更直接。",
    relationship: "把它翻译到关系里，就是你会更在意谁能和你一起稳定承压。",
    transition: "把它翻译到换挡里，就是环境一变，你的状态也会跟着明显变。",
    growth: "把它翻译到成长上，就是你要先看自己真正依赖什么来恢复心力。"
  };
  const target = PALACES.find((item) => item.name === name);
  return `${target?.note ?? ""} ${focusTail[focus]}`;
}

function buildDomainScores(
  counts: Record<FiveElement, number>,
  focus: OracleFocus,
  phase: OraclePhase,
  palaceIndex: number,
  bodyPalaceIndex: number,
  precision: BirthPrecision
) {
  const blueprints = [
    {
      name: "事业推进",
      focusKey: "career" as OracleFocus,
      weights: { 木: 8, 火: 10, 土: 9, 金: 12, 水: 6 },
      palaceBoost: [0, 8]
    },
    {
      name: "资源调度",
      focusKey: "wealth" as OracleFocus,
      weights: { 木: 5, 火: 6, 土: 12, 金: 11, 水: 7 },
      palaceBoost: [4, 9]
    },
    {
      name: "关系温度",
      focusKey: "relationship" as OracleFocus,
      weights: { 木: 8, 火: 11, 土: 7, 金: 5, 水: 9 },
      palaceBoost: [2, 7]
    },
    {
      name: "换挡开拓",
      focusKey: "transition" as OracleFocus,
      weights: { 木: 10, 火: 9, 土: 5, 金: 6, 水: 10 },
      palaceBoost: [6, 11]
    },
    {
      name: "恢复蓄能",
      focusKey: "growth" as OracleFocus,
      weights: { 木: 5, 火: 4, 土: 9, 金: 6, 水: 12 },
      palaceBoost: [5, 10]
    },
    {
      name: "表达呈现",
      focusKey: "career" as OracleFocus,
      weights: { 木: 9, 火: 12, 土: 6, 金: 7, 水: 5 },
      palaceBoost: [3, 6]
    }
  ];

  const phaseBonusMap: Record<OraclePhase, Record<string, number>> = {
    steady: {
      事业推进: 2,
      资源调度: 3,
      关系温度: 2,
      换挡开拓: -1,
      恢复蓄能: 3,
      表达呈现: 1
    },
    pressure: {
      事业推进: 1,
      资源调度: 4,
      关系温度: -1,
      换挡开拓: -2,
      恢复蓄能: 5,
      表达呈现: -1
    },
    reset: {
      事业推进: -2,
      资源调度: 2,
      关系温度: 1,
      换挡开拓: 1,
      恢复蓄能: 6,
      表达呈现: 0
    },
    expansion: {
      事业推进: 5,
      资源调度: 2,
      关系温度: 2,
      换挡开拓: 5,
      恢复蓄能: -2,
      表达呈现: 4
    }
  };

  return blueprints.map((item) => {
    const raw = ELEMENTS.reduce((sum, element) => sum + counts[element] * item.weights[element], 0);
    const focusBonus = focus === item.focusKey ? 8 : 0;
    const palaceBonus =
      item.palaceBoost.includes(palaceIndex) || item.palaceBoost.includes(bodyPalaceIndex) ? 6 : 0;
    const precisionPenalty = precision === "unknown" ? -4 : precision === "approx" ? -2 : 0;
    const score = clamp(
      Math.round(20 + raw / 1.6 + focusBonus + palaceBonus + phaseBonusMap[phase][item.name] + precisionPenalty),
      38,
      94
    );
    const tone: "positive" | "watch" | "calm" =
      score >= 78 ? "positive" : score < 60 ? "watch" : "calm";
    const note =
      score >= 78
        ? "这是当前更容易形成正反馈的面，适合主动发力。"
        : score < 60
          ? "这里不是短板定论，而是更需要刻意设计环境和节奏。"
          : "这块属于可稳步经营区，方法对了会持续变好。";

    return {
      name: item.name,
      score,
      tone,
      note
    };
  });
}

function buildActionExperiments(
  focus: OracleFocus,
  phase: OraclePhase,
  missingElement: FiveElement
) {
  const focusAction: Record<OracleFocus, string> = {
    career: "把本周最重要的推进事项压缩成一个可交付物，而不是五个模糊方向。",
    wealth: "给自己的钱做三层分区：安全垫、增长仓、可试错仓，别都混在一起。",
    relationship: "找一段关键关系，先讲真实需求，再讲你希望对方如何配合。",
    transition: "把想换的方向先做成一个低成本样本，别一上来就全盘重启。",
    growth: "选一个你总说要补的能力，给它排一个连续七天的固定窗口。"
  };

  const phaseAction: Record<OraclePhase, string> = {
    steady: "现在不需要过度求变，把小步快跑和固定复盘坚持住，效果会比大动作更好。",
    pressure: "高压期先砍掉一件并非当下必须做好的事，给核心事项留呼吸空间。",
    reset: "重整期先清理旧承诺和旧待办，再谈新目标，不然只是换了个地方继续堆积。",
    expansion: "扩张期提前设一个收束点，比如时间上限、预算上限或合作边界。"
  };

  const missingAction: Record<FiveElement, string> = {
    木: "木偏弱时最需要的是长线规划，把接下来三个月只保留一个主方向。",
    火: "火偏弱时别等状态，先用更公开的承诺和更小的启动动作点火。",
    土: "土偏弱时优先补稳定作息、固定空间和每周一次结构化整理。",
    金: "金偏弱时要练切分和舍弃，先明确什么不做，再谈做什么。",
    水: "水偏弱时要给自己留输入和恢复窗口，不要长期只输出不回补。"
  };

  return [focusAction[focus], phaseAction[phase], missingAction[missingElement]];
}

function buildRealityChecks(
  topRelation: EnergyRelation,
  palace: string,
  yearlyThemeTitle: string
) {
  const relationChecks: Record<EnergyRelation, string> = {
    peer: "过去三个月里，你是否经常把事情揽回自己身上，嘴上说没事，身体却明显变紧？",
    output: "最近你是不是一想到“要给别人看”就更容易兴奋，也更容易紧张？",
    wealth: "最近你做决定时，是不是越来越看重值不值、划不划算、能不能形成复利？",
    authority: "最近有没有明显感觉到规则、责任和现实约束比以前更重？",
    resource: "你是不是越来越需要独处、学习或恢复性空间，才能重新有判断力？"
  };

  const palaceCheck =
    palace === "官禄宫"
      ? "你最近的情绪起伏，是否和事业角色、外部评价或产出节奏明显绑定？"
      : palace === "财帛宫"
        ? "你最近的安全感是否更容易被现金流、储蓄和资源掌控感牵动？"
        : palace === "福德宫"
          ? "你最近的问题，有多少其实不是能力问题，而是恢复质量下降？"
          : `如果把最近最纠结的问题单独拿出来看，它是否确实和“${palace.replace("宫", "")}”主题有关？`;

  return [relationChecks[topRelation], palaceCheck, `如果今年的主调是“${yearlyThemeTitle}”，你过去 30 天的实际行动有没有对应上？`];
}

function buildCalibrationQuestions(
  precision: BirthPrecision,
  question: string,
  supportingElement: FiveElement
) {
  const timingNote =
    precision === "exact"
      ? "如果你提供的出生时刻比较准，优先看宫位和行动实验是否贴合。"
      : "出生时刻不够准时，不要过度执着于某一宫位，更该看大方向是否成立。";
  const questionNote = question.trim()
    ? `你带着“${question.trim()}”来测，这份结果最该验证的是：它有没有帮助你看清真正的问题，而不是只让你觉得被理解。`
    : "如果一条解读只让你觉得“像”，却没有引出新的观察和动作，那它的价值其实有限。";

  return [
    timingNote,
    questionNote,
    `${supportingElement}是你的辅助能量，最近你有没有在它对应的方式里得到明显恢复或助推？`
  ];
}

export function buildOracleReading(draft: OracleDraft): OracleResult {
  const { year, month, day } = parseBirthDate(draft.birthDate);
  const { hour, minute } = parseBirthTime(draft.birthTime);
  const confidence = buildConfidence(draft.precision);
  const yearPillar = getYearPillar(year, month, day);
  const monthPillar = getMonthPillar(yearPillar.stem, month, day);
  const dayPillar = getDayPillar(year, month, day);
  const hourPillar = getHourPillar(dayPillar.stem, hour, minute);
  const pillars: OraclePillar[] = [yearPillar, monthPillar, dayPillar, hourPillar];
  const dayMasterElement = STEM_ELEMENTS[dayPillar.stem];

  const counts = pillars.reduce<Record<FiveElement, number>>(
    (current, pillar) => {
      current[pillar.stemElement] += 1;
      current[pillar.branchElement] += 1;
      return current;
    },
    {
      木: 0,
      火: 0,
      土: 0,
      金: 0,
      水: 0
    }
  );

  const sortedElements = [...ELEMENTS]
    .map((element) => ({
      element,
      count: counts[element]
    }))
    .sort((a, b) => b.count - a.count || ELEMENTS.indexOf(a.element) - ELEMENTS.indexOf(b.element));
  const dominantElement = sortedElements[0].element;
  const supportingElement = sortedElements[1].element;
  const missingElement = sortedElements[sortedElements.length - 1].element;

  const elementBalance = ELEMENTS.map((element) => {
    const tone: "positive" | "watch" | "calm" =
      element === dominantElement
        ? "positive"
        : element === missingElement || counts[element] === 0
          ? "watch"
          : "calm";

    return {
      element,
      count: counts[element],
      ratio: Number((counts[element] / 8).toFixed(2)),
      tone,
      summary: elementSummary(element, counts[element], dominantElement, missingElement)
    };
  });

  const relationBuckets: Record<EnergyRelation, number> = {
    peer: 0,
    output: 0,
    wealth: 0,
    authority: 0,
    resource: 0
  };
  const comparisonElements: FiveElement[] = [
    yearPillar.stemElement,
    yearPillar.branchElement,
    monthPillar.stemElement,
    monthPillar.branchElement,
    dayPillar.branchElement,
    hourPillar.stemElement,
    hourPillar.branchElement
  ];

  for (const item of comparisonElements) {
    relationBuckets[relationOf(dayMasterElement, item)] += 1;
  }

  const relationMix = (Object.keys(relationBuckets) as EnergyRelation[])
    .map((relation) => ({
      relation,
      label: RELATION_LABELS[relation],
      count: relationBuckets[relation]
    }))
    .sort((a, b) => b.count - a.count);

  const monthInfo = getMonthInfo(month, day);
  const hourBranch = getHourBranch(hour, minute);
  const palaceIndex = mod(monthInfo.order * 2 + BRANCHES.indexOf(hourBranch) + day, PALACES.length);
  const bodyPalaceIndex = mod(palaceIndex + dayPillar.cycleIndex + BRANCHES.indexOf(hourBranch), PALACES.length);
  const palace = PALACES[palaceIndex];
  const bodyPalace = PALACES[bodyPalaceIndex];
  const yearlyTheme = buildYearlyTheme(dayMasterElement, draft.focus, draft.phase);

  const starPrimaryIndex = mod(dayPillar.cycleIndex + monthInfo.order + BRANCHES.indexOf(hourBranch), STAR_ARCHETYPES.length);
  let starSecondaryIndex = mod(
    STEMS.indexOf(dayPillar.stem) * 3 + month + day + BRANCHES.indexOf(hourBranch),
    STAR_ARCHETYPES.length
  );
  if (starSecondaryIndex === starPrimaryIndex) {
    starSecondaryIndex = mod(starSecondaryIndex + 5, STAR_ARCHETYPES.length);
  }
  const stars = [STAR_ARCHETYPES[starPrimaryIndex], STAR_ARCHETYPES[starSecondaryIndex]];

  const domainScores = buildDomainScores(
    counts,
    draft.focus,
    draft.phase,
    palaceIndex,
    bodyPalaceIndex,
    draft.precision
  ).sort((a, b) => b.score - a.score);

  const topRelation = relationMix[0]?.relation ?? "resource";
  const focusPrefix: Record<OracleFocus, string> = {
    career: "这次你更关心事业推进，所以结果会更偏向角色定位、节奏与交付方式。",
    wealth: "这次你更关心资源与财务，所以会更强调安全感、调度权与结构选择。",
    relationship: "这次你更关心关系议题，所以重点会放在连接方式、边界和稳定承压能力。",
    transition: "这次你更关心换挡，所以重点会放在环境变化、时机和试错成本。",
    growth: "这次你更关心成长，所以会更多看恢复方式、长期能力和内在秩序。"
  };
  const phasePrefix: Record<OraclePhase, string> = {
    steady: "当前是稳定经营期，适合做细调和持续经营。",
    pressure: "当前压力比较高，越是这样越要靠结构化方法保持清醒。",
    reset: "当前处在重整阶段，不必急于证明，只要把顺序校准。",
    expansion: "当前更像扩张期，适合放大机会，但要留出收束机制。"
  };
  const questionLine = draft.question.trim()
    ? `你带着“${draft.question.trim()}”来看这张盘，所以更值得看的是哪些建议能直接转成动作。`
    : "这份结果更适合拿来做观察和校准，而不是给自己贴死标签。";
  const focusSummary = `${focusPrefix[draft.focus]} ${phasePrefix[draft.phase]} ${questionLine}`;

  const sections: AnalysisSection[] = [
    {
      title: "八字底色",
      items: [
        `${dayPillar.stem}日主是你的核心出手方式，对应的底色更接近“${DAY_MASTER_TITLES[dayPillar.stem]}”。`,
        `${buildRhythmLabel(dominantElement, supportingElement)}是当前盘面最明显的节奏组合，说明你处理问题时会优先走这类路径。`,
        `${RELATION_LABELS[topRelation]}占比更高，意味着你在现实里最容易被这类议题拉动。`
      ]
    },
    {
      title: "紫微斗数镜像",
      items: [
        `命宫落在${palace.name}，说明你的人生叙事很容易被这类主题放大。`,
        `身宫落在${bodyPalace.name}，通常代表真正进入行动状态时，你会更像这里描述的样子。`,
        `主叙事星偏向${stars[0].name}与${stars[1].name}，一个管主轴，一个管转弯时的姿态。`
      ]
    },
    {
      title: "如何使用这份结果",
      items: [
        yearlyTheme.advice,
        "先看现实校验，再看行动实验，最后决定哪些内容值得长期采用。",
        "如果一段解读只能带来情绪安慰，却不能带来新观察或新动作，它就不够好。"
      ]
    }
  ];

  return {
    chartTitle: `${buildRhythmLabel(dominantElement, supportingElement)} · ${DAY_MASTER_TITLES[dayPillar.stem]}`,
    confidenceLabel: confidence.label,
    confidenceNote: confidence.note,
    focusLabel: FOCUS_LABELS[draft.focus],
    phaseLabel: PHASE_LABELS[draft.phase],
    focusSummary,
    dayMaster: {
      stem: dayPillar.stem,
      element: dayMasterElement,
      title: DAY_MASTER_TITLES[dayPillar.stem],
      brief: DAY_MASTER_BRIEFS[dayPillar.stem]
    },
    pillars,
    elementBalance,
    dominantElement,
    supportingElement,
    missingElement,
    rhythmLabel: buildRhythmLabel(dominantElement, supportingElement),
    relationMix,
    yearlyTheme,
    palace: {
      name: palace.name,
      note: palaceNote(palace.name, draft.focus)
    },
    bodyPalace: {
      name: bodyPalace.name,
      note: palaceNote(bodyPalace.name, draft.focus)
    },
    stars: [...stars],
    domainScores,
    sections,
    realityChecks: buildRealityChecks(topRelation, palace.name, yearlyTheme.title),
    actionExperiments: buildActionExperiments(draft.focus, draft.phase, missingElement),
    calibrationQuestions: buildCalibrationQuestions(draft.precision, draft.question, supportingElement)
  };
}
