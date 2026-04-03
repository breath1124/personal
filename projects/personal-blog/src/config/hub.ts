export type Channel = {
  slug: "blog" | "vibe";
  eyebrow: string;
  title: string;
  description: string;
  href: string;
  ctaLabel: string;
  highlights: string[];
};

export type LabApp = {
  slug: string;
  title: string;
  category: string;
  status: "live" | "beta" | "planned";
  tagline: string;
  description: string;
  href: string;
  estimatedTime: string;
  highlights: string[];
  inputs: string[];
  useCases: string[];
  disclaimer?: string;
};

export const HOME_CHANNELS: Channel[] = [
  {
    slug: "blog",
    eyebrow: "长期内容沉淀",
    title: "博客",
    description:
      "继续保留现在的技术博客区，用来记录论文阅读、工程实践、产品思考和长期判断。",
    href: "/blog/",
    ctaLabel: "进入博客",
    highlights: ["文章归档", "标签/分类浏览", "适合持续更新"]
  },
  {
    slug: "vibe",
    eyebrow: "可直接使用的实验项目",
    title: "Vibe Coding 实验室",
    description:
      "把一些有意思的想法做成交互界面，先从 MBTI 测试助手和基金买入卖出助手开始，后续会继续扩展更多可玩的页面。",
    href: "/vibe/",
    ctaLabel: "进入实验室",
    highlights: ["小而实用", "页面级扩展", "未来可持续新增"]
  }
];

export const LAB_OVERVIEW = {
  title: "Vibe Coding 实验室",
  description:
    "这里放的是可以直接打开、输入信息、获得反馈的界面型项目。它们不是文章附属页，而是站点的第二条主入口。",
  principles: [
    "每个项目都应该能单独访问和独立扩展。",
    "项目卡片和详情页由同一份注册数据驱动，减少重复维护。",
    "后续新增项目时，只需要补一个组件并在这里注册。"
  ]
} as const;

export const LAB_APPS: LabApp[] = [
  {
    slug: "mbti-assistant",
    title: "MBTI 测试助手",
    category: "人格探索",
    status: "live",
    tagline: "用一组更适合中文语境的问题，快速判断你的 MBTI 倾向。",
    description:
      "通过 12 道题聚合四个维度的偏好分数，给出类型、维度解释和针对工作沟通的建议。",
    href: "/vibe/mbti-assistant/",
    estimatedTime: "约 3-5 分钟",
    highlights: ["12 道直觉题", "即时类型结果", "沟通/协作建议"],
    inputs: ["题目选择", "关注场景", "当前状态自评"],
    useCases: [
      "想快速了解自己的 MBTI 倾向",
      "做团队协作前的沟通风格梳理",
      "需要一个更轻量的自测入口"
    ]
  },
  {
    slug: "fund-buy-sell-assistant",
    title: "基金买入卖出助手",
    category: "投资决策整理",
    status: "live",
    tagline: "把仓位、估值、盈亏和流动性需求放到同一个决策框架里。",
    description:
      "输入你的仓位、目标配置、估值判断和现金需求后，得到偏买入、偏持有或偏减仓的结构化建议。",
    href: "/vibe/fund-buy-sell-assistant/",
    estimatedTime: "约 2 分钟",
    highlights: ["仓位诊断", "节奏建议", "风险提醒"],
    inputs: ["仓位/目标仓位", "盈亏幅度", "估值/流动性判断"],
    useCases: [
      "不知道现在该补仓还是继续等",
      "账户盈利后想判断是否需要止盈",
      "需要先整理思路再决定动作"
    ],
    disclaimer:
      "该工具仅用于帮助梳理决策因素，不构成任何投资建议，也不会接入实时行情。"
  }
];

export function getLabApp(slug: string) {
  return LAB_APPS.find((app) => app.slug === slug);
}
