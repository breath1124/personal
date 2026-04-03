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
    eyebrow: "Writing",
    title: "博客",
    description:
      "保留长期写作，记录论文阅读、工程实践、产品思考和更慢一些的判断。",
    href: "/blog/",
    ctaLabel: "进入博客",
    highlights: ["长文", "归档", "持续更新"]
  },
  {
    slug: "vibe",
    eyebrow: "Lab",
    title: "实验室",
    description:
      "把一些想法做成可以直接打开使用的界面，而不是只停留在文章或截图里。",
    href: "/lab/",
    ctaLabel: "进入 Lab",
    highlights: ["可直接使用", "数据驱动", "AI 辅助"]
  }
];

export const LAB_OVERVIEW = {
  title: "实验室",
  description:
    "这里放的是可以直接打开和使用的交互项目。",
  principles: [
    "每个项目都应该能单独访问和独立扩展。",
    "项目卡片和详情页由同一份注册数据驱动，减少重复维护。",
    "新增项目时，只需要补一个组件并在这里注册。"
  ]
} as const;

export const LAB_APPS: LabApp[] = [
  {
    slug: "mbti-assistant",
    title: "MBTI 测试助手",
    category: "人格探索",
    status: "live",
    tagline: "更完整的题组、维度结果和针对工作沟通的结果报告。",
    description:
      "完成更完整的 MBTI 自评后，可以看到类型、维度信心、场景化建议，以及可选的 AI 解读。",
    href: "/lab/mbti/",
    estimatedTime: "约 8 分钟",
    highlights: ["24 题测评", "结果报告", "AI 解读"],
    inputs: ["题目回答", "场景选择", "当前关注问题"],
    useCases: [
      "重新梳理自己的工作风格",
      "做团队协作前的沟通风格复盘",
      "需要一份更像报告而不是小测试的结果"
    ]
  },
  {
    slug: "fund-buy-sell-assistant",
    title: "基金买入卖出助手",
    category: "投资决策整理",
    status: "live",
    tagline: "录入真实持仓、同步基金画像、公告信号和 AI 研判。",
    description:
      "录入持仓后，应用会拉基金净值、基金经理、规模、资产配置和重仓股公告，再结合你的问题生成分析。",
    href: "/lab/fund/",
    estimatedTime: "约 5 分钟",
    highlights: ["持仓管理", "公告信号", "AI 研判"],
    inputs: ["基金代码", "持仓份额/成本", "买入逻辑与问题"],
    useCases: [
      "把零散持仓整理成一个清楚的工作台",
      "在继续拿、分批加仓和控制仓位之间做判断",
      "把基金重仓股最近公告纳入决策视野"
    ],
    disclaimer:
      "该工具仅用于帮助梳理决策因素，不构成任何投资建议，也不会接入实时行情。"
  }
];

export function getLabApp(slug: string) {
  return LAB_APPS.find((app) => app.slug === slug);
}
