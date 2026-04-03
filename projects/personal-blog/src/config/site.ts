export const SITE = {
  title: "个人网站",
  description: "技术写作与可直接使用的交互实验。",
  author: "Your Name",
  locale: "zh-CN"
} as const;

export const NAV = [
  { label: "首页", href: "/" },
  { label: "博客", href: "/blog/" },
  { label: "实验室", href: "/lab/" }
] as const;
