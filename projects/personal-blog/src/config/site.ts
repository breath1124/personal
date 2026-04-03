export const SITE = {
  title: "个人网站",
  description: "博客、Vibe Coding 实验与可直接使用的小工具。",
  author: "Your Name",
  locale: "zh-CN"
} as const;

export const NAV = [
  { label: "首页", href: "/" },
  { label: "博客", href: "/blog/" },
  { label: "Vibe Coding", href: "/vibe/" }
] as const;
