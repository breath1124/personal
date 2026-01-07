export const SITE = {
  title: "个人博客",
  description: "记录技术、产品与生活。",
  author: "Your Name",
  locale: "zh-CN"
} as const;

export const NAV = [
  { label: "首页", href: "/" },
  { label: "博客", href: "/blog/" },
  { label: "标签", href: "/tags/" },
  { label: "分类", href: "/categories/" },
  { label: "搜索", href: "/search/" },
  { label: "关于", href: "/about/" }
] as const;
