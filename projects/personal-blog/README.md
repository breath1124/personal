# personal-blog（Astro）

一个面向 GitHub Pages 的静态个人网站，当前包含两条主入口：

- 博客：长期内容沉淀
- Vibe Coding 实验室：可直接打开使用的小工具和界面项目

## 功能

- 文章：列表页、详情页（Content Collections）
- 门户首页：博客 / 实验室双入口
- 实验室：`/vibe/` 项目列表页
- 工具页：`/vibe/<slug>/`，当前包含 MBTI 测试助手、基金买入卖出助手
- 标签：`/tags/`、`/tags/<tag>/`
- 分类：`/categories/`、`/categories/<category>/`
- 搜索：`/search/`（构建时生成 `search.json`，前端本地检索）
- 评论：Giscus（可选）
- 统计：Plausible / Umami / Google Analytics（可选）

## 开发

```bash
cd projects/personal-blog
npm install
npm run dev
```

## 构建与预览

```bash
cd projects/personal-blog
npm run build
npm run preview
```

## 写文章

- 新增文章：`src/content/blog/*.md`
- 文章字段定义：`src/content/config.ts`
- `tags`：标签数组（可选）
- `category` 或 `categories`：分类（可选）

## 开启评论（Giscus）

1. 按 `https://giscus.app/` 的指引，为你的仓库启用 Giscus。
2. 编辑 `src/config/integrations.ts`：将 `GISCUS.enabled` 设为 `true`，并填入 `repo` / `repoId` / `categoryId` 等配置。

## 开启统计（可选）

编辑 `src/config/integrations.ts` 的 `ANALYTICS`：

- `provider: "plausible"`：填写 `plausibleDomain`
- `provider: "umami"`：填写 `umamiScriptSrc` 与 `umamiWebsiteId`
- `provider: "google"`：填写 `googleMeasurementId`

## 部署到 GitHub Pages

1. 将本仓库推到 GitHub（默认分支 `main`）。
2. GitHub 仓库设置：`Settings → Pages → Build and deployment → Source` 选择 `GitHub Actions`。
3. 推送到 `main` 后会触发工作流：`.github/workflows/deploy-personal-blog.yml`。

## 自定义站点信息

编辑：`src/config/site.ts`

## 新增一个实验室项目

当前实验室区域按“注册表 + 独立组件”的方式组织，便于后续继续扩展界面型项目。

1. 在 `src/config/hub.ts` 的 `LAB_APPS` 里注册新项目元数据。
2. 在 `src/components/vibe/` 下新增对应的 Astro 组件。
3. 在 `src/pages/vibe/[slug].astro` 中把 `slug` 映射到新组件。

完成后，新项目会自动出现在：

- 首页实验室精选
- `/vibe/` 项目列表页
- 对应详情路由

## 自定义域名（可选）

若你为 Pages 绑定了自定义域名，建议在构建时覆盖站点地址（用于 `sitemap.xml` / `rss.xml` 等生成正确的绝对链接）：

- 推荐在 GitHub 仓库里设置 Actions 变量（无需改代码）：
  - `Settings → Secrets and variables → Actions → Variables`
  - 新增 `SITE`：例如 `https://example.com`
  - 新增 `BASE_PATH`：自定义域名通常用根路径，设为 `/`
