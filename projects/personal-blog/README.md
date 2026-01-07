# personal-blog（Astro）

一个面向 GitHub Pages 的静态个人博客站点骨架（SEO：`sitemap.xml` / `robots.txt` / `rss.xml`）。

## 功能

- 文章：列表页、详情页（Content Collections）
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

## 自定义域名（可选）

若你为 Pages 绑定了自定义域名，建议在构建时覆盖站点地址：

- 在工作流 `Build` 步骤中设置环境变量 `SITE`（例如 `https://example.com`）
- 如需要根路径部署，可同时设置 `BASE_PATH=/`
