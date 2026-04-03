# vibe-lab

独立的 `Vibe Coding` 应用区，面向个人网站中的可交互项目。

当前包含：

- 基金助手：持仓录入、基金数据拉取、规则诊断、AI 研判
- MBTI 助手：完整测评、结果报告、AI 解读

## 开发

```bash
cd projects/vibe-lab
npm install
npm run dev
```

## 构建

```bash
cd projects/vibe-lab
npm run build
```

构建前会自动生成一份市场简报数据，输出到 `src/generated/market-brief.json`。

## AI 设置

应用默认按 OpenAI 兼容接口工作，用户可在设置页自行输入：

- Base URL
- Model
- API Key

这些配置只保存在浏览器本地，不会提交到仓库。

## 部署

当前计划由仓库中的 GitHub Actions 统一构建，并将静态产物挂载到主站的 `/lab/` 路径下。
