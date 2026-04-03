# vibe-lab

独立的 `Vibe Coding` 应用区，面向个人网站中的可交互项目。

当前包含：

- 基金助手：持仓录入、基金数据拉取、规则诊断、AI 研判
- MBTI 助手：完整测评、结果报告、AI 解读
- 命理实验室：八字实验盘、紫微镜像、现实校验与行动建议
- 反应力实验室：三段反应测试、逐轮数据和综合分析

AI 相关约定：

- 三个分析型应用共用一套 OpenAI 兼容模型设置
- 每个应用内部再提供各自的 AI skill 选择器，用不同的 system prompt 和输出结构处理不同任务
- 命理实验室的 AI 解读会明确约束为启发式表达，不以宿命论方式输出

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
