---
title: "OpenCode 入门与实战：开源 AI 编码代理的配置、工作流与最佳实践"
description: "基于 opencode.ai 官方文档整理：从安装、连接模型供应商、/init 生成 AGENTS.md，到权限控制、配置层级、扩展（命令/Agent/工具/MCP/插件）、分享与 IDE 集成，给出一份偏工程实践的中文科普与上手指南。"
pubDate: 2026-01-14
tags: ["工具教程", "AI编程", "OpenCode", "编码代理", "LLM", "CLI", "TUI"]
category: "工具"
draft: false
---

> 官网：<https://opencode.ai/>  
> 官方文档：<https://opencode.ai/docs/>  
> 开源仓库（文档指向）：<https://github.com/anomalyco/opencode>

OpenCode（opencode.ai）是一款**开源 AI 编码代理（AI coding agent）**：你可以把它当作“能读代码、能跑命令、能改文件、还能回滚”的智能结对工程师。

它的形态不止一种：**终端 TUI**、**桌面应用**、**IDE 扩展**都支持；日常最常用的是在项目目录里直接跑 `opencode` 进入交互界面。

这篇文章以“科普 + 干货”为主，尽量把关键概念讲清楚，同时给出你能直接复制使用的命令与配置示例（均来自官方文档与其配置 schema）。

## 0. OpenCode 解决的到底是什么问题？

很多人第一次用 coding agent，会卡在三个点：

- **上下文怎么喂？**（要不要把仓库都贴进去？）
- **怎么确保它不乱改？**（尤其是执行 shell、批量改文件）
- **怎么做团队级复用？**（统一规则、统一模型、统一扩展能力）

OpenCode 的设计核心，基本围绕这三件事：

1. **项目“可被理解”**：用 `/init` 生成 `AGENTS.md`（类似 CLAUDE.md / Cursor Rules），把项目结构与约定显式写成规则，让模型更懂你的代码库。
2. **能力“可被约束”**：通过 `permission`（允许/询问/拒绝）精细控制 `bash`、`edit/write/patch`、`webfetch` 等工具的使用；并提供默认的“Plan（只规划不改代码）/Build（能动手改）”工作流。
3. **扩展“可插拔”**：支持自定义命令、Agent、技能（SKILL.md）、插件、MCP（Model Context Protocol）服务器，以及通过 `opencode serve/web` 暴露服务端接口方便二次集成。

## 1. 快速上手（3 分钟版）

如果你只想最快跑起来，按这个顺序：

1. 安装 OpenCode
2. 在 TUI 里 `/connect` 绑定一个模型供应商
3. 进入项目目录运行 `opencode`，再执行 `/init` 生成 `AGENTS.md`

下面逐步展开。

## 2. 安装（macOS / Linux / Windows）

### 2.1 一键脚本（最省事）

官方推荐的安装方式是脚本：

```bash
curl -fsSL https://opencode.ai/install | bash
```

### 2.2 通过包管理器安装

OpenCode 也提供了多种安装渠道（以下为文档示例）：

**Node.js（全平台）**

```bash
npm install -g opencode-ai
```

也支持：

```bash
bun install -g opencode-ai
pnpm install -g opencode-ai
yarn global add opencode-ai
```

**Homebrew（macOS / Linux）**

```bash
brew install anomalyco/tap/opencode
```

**Arch（Paru）**

```bash
paru -S opencode-bin
```

**Windows（示例）**

```bash
choco install opencode
```

或：

```bash
scoop bucket add extras
scoop install extras/opencode
```

**Docker（快速试用）**

```bash
docker run -it --rm ghcr.io/anomalyco/opencode
```

## 3. 连接模型供应商（/connect、Zen、以及“自定义 OpenAI 兼容”）

OpenCode 支持大量 LLM 供应商（文档描述：基于 AI SDK + Models.dev），也支持本地模型。配置思路可以分两层：

- **凭证（API Key / OAuth / Profile）**：通过 `/connect` 添加；会落盘到本机的凭证文件（文档示例路径：`~/.local/share/opencode/auth.json`）。
- **行为（默认模型、baseURL、额外 headers、token 上限…）**：通过 `opencode.json`/`opencode.jsonc` 的 `provider`/`model` 等字段配置。

### 3.1 推荐新手：用 OpenCode Zen

文档里强烈建议新手从 **OpenCode Zen** 开始（“测试过、验证过的一组模型”）：

1. 在 TUI 里运行：

```text
/connect
```

2. 选择 `opencode`，打开 `opencode.ai/auth` 登录并拿到 API key
3. 回到终端粘贴 API key
4. 用 `/models` 查看可用模型（文档示例）：

```text
/models
```

### 3.2 配置 provider：baseURL（代理 / 网关 / 私有端点很常用）

你可以为任意 provider 配置 `baseURL`（文档示例以 Anthropic 为例）：

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "anthropic": {
      "options": {
        "baseURL": "https://api.anthropic.com/v1"
      }
    }
  }
}
```

### 3.3 自定义 provider：对接“OpenAI 兼容接口”

如果你用的是某个不在列表里的 OpenAI-compatible 服务，按文档步骤：

1. `/connect` → `Other`
2. 输入 provider id（例如 `myprovider`）
3. 输入 API key
4. 在项目里创建/修改 `opencode.json` 配置 `provider`：

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "myprovider": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "My AI Provider Display Name",
      "options": {
        "baseURL": "https://api.myprovider.com/v1"
      },
      "models": {
        "my-model-name": {
          "name": "My Model Display Name"
        }
      }
    }
  }
}
```

如果你还想在配置中显式声明 token 上限（让 OpenCode 更好估算上下文剩余），文档给了 `limit` 示例：

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "myprovider": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "My AI Provider Display Name",
      "options": {
        "baseURL": "https://api.myprovider.com/v1",
        "apiKey": "{env:ANTHROPIC_API_KEY}",
        "headers": {
          "Authorization": "Bearer custom-token"
        }
      },
      "models": {
        "my-model-name": {
          "name": "My Model Display Name",
          "limit": {
            "context": 200000,
            "output": 65536
          }
        }
      }
    }
  }
}
```

## 4. 在项目里开始工作：opencode + /init + AGENTS.md

### 4.1 启动 TUI

在项目根目录（或任意子目录）运行：

```bash
opencode
```

也可以指定项目路径：

```bash
opencode /path/to/project
```

### 4.2 初始化项目规则：/init

进入 TUI 后运行：

```text
/init
```

文档说明：这会分析项目并在项目根目录创建（或更新）`AGENTS.md`，帮助 OpenCode 理解项目结构、编码模式。官方也明确建议：**把 `AGENTS.md` 提交到 Git**，便于团队共享。

你后面日常提需求、让它改代码，效果的上限很大程度由 `AGENTS.md` 决定。

## 5. TUI 日常用法（你最常用的 4 件事）

### 5.1 引用文件：用 `@`（强烈推荐）

在消息里输入 `@` 会在当前工作目录做模糊搜索，并把文件内容自动注入上下文。例如（文档示例）：

```text
How is auth handled in @packages/functions/src/api/index.ts?
```

### 5.2 跑命令：用 `!`

消息以 `!` 开头会执行 shell 命令，并把输出作为 tool result 加进对话：

```text
!ls -la
```

### 5.3 运行内置命令：用 `/`

最常用的一批（均来自 TUI 文档）：

- `/help`：帮助与命令面板
- `/connect`：添加 provider 凭证
- `/models`：列出可用模型
- `/new`（别名 `/clear`）：新会话
- `/sessions`（别名 `/resume`、`/continue`）：切换会话
- `/undo` / `/redo`：回滚/重做（文档说明：底层用 Git 管理变更，所以项目需要是 Git 仓库）
- `/share` / `/unshare`：分享/取消分享会话
- `/editor`：调用外部编辑器写长 prompt（依赖 `$EDITOR`）
- `/export`：导出当前对话为 Markdown 并用 `$EDITOR` 打开

### 5.4 Plan vs Build：用 `Tab` 切换“先规划、再动手”

OpenCode 内置两个 primary agent：

- **Build**：默认，工具全开，用于真正改代码
- **Plan**：受限模式，默认把“文件改动”和“bash”设为需要确认（ask），适合做方案/评审/拆任务

一个非常稳的工作流是：

1. 先切到 Plan，让它给方案、列 TODO、指出风险
2. 你确认后切回 Build，让它落地实现

### 5.5 CLI 模式（脚本化/自动化）：把 OpenCode 当成命令行工具

除了交互式 TUI，OpenCode 也支持“像命令一样调用”的用法（文档示例）：

```bash
opencode run "Explain how closures work in JavaScript"
```

一些常用的 CLI 子命令（均有文档说明）：

- 查看已登录的 provider（`auth.json`）：

```bash
opencode auth list
```

- 列出可用模型（可指定 provider 过滤；也可 `--refresh` 刷新缓存）：

```bash
opencode models
opencode models anthropic
opencode models --refresh
```

- 查看会话与成本统计：

```bash
opencode session list
opencode stats
```

- 导出/导入会话（支持从分享链接导入）：

```bash
opencode export [sessionID]
opencode import session.json
opencode import https://opncd.ai/s/abc123
```

## 6. 权限系统（permission）：把“能做什么”收进你的控制范围

OpenCode 的权限系统用三种动作控制每个工具/行为：

- `"allow"`：直接执行
- `"ask"`：执行前询问你是否批准
- `"deny"`：直接拒绝

最简单的写法是整体开关：

```json
{ "$schema": "https://opencode.ai/config.json", "permission": "allow" }
```

更常见的是“默认 ask，再放行白名单”（文档示例风格）：

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "permission": {
    "*": "ask",
    "bash": "allow",
    "edit": "deny"
  }
}
```

### 6.1 更细粒度：按命令/路径写规则（最后匹配的规则生效）

权限支持对象语法，对 `bash` 可以按命令前缀匹配，对 `edit/read` 可以按路径匹配（文档示例）：

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "permission": {
    "bash": {
      "*": "ask",
      "git *": "allow",
      "npm *": "allow",
      "rm *": "deny"
    },
    "edit": {
      "*": "deny",
      "packages/web/src/content/docs/*.mdx": "allow"
    }
  }
}
```

### 6.2 默认行为与安全护栏

文档提到两个“安全护栏”：

- `external_directory`：工具触及项目工作目录之外时触发（默认 ask）
- `doom_loop`：相同 tool call 连续重复 3 次触发（默认 ask）

另外，`read` 默认允许，但对 `.env` 默认拒绝（文档示例）：

```jsonc
{
  "permission": {
    "read": {
      "*": "allow",
      "*.env": "deny",
      "*.env.*": "deny",
      "*.env.example": "allow"
    }
  }
}
```

### 6.3 权限可以按 Agent 覆盖

你可以在 `agent` 里单独覆盖某个 agent 的权限（文档示例）：

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "permission": {
    "bash": { "*": "ask", "git status": "allow" }
  },
  "agent": {
    "build": {
      "permission": {
        "bash": { "*": "ask", "git status": "allow", "git push": "allow" }
      }
    }
  }
}
```

## 7. 配置文件 opencode.json：放哪、怎么合并、怎么覆盖？

OpenCode 支持 `JSON` 和 `JSONC`（带注释）两种格式；并且**配置会合并（merge），不是整文件替换**。

文档给出的优先级（从低到高，后者覆盖前者冲突 key）是：

1. Remote config：`.well-known/opencode`（组织级默认）
2. Global：`~/.config/opencode/opencode.json`
3. Custom config：环境变量 `OPENCODE_CONFIG`
4. Project：项目根目录 `opencode.json`
5. `.opencode` 目录：agents / commands / plugins 等
6. Inline：环境变量 `OPENCODE_CONFIG_CONTENT`

### 7.1 一个最小可用的配置模板

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "theme": "opencode",
  "model": "anthropic/claude-sonnet-4-5",
  "autoupdate": true
}
```

### 7.2 变量替换：把敏感信息放到 env 或文件里

文档支持两种替换：

- `{env:VAR_NAME}`：读环境变量
- `{file:/path/to/file}`：读文件内容（可用于把 API key 放在不进 Git 的文件里）

示例（文档风格）：

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "model": "{env:OPENCODE_MODEL}",
  "provider": {
    "anthropic": {
      "options": {
        "apiKey": "{env:ANTHROPIC_API_KEY}"
      }
    }
  }
}
```

## 8. 扩展能力：命令、Agent、Skill、插件、MCP、自定义工具

这一节是“干货浓度最高”的部分：你把这些搭好后，OpenCode 就不再只是聊天，而是能变成你团队的“可复用工程能力层”。

### 8.1 自定义命令（/xxx）：.opencode/command/*.md

在 `.opencode/command/` 下创建 markdown 文件即可（文档示例）：

```md
---
description: Run tests with coverage
agent: build
model: anthropic/claude-3-5-sonnet-20241022
---
Run the full test suite with coverage report and show any failures.
Focus on the failing tests and suggest fixes.
```

文件名就是命令名，例如 `test.md` 对应：

```text
/test
```

模板支持：

- `$ARGUMENTS` / `$1` `$2` …：命令参数
- `!` + 反引号：把 shell 输出注入 prompt（例如 `!`\\`git log --oneline -10\\`）
- `@file`：注入文件内容

### 8.2 Agent：primary / subagent + 权限 + 工具开关

Agent 分两类：

- **primary**：你用 `Tab` 切换的主工作模式
- **subagent**：可被主 agent 调度，也可手动 `@xxx` 调用

你可以通过 `opencode agent create` 交互式创建；也可以写配置。文档支持两种定义方式：

- `opencode.json` 的 `agent` 字段
- `.opencode/agent/*.md` 或 `~/.config/opencode/agent/*.md`

（Agent 的关键点：模型、prompt、可用工具、权限规则、以及是否隐藏。）

### 8.3 Skills：把“可复用工作流”写成 SKILL.md（按需加载）

把技能放到 `.opencode/skill/<name>/SKILL.md`（或全局 `~/.config/opencode/skill/...`）即可。文档强调：

- `name` 必须小写字母数字 + 单个 `-` 分隔，且要和目录名一致
- skill 会在需要时由 `skill` 工具按需加载，而不是启动就全部塞进上下文（更省 token）

### 8.4 MCP Servers：接入外部系统（DB / 工单 / 文档检索等）

OpenCode 支持本地与远程 MCP server；启用后，MCP 工具会和内置工具一起供模型调用。

注意事项（文档原话要点）：**MCP 工具会增加上下文开销**，尤其是像 GitHub 这类工具多的 server，容易把 context 撑爆；所以要“少而精”地启用。

本地 MCP（`type: local`）示例（文档）：

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "mcp_everything": {
      "type": "local",
      "command": ["npx", "-y", "@modelcontextprotocol/server-everything"]
    }
  }
}
```

远程 MCP（`type: remote`）示例（文档）：

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "my-remote-mcp": {
      "type": "remote",
      "url": "https://my-mcp-server.com",
      "enabled": true,
      "headers": { "Authorization": "Bearer MY_API_KEY" }
    }
  }
}
```

文档还提到：远程 MCP 的 OAuth 会自动处理，token 会存到 `~/.local/share/opencode/mcp-auth.json`；你也可以用：

```bash
opencode mcp auth <name>
opencode mcp list
opencode mcp logout <name>
opencode mcp debug <name>
```

### 8.5 插件与自定义工具：把你的工程规范“自动化”

- 插件目录：`.opencode/plugin/`（项目级）或 `~/.config/opencode/plugin/`（全局）
- 也可以在 `opencode.json` 里通过 `plugin: []` 从 npm 加载（文档示例：`opencode-helicone-session`、`opencode-wakatime`）
- npm 插件会在启动时用 Bun 自动安装，缓存到 `~/.cache/opencode/node_modules/`

自定义工具（Custom tools）目录：

- `.opencode/tool/`（项目级）
- `~/.config/opencode/tool/`（全局）

文档示例使用 `@opencode-ai/plugin` 的 `tool()` helper 定义参数 schema 和执行逻辑；文件名就是工具名，也支持一个文件导出多个工具（工具名形如 `<filename>_<exportname>`）。

## 9. 分享会话（/share）：协作很好用，但默认是“公开链接”

OpenCode 的 share 会把会话同步到服务端并生成 `opncd.ai/s/<share-id>` 的公开链接。

三种模式（文档）：

- `manual`（默认）：手动 `/share`
- `auto`：新会话自动分享
- `disabled`：禁用

取消分享：

```text
/unshare
```

强烈建议：**涉及私有代码/敏感信息的项目，把 share 设为 disabled**。

## 10. IDE 集成（VS Code / Cursor 等）：跑一次 opencode 就能自动装扩展

文档说明：在 VS Code（及 Cursor、Windsurf、VSCodium 等 forks）的**集成终端**里运行 `opencode`，扩展会自动安装。

还提供了快捷键（文档）：

- 快速打开：macOS `Cmd+Esc` / Windows/Linux `Ctrl+Esc`
- 新开会话：macOS `Cmd+Shift+Esc` / Windows/Linux `Ctrl+Shift+Esc`
- 插入文件引用：macOS `Cmd+Option+K` / Windows/Linux `Alt+Ctrl+K`

如果你希望 `/editor` 或 `/export` 调起 VS Code，需要设置（文档建议）：

```bash
export EDITOR="code --wait"
```

## 11. 高级玩法：Server / Web / Attach / SDK

OpenCode 的架构是“客户端（TUI）+ 本地 server”，并且可以把 server 单独跑起来：

### 11.1 opencode serve：启动 headless HTTP server（OpenAPI）

```bash
opencode serve
```

文档给出的默认值：`port=4096`，`hostname=127.0.0.1`，支持 `--cors` 多次传入。

要启用 basic auth（对 `serve` 和 `web` 都生效）：

```bash
OPENCODE_SERVER_PASSWORD=your-password opencode serve
```

### 11.2 opencode web：带 Web UI 的 headless server

```bash
opencode web
```

### 11.3 attach：TUI 连接到远端 server（适合把 server 放在一台更强的机器上）

文档示例：

```bash
opencode web --port 4096 --hostname 0.0.0.0
opencode attach http://10.20.30.40:4096
```

### 11.4 SDK：用 JS/TS 程序化控制 OpenCode

文档 SDK 包：

```bash
npm install @opencode-ai/sdk
```

并可用 `createOpencode()` 创建 client（更适合做内部平台集成/自动化）。

## 12. 网络与排障（企业网络尤其重要）

### 12.1 代理（Proxy）

OpenCode 遵循标准代理环境变量（文档示例）：

```bash
export HTTPS_PROXY=https://proxy.example.com:8080
export NO_PROXY=localhost,127.0.0.1
```

注意：文档明确提醒——**TUI 会和本地 HTTP server 通信，必须让它走直连（NO_PROXY）**，否则可能形成路由环。

### 12.2 自定义 CA 证书

如果企业网络使用自定义 CA，文档建议：

```bash
export NODE_EXTRA_CA_CERTS=/path/to/ca-cert.pem
```

### 12.3 看日志与本地存储

文档给出排障方向：

- 日志目录：`~/.local/share/opencode/log/`（Windows 为 `%USERPROFILE%\.local\share\opencode\log\`）
- 可用 `--log-level DEBUG` 提升日志详细程度
- 本地数据目录：`~/.local/share/opencode/`，包含 `auth.json`、`log/`、`project/` 等

## 13. 我推荐的“稳健落地”路线（个人经验 + 文档建议融合）

最后给一条很实用的落地路线：

1. **先把权限收紧**：至少做到 `edit/bash` 默认 ask，`rm` deny（避免一键事故）。
2. **强制 /init + 提交 AGENTS.md**：团队协作时，这是“效果上限”的核心。
3. **把重复工作固化成 /commands + skills**：例如 `/test`、`/review`、`/release`，让团队每次提需求都更一致。
4. **谨慎启 MCP**：只启你真正用的 1–3 个；工具越多，上下文越容易爆。
5. **需要规模化再上 server/sdk/github agent**：先把日常体验跑通，再考虑平台化集成。

如果你打算在团队里推广 OpenCode，建议从“权限 + 规则 + 命令模板”三件套先做起来，效果会比单纯换模型更立竿见影。
