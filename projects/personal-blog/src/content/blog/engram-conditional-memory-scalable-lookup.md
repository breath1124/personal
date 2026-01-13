---
title: "DeepSeek Engram 论文解读：用可扩展查表给 LLM 增加“条件记忆”稀疏轴"
description: "解读 Conditional Memory via Scalable Lookup（arXiv:2601.07372v1）：提出 Engram，把经典 N-gram 表征升级成可扩展、可训练、可 CPU Offload 的 O(1) 查表记忆模块；用上下文门控与轻量卷积把静态记忆与动态隐状态融合；并给出 MoE 与 Engram 的稀疏预算分配规律（U 形 scaling law），在等激活参数/等 FLOPs 下显著提升知识、推理、代码/数学与长上下文检索。"
pubDate: 2026-01-13
tags: ["论文解读", "LLM", "稀疏模型", "MoE", "Memory", "Engram", "N-gram", "Scaling Laws", "长上下文"]
category: "论文解读"
draft: false
---

> 论文：Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models（arXiv:2601.07372v1）  
> 链接：<https://arxiv.org/abs/2601.07372>（HTML：<https://arxiv.org/html/2601.07372v1>）  
> 代码：<https://github.com/deepseek-ai/Engram>

大模型的“稀疏化”这几年主要沿着一条轴在卷：**MoE（Mixture-of-Experts）**。它的逻辑很直接：模型参数可以很大，但每个 token 只激活少数专家，于是 **容量 ↑**、**计算量（FLOPs）不随总参数线性增加**。

但这篇 DeepSeek 的论文提出了一个非常尖锐的观察：

- **MoE 解决的是“条件计算（conditional computation）”**：把“算什么”做稀疏化。
- 大多数 Transformer 仍然缺一个原语：**“条件记忆（conditional memory）/ 知识查找（knowledge lookup）”**。它们往往只能用更多计算（更多层/更多注意力）去“模拟检索”，效率不高。

于是论文提出 **Engram**：一种“现代化的 N-gram 记忆模块”，核心是把“局部模式/静态知识”做成**可训练的超大查表（embedding table）**，通过**确定性 hash** 用 $\mathcal{O}(1)$ 时间取回向量，再用一个**上下文门控**决定“这条记忆到底该不该用”。

如果把 MoE 看作“脑区里可被召唤的计算回路”，Engram 更像“可被条件触发的长期记忆痕迹（engram）”。更关键的是：**记忆容量可以大到 100B+，还能从 GPU HBM 迁到 CPU/DRAM，推理吞吐几乎不掉（<3%）**——这就把“参数规模”与“GPU 显存”这条硬约束撬开了一点。

下面我按 **动机 → Engram 模块（重点公式）→ 稀疏预算怎么分（关键公式）→ 实验与分析**，把最重要的思想与公式讲清楚。

## 1. 为什么说 Transformer 缺一个“查表原语”？

直觉上，Transformer 做“记忆”的方式有两类：

1. **通过参数存知识**：某些事实/模式被编码进 FFN/Attention 的权重里；需要时靠多层计算“重建”出来。
2. **通过注意力做检索**：在长上下文里，模型要把注意力预算用在“找对 token”上；但注意力既要处理局部依赖（n-gram 级别的拼写/词形/短语），又要处理远程依赖（跨段落检索/多跳推理），会互相挤占。

论文的核心假设是：**很多“静态且局部”的东西，其实更像查表**，不必每次都让网络重新算一遍；把它们卸载出去，能让主干网络更专注于“动态推理”和“全局整合”。

Engram 就是在这一点上“给 Transformer 增加一个新积木”。

## 2. Engram 模块：两阶段（检索 → 融合）

在模型的某些层（不是每一层）插入 Engram。给定 token 序列

$$
X=(x_1,\dots,x_T),
$$

以及第 $\ell$ 层的隐状态矩阵

$$
\mathbf{H}^{(\ell)}\in\mathbb{R}^{T\times d},
$$

Engram 对每个位置 $t$ 做两件事：

1. **从“suffix N-gram”确定性地检索静态向量**（检索阶段）
2. **用当前上下文隐状态决定“用不用/用多少”**（融合阶段）

论文默认用到 **2-gram 与 3-gram（$N=3$）**，因为这些局部模式最常见、最像“静态短语/词形规律”。

### 2.1 Tokenizer Compression：先把“等价 token”折叠掉

N-gram 的一个尴尬点是：**子词 tokenizer 的 ID 并不“语义一致”**。比如同一个词可能因为空格/大小写/NFKC 归一化而被编码成不同 token（论文举例：`Apple` vs `␣apple`）。

如果你直接在原始 token ID 上做 N-gram，记忆表会被“同义但不同 ID 的模式”稀释，浪费容量。

因此论文引入一个预处理：构造一个满射（surjective）映射

$$
\mathcal{P}:V\to V',
$$

把原始词表 $V$ 的 token ID 折叠成更小、更“规范化”的词表 $V'$（基于 NFKC、lowercase 等文本等价规则）。论文报告：对 128k tokenizer，这一步能让有效词表规模减少 **23%**。

对位置 $t$：

$$
x'_t=\mathcal{P}(x_t),
$$

然后构造 suffix $n$-gram：

$$
g_{t,n}=(x'_{t-n+1},\dots,x'_t).
$$

你可以把它理解成：**先把“字面差异”压掉，再让记忆表学“模式本身”**。

### 2.2 哈希检索：用 $\mathcal{O}(1)$ 查表近似“组合爆炸”的 N-gram 空间

如果你真的给所有可能的 2/3-gram 都建一个独立 embedding，参数会爆炸（组合空间太大）。

Engram 的做法是：**哈希到大表里**，并用 **多头哈希（multi-head hashing）**减轻碰撞。

对每个 $n$-gram 阶数 $n$，使用 $K$ 个哈希头。第 $k$ 个头对应一个 embedding 表 $\mathbf{E}_{n,k}$（表大小是素数 $M_{n,k}$），用确定性函数 $\varphi_{n,k}$ 把 $g_{t,n}$ 映射到表的行号：

$$
z_{t,n,k}\triangleq\varphi_{n,k}(g_{t,n}),\quad
\mathbf{e}_{t,n,k}=\mathbf{E}_{n,k}[z_{t,n,k}].
$$

（论文实现里 $\varphi_{n,k}$ 是轻量的 multiplicative-XOR hash。）

最后把所有检索到的向量拼起来，得到位置 $t$ 的“静态记忆向量”：

$$
\mathbf{e}_t\triangleq \mathop{\|}_{n=2}^{N}\mathop{\|}_{k=1}^{K}\mathbf{e}_{t,n,k}
\in \mathbb{R}^{d_{\text{mem}}}.
$$

直觉：

- **单头哈希**：碰撞多，噪声大。
- **多头哈希**：同一个 n-gram 会取到多个槽位向量；即使部分碰撞，仍可能保留可用信号；后面再靠“门控”把不一致的部分压下去。

一个把流程“落到手上”的小例子（简化版）：

- 假设规范化后的 token 是：`x' = [..., "new", "york", "city"]`，并且当前位置 $t$ 对应 `"city"`。
- 2-gram：$g_{t,2}=(x'_{t-1},x'_t)=("york","city")$
- 3-gram：$g_{t,3}=(x'_{t-2},x'_{t-1},x'_t)=("new","york","city")$
- 对每个 $g_{t,n}$，每个哈希头 $k$ 都会算出一个槽位索引 $z_{t,n,k}$，然后从表 $\mathbf{E}_{n,k}$ 取出 $\mathbf{e}_{t,n,k}$。
- 拼接得到 $\mathbf{e}_t$ 后，门控会用当前上下文隐状态 $\mathbf{h}_t$ 判断：这条短语模式到底是不是“我正在讨论的那个 New York City”（而不是碰撞到的别的模式）。如果不匹配，$\alpha_t\approx 0$，这段记忆就基本不会进入主干。

### 2.3 上下文门控（Context-aware Gating）：用“全局上下文”消歧与去噪

仅靠哈希检索得到的 $\mathbf{e}_t$ 是**上下文无关（context-independent）**的；它可能因为哈希碰撞或一词多义而“取错”。

所以 Engram 不是把它硬塞进主干，而是引入一个门控：用当前层的隐状态 $\mathbf{h}_t$ 做 Query，让记忆向量经投影后充当 Key/Value。

先做线性投影：

$$
\mathbf{k}_t=\mathbf{W}_K\mathbf{e}_t,\quad
\mathbf{v}_t=\mathbf{W}_V\mathbf{e}_t.
$$

然后计算一个标量门值 $\alpha_t\in(0,1)$。论文在 Query/Key 上做 RMSNorm，并用 sigmoid 得到门值：

$$
\alpha_t=\sigma\left(
\frac{\text{RMSNorm}(\mathbf{h}_t)^\top\text{RMSNorm}(\mathbf{k}_t)}{\sqrt{d}}
\right).
$$

最后把 Value 乘上门控：

$$
\tilde{\mathbf{v}}_t=\alpha_t\cdot \mathbf{v}_t.
$$

这个设计非常关键：**当检索到的静态模式与当前语境矛盾时**（比如碰撞到了不相关的短语），$\alpha_t$ 会趋近于 0，把噪声压掉。

### 2.4 轻量卷积融合：让记忆不仅“看当前 token”，还看一个短窗口

门控后的序列 $\tilde{\mathbf{V}}\in\mathbb{R}^{T\times d}$（把所有 $\tilde{\mathbf{v}}_t$ 堆起来）再过一个**深度可分离的因果卷积**，并加残差：

$$
\mathbf{Y}=\text{SiLU}\left(\text{Conv1D}(\text{RMSNorm}(\tilde{\mathbf{V}}))\right)+\tilde{\mathbf{V}}.
$$

论文设置卷积核大小 $w=4$，dilation $\delta$ 取最大 n-gram 阶数（这里 $N=3$）。直觉上：

- 记忆检索是“离散、静态”的；
- 卷积提供一个小的连续可学习混合，让相邻位置的门控记忆能合成更稳定的局部特征（类似“短程特征提取器”）。

最后，Engram 以残差的方式注入主干：

$$
\mathbf{H}^{(\ell)}\leftarrow \mathbf{H}^{(\ell)}+\mathbf{Y},
$$

再接标准的 Attention / MoE 块。论文强调：**Engram 不需要每层都放**，放在哪里要同时考虑建模效果与系统延迟（后面会讲）。

### 2.5 和多分支（multi-branch）主干的结合：共享表、分支特定门控

论文默认主干不是单一 residual stream，而是把 residual 扩成 $M$ 个并行分支的信息流（multi-branch / hyperconnections 一类）。

Engram 适配多分支时采取“参数共享 + 门控分支化”：

- **共享**：一个稀疏 embedding 表 + 一个 $\mathbf{W}_V$，所有分支共用（省参数）
- **分支特定**：每个分支有自己的 $\mathbf{W}_K^{(m)}$，让不同分支学到不同的门控模式

第 $m$ 个分支的门控为：

$$
\alpha_t^{(m)}=\sigma\left(
\frac{\text{RMSNorm}(\mathbf{h}_t^{(m)})^\top\text{RMSNorm}(\mathbf{W}_K^{(m)}\mathbf{e}_t)}{\sqrt{d}}
\right).
$$

这背后的直觉是：**分支做“不同视角的特征通道”**，让同一份静态记忆在不同通道里被不同程度地利用，从而提升表达力。

## 3. “稀疏预算”怎么在 MoE 与 Engram 之间分？（U 形 scaling law）

Engram 的一个很实用的问题是：如果我已经用了 MoE，我该给 Engram 多大预算？会不会“挤占专家”反而变差？

论文把这个问题形式化成 **Sparsity Allocation**。

### 3.1 三个参数量：总参数、激活参数、稀疏参数

论文用三个量描述“计算成本 vs 容量”：

- $P_{\mathrm{tot}}$：总可训练参数（不含 vocab embedding 与 LM head）
- $P_{\mathrm{act}}$：每个 token 实际激活的参数量（决定 FLOPs）
- $P_{\mathrm{sparse}}\triangleq P_{\mathrm{tot}}-P_{\mathrm{act}}$：不增加每 token 计算、但可以扩容的“稀疏预算”（例如未被路由的专家参数、未被访问的记忆槽位）

在严格的 iso-FLOPs 比较里，我们希望 $P_{\mathrm{tot}}$ 和 $P_{\mathrm{act}}$ 都匹配；那么差异主要体现在 **$P_{\mathrm{sparse}}$ 怎么花**。

### 3.2 分配系数 $\rho$：给 MoE 多少、给 Engram 多少

定义 $\rho\in[0,1]$ 为“稀疏预算中分给 MoE 专家容量的比例”，则：

$$
P_{\mathrm{MoE}}^{(\mathrm{sparse})}=\rho\,P_{\mathrm{sparse}},\qquad
P_{\mathrm{Engram}}=(1-\rho)\,P_{\mathrm{sparse}}.
$$

两个极端：

- $\rho=1$：纯 MoE（所有稀疏预算都变成专家）
- $\rho\to 0$：Engram 占主导（专家大幅减少、记忆巨大）

### 3.3 关键发现：验证损失 vs $\rho$ 呈稳定 U 形，最优点在 75%–80% 左右

论文在两个算力预算下做 sweep，发现验证损失随 $\rho$ 是**稳定 U 形**：

- **MoE 占太多（$\rho\to 100\%$）**：没有专用记忆，“静态模式”要靠深度与计算去重建，效率差。
- **Engram 占太多（$\rho\to 0\%$）**：模型缺少条件计算能力，动态推理/上下文相关的任务受损——**记忆替代不了计算**。

更具体地，论文报告：

- 即使把 MoE 的分配降到 $\rho\approx 40\%$，Engram 模型也能做到接近纯 MoE（$\rho=100\%$）的性能（但这不是最优点）。
- 最优点通常在 $\rho\approx 75\%\text{--}80\%$，也就是把 **20%–25% 的稀疏预算**挪给 Engram。
- 在 10B 量级算力设定（$C=6\times 10^{20}$）里，验证 loss 从 $\rho=100\%$ 的 **1.7248** 降到最优附近（$\rho\approx 80\%$）的 **1.7109**（$\Delta=0.0139$）。

这条结论很值得记住：**Engram 不是“加法外挂”，它和 MoE 在预算上是互补且可优化的。**

## 4. 大规模预训练实验：等激活参数/等 FLOPs 下，Engram 全面赢过 MoE

论文在 262B tokens 预训练后，对比了 4 类模型（都匹配激活参数 3.8B）：

- Dense-4B：稠密基线
- MoE-27B：MoE 稀疏基线（总参约 26.7B）
- Engram-27B：与 MoE-27B **等总参/等 FLOPs**，通过减少路由专家，把 **5.7B** 参数挪给 Engram 记忆
- Engram-40B：在激活参数固定下继续扩记忆（记忆 **18.5B**，总参约 39.5B）

下面摘 Table 1 的关键结果（均为论文原表数字）：

- **语言建模 loss**：Pile loss 从 MoE-27B 的 1.960 → Engram-27B 的 1.950（Engram-40B 进一步到 1.942）
- **知识/推理（代表）**
  - MMLU：57.4 → 60.4（+3.0）
  - CMMLU：57.9 → 61.9（+4.0）
  - BBH：50.9 → 55.9（+5.0）
  - ARC-Challenge：70.1 → 73.8（+3.7）
- **代码/数学（代表）**
  - HumanEval Pass@1：37.8 → 40.8（+3.0）
  - GSM8K：58.4 → 60.6（+2.2）
  - MATH：28.3 → 30.7（+2.4）

一个很反直觉但很重要的现象是：**记忆模块不仅提升知识类任务，还更明显地提升“推理/数学/代码”**。

论文给的解释方向是：Engram 把主干早期层从“静态重建”里解放出来，使网络“有效深度”更像变深了；同时把局部依赖交给查表，让注意力把预算留给更关键的全局信息（长上下文检索尤其明显）。

## 5. 长上下文：为什么 Engram 能把 MQ-NIAH 从 84.2 拉到 97.0？

长上下文评价主要看两类：

- **LongPPL (32k)**：长文本困惑度
- **RULER (32k)**：一组长上下文检索/跟踪任务（NIAH、Variable Tracking 等）

Table 2 给了一个非常醒目的结果：在 32k 设置下，

- Multi-Query NIAH（MQ）：**84.2 → 97.0**
- Variable Tracking（VT）：**77.0 → 89.0**

论文在引言里用一句话总结了直觉：**把局部依赖交给 Engram 的 O(1) 查表，注意力就能更专注于“全局检索与整合”。**

你可以把它类比成：

- 以前注意力既要当“拼写检查器/短语解析器”，又要当“信息检索器”；
- 现在有一部分“短语解析”交给 Engram，注意力更像“专职检索器”。

## 6. 机制分析：Engram 像是在“加深模型”，也像是在“搬走一部分知识仓库”

论文做了不少 mechanistic analysis，这里挑和理解 Engram 最相关的三条。

### 6.1 “有效深度”视角：预测更早收敛、表征相似性对齐上移

作者提出一个假设：显式查表让模型少走“早期逐层拼装特征”的弯路，于是表现得像“更深”。

他们用两类工具验证：

- **LogitLens**：看各层 hidden state 经过最终 LM head 后的预测分布，观察预测随层数怎么收敛。
- **CKA（Centered Kernel Alignment）**：比较两模型不同层表征的相似性。论文给出 CKA 形式：

$$
\text{CKA}(K,L)=\frac{\text{HSIC}(K,L)}{\sqrt{\text{HSIC}(K,K)\text{HSIC}(L,L)}}.
$$

结论是：Engram 的浅层表征与 MoE 的更深层更相似（相似性对角线整体“上移”），支持“有效深度增加”的解释。

论文还把这种“对齐上移”进一步量化成一条对齐曲线：先定义相似度分数 $S_{i,j}$（MoE 第 $i$ 层 vs Engram 第 $j$ 层），再用 top-$k$ 过滤后的加权平均位置作为 Engram 层 $j$ 对应的“等效 MoE 深度”：

$$
\mathcal{I}_j = \operatorname{TopK}_i(S_{i,j},k),\qquad
a_j=\frac{\sum_{i\in\mathcal{I}_j}S_{i,j}\cdot i}{\sum_{i\in\mathcal{I}_j}S_{i,j}}.
$$

当大量层满足 $a_j>j$ 时，就意味着：**Engram 在更浅的层就形成了 MoE 需要更深层才形成的表征**。

### 6.2 位置选择的权衡：早放利于卸载局部模式，但门控需要一定上下文

论文指出 Engram 放哪一层有一个权衡：

- 放得**早**：能更早卸载局部模式重建，节省主干深度
- 放得**晚**：门控 Query（$\mathbf{h}_t$）更上下文化，门控更精确；同时系统上更容易隐藏 PCIe 预取延迟

他们的消融里提到：单层注入时，Layer 2 的效果最好；把同样的记忆预算拆成两份，放在 Layer 2 和 Layer 6 更好（同时也更利于系统层的预取/缓存层级）。

### 6.3 把 Engram 输出“强行置零”会发生什么？知识任务崩、阅读理解相对稳

论文做了一个 stress test：推理时完全抑制 Engram 的稀疏 embedding 输出（主干不变）。

结果呈现强烈的两极分化：

- **事实性知识（TriviaQA 等）灾难性下降**：只保留 29%–44% 的原性能（TriviaQA 约 29%）
- **阅读理解（C3 等）相对稳**：仍保留 81%–93%（C3 约 93%）

这很符合 Engram 的定位：它更像“静态知识/局部模式仓库”；而阅读理解更依赖“上下文内的注意力整合”。

## 7. 系统效率：为什么说 Engram 比 MoE 更容易做“硬件友好”的优化？

MoE 的 routing 依赖运行时 hidden state，专家选择是动态的；而 Engram 的索引只依赖 token 序列（hash 后的 ID），因此 **访问模式是确定性的**：

- 还没跑到那一层，系统就已经知道下一步要访问哪些槽位
- 可以做 **prefetch-and-overlap**：把 CPU→GPU 的传输和前面若干层的计算重叠起来

论文在 H800 上做了一个极端实验：在第二个 Transformer block 插一个 **100B 参数**的 Engram 层，整张表放在 host DRAM，通过预取隐藏 PCIe 延迟。Table 4 的吞吐显示：

- 4B-Dense：9031.62 tok/s → +100B Engram（CPU Offload）后 8858.28 tok/s
- 8B-Dense：6315.52 tok/s → +100B Engram（CPU Offload）后 6140.02 tok/s

即吞吐下降都在 **3%以内**。这条结果的意义在于：**“参数规模”不再被 GPU HBM 卡死**，你可以把“静态容量”更大胆地放到更便宜的存储层级里。

再结合语言 N-gram 的 Zipf 分布（少数模式占大多数访问），论文还提出多级缓存层级（HBM/DRAM/NVMe）来进一步扩容而不显著伤延迟。

## 8. 总结：Engram 给了我们哪些可复用的启示？

我觉得这篇论文最值得带走的是三句话：

1. **稀疏不只有 MoE**：除了“条件计算”，还可以把“条件记忆”作为另一条稀疏轴。
2. **显式查表 + 上下文门控**是一种很强的组合：它把“静态局部模式”从主干计算里卸载，但又不会把噪声硬塞进来。
3. **预算分配是可优化的**：在严格 iso-FLOPs 下，最优点稳定出现在“给 Engram 20%–25% 稀疏预算”的附近；同时系统层还能用确定性预取把 100B 记忆从 GPU 迁到 CPU，吞吐几乎不掉。

如果你在做 MoE 或任何稀疏大模型，这篇论文提供了一个很实用的思路：**把“能查表的”交给 memory，把“要推理的”交给 compute**，两者在预算上做最优折中，而不是只在一条轴上堆参数。
