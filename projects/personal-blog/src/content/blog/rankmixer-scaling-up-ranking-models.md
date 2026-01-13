---
title: "RankMixer 论文解读：把排序模型做成“GPU 友好”的统一骨干，如何把 MFU 从 4.5% 拉到 45%？"
description: "解读 RankMixer（arXiv:2507.15551v3）：用语义分组的 Feature Tokenization + 参数无关的 Multi-head Token Mixing + Per-token FFN，并结合 ReLU Routing 与 Dense-training/Sparse-inference 的 Sparse-MoE，把工业排序模型推到 1B 参数且延迟基本不变。"
pubDate: 2026-01-13
tags: ["论文解读", "推荐系统", "排序模型", "Scaling Laws", "GPU", "MoE"]
category: "论文解读"
draft: false
---

> 论文：RankMixer: Scaling Up Ranking Models in Industrial Recommenders（arXiv:2507.15551v3）  
> 链接：<https://arxiv.org/abs/2507.15551>

RankMixer 这篇论文的气质非常“工业”：它不是去卷一个更复杂的交叉结构（DCN/AutoInt…），而是追问一个更底层的问题：

> **为什么排序模型在 GPU 上算得这么慢，以至于“把模型做大”很难真正带来 ROI？**

论文的答案很直接：很多工业排序模型的 Dense 交互模块，继承自 CPU 时代的手工 cross-feature / 交叉结构，**算子本质是 memory-bound**（大量不规则访存、小矩阵、Kernel 启动开销），在现代 GPU 上 **MFU（Model FLOPs Utilization）只有个位数**，参数一增，延迟和成本就炸。

RankMixer 想做的是一条“LLM 化”的路：给排序模型一个**统一、可扩展、硬件友好（hardware-aware）**的 backbone，让你能像扩 LLM 一样扩 ranking model —— 至少在系统和算子层面不被拖后腿。

下面我按 **背景 → 方法（含公式）→ 为什么快 → MoE 扩展 → 实验与线上效果 → 启示** 讲清楚。

## 1. 背景：排序模型为什么难 scale？

### 1.1 工业约束：高 QPS + 严格延迟

NLP 可以用“更慢一点换更好效果”，但排序阶段通常是在线链路核心，既要高 QPS，又要卡住 P99 延迟。于是推荐系统里经常出现这种局面：

- 业务数据量巨大（论文里是 Douyin 的 trillion 级日志），理论上很适合“更大模型吃更多数据”
- 但线上推理延迟、吞吐、成本把模型规模牢牢限制住

### 1.2 关键瓶颈：低 MFU（模型没把 GPU 吃满）

论文强调一个指标：**MFU = 模型实际用掉的 FLOPs / 硬件理论 FLOPs**。

- 如果 MFU 很低，说明你的模型算子并没有把 GPU 算力吃满，整体是 **memory-bound** 或被各种小 kernel/不规则操作拖死。
- 这会导致一个反直觉：**你把参数翻倍，延迟可能接近翻倍**，因为模型没有进入“算力受限（compute-bound）”状态，无法享受规模带来的并行效率。

论文给了一个对比（Table 6）：他们原线上 16M 模型 MFU 约 **4.47%**，RankMixer-1B 做到 **44.57%**，于是才能在“参数 ×70”的情况下把延迟维持在 **14ms 左右**。

## 2. RankMixer 的核心设计：统一、并行、可扩展

RankMixer 把“排序模型的 Dense 交互”抽象成：**若干个 feature token** 在多个 block 里反复做两类事情：

1. **TokenMixing：让不同 token 交换信息（全局交互）**
2. **PFFN：在各自 token 的语义子空间里做非线性建模（局部建模）**

整体结构非常像 Transformer block（残差 + LN + 两段子层），但关键算子不是 self-attention，而是更“硬件友好”的 token mixing。

## 3. 输入层：Feature Tokenization（把异构特征变成固定维度 token）

推荐系统特征极其异构：user/item id、数值统计、cross feature、序列特征、内容特征……embedding 维度也五花八门。

RankMixer 先做一个工程上很关键的决定：**把所有 embedding “对齐”成固定维度的 token**，后续才能做高度并行的矩阵乘。

### 3.1 语义分组，而不是“一特征一 token”

直觉上“一特征一 token”很自然，但论文指出工业里特征通常上百个：

- token 太多：每个 token 分到的参数/计算太少，重要特征建模不够；同时 token 维度小、矩阵形状碎，GPU 利用率差
- token 太少（甚至 1 个）：退化成普通 DNN，特征子空间不分离，容易出现“高频字段淹没长尾信号”

因此他们用**领域知识**把特征按语义分成若干组，拼成一个大向量，再切成固定数量的 token。

先把分组后的 embedding 串起来：

$$
\mathbf{e}_{\text{input}}=[\mathbf{e}_1;\mathbf{e}_2;\dots;\mathbf{e}_N].
$$

再按固定 chunk 大小切分并投影到统一维度 $D$（论文式(2)）：

$$
\mathbf{x}_i = \mathrm{Proj}\bigl(\mathbf{e}_{\text{input}}[d(i-1):di]\bigr),\quad i=1,\dots,T.
$$

得到 $T$ 个 feature token：$\mathbf{X}_0\in\mathbb{R}^{T\times D}$。

## 4. RankMixer Block：TokenMixing + Per-token FFN（含公式与直觉）

### 4.1 总体 block 形式

第 $n$ 个 block 的更新（论文式(1)）：

$$
\begin{aligned}
\mathbf{S}_{n-1} &= \mathrm{LN}\bigl(\mathrm{TokenMixing}(\mathbf{X}_{n-1}) + \mathbf{X}_{n-1}\bigr),\\
\mathbf{X}_n &= \mathrm{LN}\bigl(\mathrm{PFFN}(\mathbf{S}_{n-1}) + \mathbf{S}_{n-1}\bigr).
\end{aligned}
$$

最后对 $\mathbf{X}_L$ 做 mean pooling 得到输出向量，用于多任务预测。

### 4.2 Multi-head Token Mixing：一种“参数无关”的全局交互

这部分是 RankMixer 的灵魂之一：**用一种非常轻量的“重排/洗牌”完成 token 间信息交换**，并且借助残差与后续 FFN 让信息真正融合。

做法（论文式(3)(4)(5)）：

1. 把每个 token 平均切成 $H$ 个 head（每个 head 维度 $D/H$）：

$$
[\mathbf{x}_t^{(1)}\|\mathbf{x}_t^{(2)}\|\dots\|\mathbf{x}_t^{(H)}] = \mathrm{SplitHead}(\mathbf{x}_t).
$$

2. 对每个 head $h$，把所有 token 的第 $h$ 个 head 拼起来形成新的 token：

$$
\mathbf{s}^{h}=\mathrm{Concat}(\mathbf{x}_1^{h},\mathbf{x}_2^{h},\dots,\mathbf{x}_T^{h}).
$$

3. 在论文实现里设置 **$H=T$**，这样输出 token 数仍然是 $T$，可以直接做残差连接：

$$
\mathbf{s}_1,\dots,\mathbf{s}_T = \mathrm{LN}\bigl(\mathrm{TokenMixing}(\mathbf{x}_1,\dots,\mathbf{x}_T) + (\mathbf{x}_1,\dots,\mathbf{x}_T)\bigr).
$$

如果用一个小例子直观理解：设 $T=4$、$H=T$，每个 token 被切成 4 段，那么输出 token 1 会拿到 “原来 4 个 token 的第 1 段” 拼起来。于是每个输出 token 都天然包含来自所有输入 token 的信息碎片，后续的非线性层就可以在这个混合表示上学习交互。

#### 为什么不用 self-attention？

论文给了一个非常“推荐特征视角”的观点：推荐特征是异构语义空间，尤其 user/item id 空间巨大，**两种语义 token 做 inner-product 相似度很难学**，attention 的“相似度计算”未必像 NLP 那样自然有效；同时 attention 带来额外计算、显存和 IO。

在消融里（Table 3），Self-Attention 相比 Multi-head Token Mixing：

- 效果略差：$\Delta\mathrm{AUC}=-0.03\%$
- 但代价更大：参数 **+16%**、FLOPs **+71.8%**

这解释了 RankMixer 的“硬件优先”的取舍：在推荐的异构特征上，宁愿用更便宜、更稳的 mixing。

### 4.3 Per-token FFN（PFFN）：每个 token 独享一套 FFN 参数

第二个关键创新：**不同 token（不同语义子空间）不共享 FFN 参数**。

对第 $t$ 个 token $\mathbf{s}_t$，其 FFN（论文式(6)(7)）：

$$
\mathbf{v}_t=f_{\text{pffn}}^{t,2}\bigl(\mathrm{GELU}(f_{\text{pffn}}^{t,1}(\mathbf{s}_t))\bigr),
$$

其中

$$
f_{\text{pffn}}^{t,i}(\mathbf{x}) = \mathbf{x}\mathbf{W}_{\text{pffn}}^{t,i} + \mathbf{b}_{\text{pffn}}^{t,i},
$$

并且 $\mathbf{W}_{\text{pffn}}^{t,1}\in\mathbb{R}^{D\times kD}$、$\mathbf{W}_{\text{pffn}}^{t,2}\in\mathbb{R}^{kD\times D}$，$k$ 是 FFN 扩展倍数。

把它写成整体（论文式(8)）：

$$
\mathbf{v}_1,\dots,\mathbf{v}_T = \mathrm{PFFN}(\mathbf{s}_1,\dots,\mathbf{s}_T).
$$

#### 直觉：既要“分而治之”，也要“全局交互”

- TokenMixing 负责把信息在 token 间“洗牌”，建立跨子空间交互
- PFFN 负责在各自子空间里“加深非线性”，并且由于参数隔离，避免高频字段把长尾信号淹没

这也解释了它和 MMoE/共享 FFN 的区别：PFFN 是 **“输入分开 + 参数也分开”**，不是 “同一输入喂给多个专家”。

## 5. Sparse MoE 扩展：ReLU Routing + Dense-training/Sparse-inference（DTSI）

当你已经把 Dense 部分做成“算力友好”的矩阵乘后，下一步自然是冲到更大参数：论文把 RankMixer 扩到 **1B**，并讨论用 Sparse-MoE 继续扩到未来 10B 级别。

但他们发现 vanilla Sparse-MoE 在 RankMixer 上会退化，原因主要是：

1. **Uniform Top-$k$**：所有 token 都路由到同样数量的专家，浪费预算在“信息量低”的 token 上
2. **专家欠训练/失衡**：PFFN 本身已经把参数乘以 token 数，再加 MoE 会导致专家数量巨大，路由极不均衡，出现“dying experts”

### 5.1 ReLU Routing：让不同 token 激活不同数量专家

论文用 ReLU gate 替代 Top-$k$，并加一个 $\ell_1$ 风格正则来控制稀疏度（论文式(10)(11)）：

$$
G_{i,j}=\mathrm{ReLU}(h(\mathbf{s}_i)),\quad \mathbf{v}_i=\sum_{j=1}^{N_e} G_{i,j}\,e_{i,j}(\mathbf{s}_i),
$$

并在损失里加入稀疏正则：

$$
\mathcal{L}=\mathcal{L}_{\text{task}}+\lambda\mathcal{L}_{\text{reg}},\quad
\mathcal{L}_{\text{reg}}=\sum_{i=1}^{N_t}\sum_{j=1}^{N_e} G_{i,j}.
$$

直觉：信息量更高的 token 会激活更多专家，低信息 token 自动更稀疏，从而把计算预算花在刀刃上。

### 5.2 DTSI-MoE：训练时“密”，推理时“稀”

他们采用两个 router：$h_{\text{train}}$ 和 $h_{\text{infer}}$，训练时两者都更新，但稀疏正则只施加在 $h_{\text{infer}}$ 上；推理只用 $h_{\text{infer}}$。

核心收益：

- 训练阶段专家更“吃饱”（dense-training），缓解欠训练
- 推理阶段保持稀疏（sparse-inference），节省成本

## 6. 规模怎么涨：四个正交方向 + 复杂度公式

论文给出 RankMixer 的四个扩展轴：

- token 数 $T$
- 宽度 $D$
- 层数 $L$
- 专家数 $E$

对 full-dense 版本，参数量与 FLOPs 近似（论文式(12)）：

$$
\#\mathrm{Param}\approx 2kLTD^2,\qquad
\mathrm{FLOPs}\approx 4kLTD^2.
$$

这也是为什么他们强调“更大的 hidden dim 带来更大的 GEMM shape、更高 MFU”：**增宽往往比加深更硬件友好**。

## 7. 为什么 1B 参数延迟还能不变？（延迟分解公式）

论文在在线成本分析里给了一个很实用的分解（式(Ex1)）：

$$
\mathrm{Latency}=
\frac{\#\mathrm{Param}\times \mathrm{FLOPs/Param}}
{\mathrm{MFU}\times \text{(Theoretical Hardware FLOPs)}}.
$$

Table 6 给了一个非常“工程论文”的结论链条：

- 参数：15.8M → 1.1B（**×70**）
- FLOPs：107G → 2106G（**×20.7**），所以 FLOPs/Param 反而 **下降 3.6×**
- MFU：4.47% → 44.57%（**×10**）
- fp32 → fp16（理论峰值 **×2**）
- 最终延迟：14.5ms → 14.3ms（基本不变）

换句话说：**只要你能把模型从 memory-bound 推到 compute-bound，把算子做成大矩阵乘并把 MFU 拉起来，“更大模型”不一定更慢。**

## 8. 实验效果：离线对比、Scaling Law、消融、MoE、线上 A/B

### 8.1 数据与指标

论文离线实验来自 Douyin 推荐系统日志：

- 300+ 特征（数值、ID、cross、序列等），用户 ID 数十亿、视频 ID 数亿
- 日志量级：每天 trillion 级记录，使用两周数据训练

指标：

- 效果：AUC / UAUC（finish、skip 两类目标）
- 效率：Dense Param、FLOPs/Batch、MFU

### 8.2 离线对比：RankMixer-100M/1B 显著优于 SOTA（Table 1）

在 ~100M 参数量级对比中，RankMixer-100M 相对 DLRM-MLP(base) 的提升（论文表格是相对增益）：

- Finish：AUC **+0.64%**，UAUC **+0.72%**
- Skip：AUC **+0.86%**，UAUC **+1.33%**

并且在 1B 规模进一步提升到：

- Finish：AUC **+0.95%**，UAUC **+1.22%**
- Skip：AUC **+1.25%**，UAUC **+1.82%**

对比 Wukong/HiFormer 等“可扩展结构”，RankMixer 不仅效果更好，100M 规模下 FLOPs 也更温和（233G vs 300G/400G 级别的 baseline）。

### 8.3 Scaling Laws：RankMixer 的曲线最陡（Figure 2）

论文展示了 “AUC gain vs Params / FLOPs” 的 scaling 曲线：

- RankMixer 在参数和 FLOPs 两个维度都呈现最陡的 scaling
- Wukong 参数曲线也比较陡，但 FLOPs 增长更快，导致 AUC vs FLOPs 曲线不占优
- DHEN/MoE 受限于交叉结构与专家失衡，scaling 不理想

他们还观察到一个与 LLM 类似的结论：**模型质量主要与总参数相关**，不同扩展方向（$L/D/T$）在效果上差异不大；但从 MFU 角度更偏好“增宽”。最终配置：

- 100M：$(D=768,\;T=16,\;L=2)$
- 1B：$(D=1536,\;T=32,\;L=2)$

### 8.4 消融：TokenMixing 与 PFFN 都是关键（Table 2/3）

RankMixer-100M 的消融（Table 2）：

- 去掉 skip connection：$\Delta\mathrm{AUC}=-0.07\%$
- 去掉 Multi-head Token Mixing：$\Delta\mathrm{AUC}=-0.50\%$
- 去掉 LayerNorm：$\Delta\mathrm{AUC}=-0.05\%$
- Per-token FFN → shared FFN：$\Delta\mathrm{AUC}=-0.31\%$

TokenMixing 路由策略对比（Table 3）：

- All-Concat-MLP：$\Delta\mathrm{AUC}=-0.18\%$
- All-Share：$\Delta\mathrm{AUC}=-0.25\%$
- Self-Attention：$\Delta\mathrm{AUC}=-0.03\%$，但 Params +16%、FLOPs +71.8%

结论很一致：**推荐特征的“子空间分离 + 轻量全局交互”是有效的**，而 attention 在这里不划算。

### 8.5 Sparse-MoE：DTSI + ReLU Routing 在高稀疏下几乎不掉点（Figure 3/4）

论文展示在激活专家比例从 1 降到 1/8 时：

- Vanilla SMoE AUC 会单调下降（专家失衡、欠训练）
- DTSI + ReLU routing 能在激进稀疏下 **几乎保留 1B dense 的精度**
- 推理吞吐提升显著（论文提到 **+50% throughput**），并认为这是通往未来 10B 级 RankMixer 的路线

### 8.6 线上 A/B：推荐与广告都获得显著收益（Table 4/5）

Feed 推荐（Douyin app Overall）：

- Active Day：**+0.2908%**
- Duration：**+1.0836%**
- Like：+2.3852%，Finish：+1.9874%，Comment：+0.7886%

并且低活跃用户收益最大（Active Day **+1.7412%**）。

广告场景（Table 5）：

- $\Delta\mathrm{AUC}$：**+0.73%**
- ADVV（Advertiser Value）：**+3.90%**

### 8.7 线上成本：参数扩大两阶，延迟基本不变（Table 6）

最值得记住的是 Table 6 的工程结论：

- **MFU：4.47% → 44.57%**
- **Latency：14.5ms → 14.3ms**

这基本证明了 RankMixer 的立论：把骨干做成 GPU 友好的大矩阵乘，你就有资格谈“排序模型的 scaling law”。

## 9. 这篇论文带来的启示（我的总结）

1. **推荐排序的可扩展性，不只靠“加参数”，更靠“让 GPU 吃饱”**：把 MFU 作为一等公民，会直接改变模型设计。
2. **异构特征不一定适合 attention 的“相似度范式”**：推荐 token 的语义空间更碎、更不统一，参数无关的 mixing + 大 FFN 可能更稳、更便宜。
3. **“分而治之”的 per-token 参数隔离很值得尝试**：它提供了一种增加参数容量但不显著增加计算路径复杂度的方式。
4. **MoE 要解决的核心往往不是 router，而是 expert training**：DTSI 这种“训练密、推理稀”的策略很实用。

如果你正在做工业排序模型，RankMixer 的价值不只是一种结构，而是一套“硬件对齐”的思路：**用统一 backbone 替换低 MFU 的异构交叉结构，把建模复杂度交给可并行的大算子**。
