---
title: "HSTU 论文解读：把推荐建模成“内容-动作”的生成式序列转导，如何扩到万亿参数？"
description: "解读 HSTU（arXiv:2402.17152v3）：将推荐系统重述为序列转导任务并做生成式训练；提出 HSTU（pointwise aggregated attention + 相对时序偏置 + 门控无 FFN）与 Stochastic Length、M-FALCON 等工程算法，实现长序列、流式训练与大规模在线推理。"
pubDate: 2026-01-13
tags: ["论文解读", "推荐系统", "生成式推荐", "序列建模", "Transformer", "HSTU", "Scaling Laws"]
category: "论文解读"
draft: false
---

> 论文：Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations（arXiv:2402.17152v3）  
> 链接：<https://arxiv.org/abs/2402.17152>

这篇论文（通常简称 **HSTU**，Hierarchical Sequential Transduction Units）做了两件事：

1. **思想层面**：把工业推荐系统的“召回/排序”统一重述为 **序列转导（sequential transduction）**，并强调真正该建模的是“系统给内容 + 用户做动作”的**联合生成过程**，而不是只预测“下一个 item”。
2. **结构与工程层面**：提出一个面向推荐的高性能自注意力编码器 **HSTU**，并配套提出 **Stochastic Length（SL）**、**M-FALCON** 等算法，让模型可以在**超长序列 + 流式训练（single-pass）+ 大规模候选在线推理**下仍然可训练、可部署，最终支撑到 **万亿级参数**并观察到类似 LLM 的 scaling law。

下面我按“背景 → 统一表述（核心公式）→ HSTU（核心公式）→ 让它能 scale 的算法 → 实验结果 → 关键启示”展开，尽量通俗但把公式讲清楚。

## 1. 背景：为什么工业推荐很难用“LLM 式 scaling”解决？

推荐系统（尤其排序阶段）有三个现实约束：

- **特征高度异构**：工业 DLRM 往往用上百种特征（用户画像、上下文、交叉统计、序列特征……），大量依赖人工特征工程。
- **训练形态是流式的**：工业日志巨大，常用 single-pass streaming（不停增长的数据，难以 full-shuffle、多 epoch）。
- **线上推理要处理海量候选**：排序阶段一次要评分上千到上万候选，任何“对每个候选都做一次重计算”的结构都很难落地。

学术界常见的序列推荐（如 SASRec、BERT4Rec）多是“只看用户交互 item 序列、multi-pass、full-shuffle”的设置；工业 DLRM 则是“特征丰富、但序列模块与交叉模块往往难以扩展”。这篇论文的目标，就是把两者合在一起：**既保留工业特征语义与目标感知（target-aware），又能像 LLM 一样扩展序列模型**。

## 2. 把推荐统一成序列转导：输入 token 与输出 token（核心表述）

论文先给一个统一定义（Sec. 2.2）：

给定按时间排序的输入 token 序列

$$
x_0,x_1,\ldots,x_{n-1},\quad x_i\in\mathbb{X},
$$

输出 token 序列

$$
y_0,y_1,\ldots,y_{n-1},\quad y_i\in\mathbb{X}\cup\{\varnothing\},
$$

其中 $y_i=\varnothing$ 表示这个位置“不需要预测”（例如该 token 只是上下文信息，不产生监督）。

推荐里有两类特别关键的 token（Sec. 2.2）：

- **内容 token**：系统展示给用户的内容 $\Phi_i\in\mathbb{X}_c$（$\mathbb{X}_c\subseteq\mathbb{X}$，而且是非平稳的：新内容不断产生）
- **动作 token**：用户对内容的反馈动作 $a_i\in\mathbb{X}$（可以是 skip/like/完播/分享等，甚至是多热向量）

用户一共与 $n_c$ 个内容发生过交互。

### 2.1 排序与召回都可以写成“序列转导任务”（Table 1）

论文用 Table 1 把 ranking / retrieval 的输入输出写得非常直观：

**排序（Ranking）**：把内容与动作交错成序列作为输入，监督是在“内容位置”预测动作：

- 输入 $x_i$：

$$
\Phi_0,a_0,\Phi_1,a_1,\ldots,\Phi_{n_c-1},a_{n_c-1}
$$

- 输出 $y_i$：

$$
a_0,\varnothing,a_1,\varnothing,\ldots,a_{n_c-1},\varnothing
$$

直觉：模型在看到候选内容 $\Phi_i$（以及过去历史）时，预测用户会产生什么动作 $a_i$，这就是“目标感知（target-aware）”的排序建模方式。

**召回（Retrieval）**：输入是内容-动作对，输出只监督“正反馈内容”（负反馈对应 $\varnothing$）：

- 输入 $x_i$：

$$
(\Phi_0,a_0),(\Phi_1,a_1),\ldots,(\Phi_{n_c-1},a_{n_c-1})
$$

- 输出 $y_i$：

$$
\Phi_1',\Phi_2',\ldots,\Phi_{n_c-1}',\varnothing,\quad
\Phi_i'=\begin{cases}
\Phi_i,& a_i\ \text{是正反馈}\\
\varnothing,& \text{否则}
\end{cases}
$$

直觉：召回要学的是“用户会喜欢什么内容”，而不是“系统展示了什么内容”。因此监督不等于 $\Phi_{i+1}$（用户可能对它是负反馈）。

### 2.2 关键思想：推荐系统里其实有两个随机过程

附录 B.2 讲得更直白：传统序列推荐多在做

$$
p(\Phi_i\mid \Phi_0,a_0,\ldots,\Phi_{i-1},a_{i-1}),
$$

但工业系统里更完整的过程是：

1. **系统选择展示内容** $\Phi_i$
2. **用户对内容产生动作** $a_i$

所以一个“生成式推荐”（Generative Recommenders, GR）更应该建模联合分布：

$$
p(\Phi_0,a_0,\Phi_1,a_1,\ldots,\Phi_{n_c-1},a_{n_c-1}).
$$

更具体地，你可以把它因子分解为交替的两步（这是对上式的一种自然展开）：

$$
p(\Phi_{0:n_c-1},a_{0:n_c-1})
=\prod_{i=0}^{n_c-1}
p(\Phi_i\mid \Phi_{<i},a_{<i})\;
p(a_i\mid \Phi_{\le i},a_{<i}).
$$

这也解释了论文在 Table 11 里列出的两个训练任务：

- **Next action token 预测**：学 $p(a_i\mid \text{prefix},\Phi_i)$（对应 ranking）
- **Next content token 预测**：学 $p(\Phi_i\mid \text{prefix})$（更像“学数据分布”，也为未来“直接生成推荐列表”埋伏笔）

## 3. HSTU：面向推荐的高性能自注意力编码器（核心公式）

有了“序列转导 + 生成式”的统一表述，接下来就是：**用什么序列模型既能表达复杂依赖，又能在工业设置里跑得动？**

论文提出的答案是 HSTU，并把一个 HSTU layer 写成 3 个核心公式（Sec. 3, Eq. 1-3）。

设输入为 $X$（一条序列的 token embedding），经过某个变换得到四组张量（每层多头）：

$$
U(X),V(X),Q(X),K(X)=\mathrm{Split}\bigl(\phi_1(f_1(X))\bigr).
$$

其中 $Q,K$ 类似 Transformer 的 query/key（维度 $d_{qk}$），$V$ 是 value（维度 $d_v$）；而 $U$ 是 HSTU 特有的**门控（gating）**张量，用来替代 Transformer 里的 FFN。

注意力聚合写成：

$$
A(X)V(X)=\phi_2\bigl(Q(X)K(X)^{\mathsf{T}}+\mathrm{rab}^{p,t}\bigr)V(X),
$$

其中 $\mathrm{rab}^{p,t}$ 是结合**相对位置**与**时间差**的相对注意力偏置（Appendix A Table 9 给出解释：可对 $(t_j-t_i)$ 做 bucketization 后查表）。

最后输出：

$$
Y(X)=f_2\Bigl(\mathrm{Norm}\bigl(A(X)V(X)\bigr)\odot U(X)\Bigr).
$$

这里 $\odot$ 是逐元素乘；$\mathrm{Norm}$ 是归一化（文中强调 pointwise pooling 后需要 layer norm 稳定训练）；$f_2$ 是输出线性/MLP（并且大量融合到单算子里以省激活内存）。

### 3.1 “Pointwise aggregated attention”：为什么不用 softmax？

标准 Transformer 的注意力是：

$$
\mathrm{softmax}(QK^\mathsf{T})V,
$$

它对每个 query 的注意力权重会在整行上归一化为 1。论文认为在推荐里有两个问题（Sec. 3.1）：

1. **强度信息（intensity）很重要**：同一兴趣信号出现次数越多，往往代表偏好越强；softmax 归一化会“抹平”这种“出现次数越多→总权重越大”的信息。
2. **流式训练下 vocab 非平稳**：新内容不断出现，softmax 的竞争归一化在噪声与非平稳下不一定是好归纳偏置。

因此 HSTU 用 $\phi_2(\cdot)$ 替代 softmax，做“逐点激活 + 聚合 + 再归一化”，核心效果是：

> 输出更像 $\sum_j \phi_2(q\cdot k_j + \text{bias}_{j})\,v_j$，当“相关历史”出现得更多时，聚合值的量级可以变大，从而更容易表达偏好强度。

论文用 Dirichlet Process 的合成流式数据做验证（Table 2）：即使把 HSTU 的相对偏置去掉，仅替换 softmax，也能显著提升 HR@10/50（Transformers: 0.0442/0.2025；HSTU(-rab, Softmax): 0.0617/0.2496；HSTU(-rab): 0.0893/0.3170）。

## 4. 让它能在工业里 scale：长序列、低成本训练与在线推理

HSTU 真正的难点不在“能不能表达”，而在“**长序列 + streaming + 多候选在线**”下能不能跑得动。论文在 Sec. 3 给了三套关键手段。

### 4.1 利用并放大稀疏性：高效 ragged attention + Stochastic Length（SL）

推荐里用户历史长度是长尾分布：有人只有几十条，有人有上万条。HSTU 做了两步：

1. **GPU 上的 ragged attention kernel**：针对 variable-length 序列做 fully raggified 的注意力计算，把它转成 grouped GEMM（Sec. 3.2），并指出在内存访问量上随

$$
\Theta\!\Bigl(\sum_i n_i^2 d_{qk}^2 R^{-1}\Bigr)
$$

增长（$n_i$ 为样本长度，$R$ 为寄存器大小）。

2. **算法性增加稀疏：Stochastic Length（SL）**  
核心思想：用户行为在时间上往往“重复”（多尺度），所以对超长序列可以随机采一个较短的子序列训练，大幅降成本而不明显伤精度。

论文给出的 SL 选择规则（Sec. 3.2, Eq. 4）可以整理成更好读的形式。设用户 $j$ 的内容交互长度为 $n_{c,j}$，全局最大长度为 $N_c=\max_j n_{c,j}$，$\alpha\in(1,2]$ 控制稀疏程度，则：

$$
\text{SL}(j)=
\begin{cases}
\text{保留全序列 }(x_i)_{i=0}^{n_{c,j}}, & n_{c,j}\le N_c^{\alpha/2} \\\\[4pt]
\text{采样子序列 }(x_{i_k})_{k=0}^{N_c^{\alpha/2}}, & n_{c,j}> N_c^{\alpha/2}\ \text{且以概率 }1-\frac{N_c^\alpha}{n_{c,j}^2} \\\\[6pt]
\text{保留全序列 }(x_i)_{i=0}^{n_{c,j}}, & n_{c,j}> N_c^{\alpha/2}\ \text{且以概率 }\frac{N_c^\alpha}{n_{c,j}^2}
\end{cases}
$$

它把注意力相关复杂度从 $O(N_c^2)$ 降到期望意义下的 $O(N_c^\alpha)$。

一个非常直观的例子：当 $N_c=4096$、$\alpha=1.6$ 时，

$$
N_c^{\alpha/2}=4096^{0.8}=2^{12\times 0.8}=2^{9.6}\approx 776,
$$

也就是论文里提到的“4096 的序列多数时候会被变成 776”（相当于移除 80%+ token）。Table 3 也显示在长序列（8192）时，$\alpha=1.6$ 的稀疏率可到 84%。

此外，论文在实验中观察到：即使稀疏率提高到 64%–84%，主任务 NE 退化不超过 0.002（Sec. 4.2）。

### 4.2 降激活内存：用门控替代 FFN + 强融合

推荐训练常用大 batch，激活内存往往比参数更成瓶颈。论文指出 HSTU 两个设计让激活更省（Sec. 3.3）：

- 把 Transformer 层里“注意力外的多层线性/FFN”大幅简化（从 6 个线性层降到 2 个），用 $U(X)$ 的 gating 与 $f_2(\cdot)$ 替代 FFN
- 把 $\phi_1(f_1(\cdot))$、LayerNorm、dropout、输出 MLP 等尽可能融合成单算子

他们估算 bfloat16 下，HSTU 每层激活状态大约是 $14d$，而 Transformer 在常见假设下约为 $33d$，从而能支撑更深网络（>2×）。

### 4.3 多候选在线推理：M-FALCON 把“对每个候选做一次”摊薄

排序线上会有上万候选。若你用目标感知的 cross attention，对每个候选都算一次，代价会是 $O(mn^2d)$。

论文提出 **M-FALCON**（Sec. 3.4）：在一次 forward 里把 $b_m$ 个候选打包，通过修改 attention mask 与 $\mathrm{rab}^{p,t}$，让这 $b_m$ 个候选共享相同的注意力计算，从而把代价从

$$
O(b_m n^2 d)
$$

降为

$$
O((n+b_m)^2 d)\approx O(n^2 d)\quad (b_m\ll n).
$$

再配合 microbatch 与 KV cache，可以把复杂度随候选数近似线性扩展，同时控制尾延迟。论文在工业最挑战的 ranking 配置下，宣称用 285× 更复杂的目标感知模型仍能做到 **1.5×–3× 更高吞吐**（Sec. 4.3）。

## 5. 实验效果：从合成数据到工业线上

### 5.1 合成流式数据：pointwise attention 明显优于 softmax（Table 2）

- Transformers：HR@10/50 = 0.0442 / 0.2025
- HSTU(-rab, Softmax)：0.0617 / 0.2496
- HSTU(-rab)：0.0893 / 0.3170

说明“去 softmax、保留强度信息”在非平稳流式场景很关键。

### 5.2 公共序列推荐数据集：同等配置下全面超过 SASRec（Table 4）

以 ML-1M 为例：

- SASRec：HR@10 0.2853
- HSTU：0.3097（+8.6%）
- HSTU-large：0.3294（+15.5%）

在 Books 这种更稀疏数据上提升更夸张：HR@10 从 0.0292 提到 0.0404（+38.4%）甚至 0.0469（+60.6%）。

### 5.3 工业流式设置：Transformer 排序不稳定，HSTU 更好更快（Table 5）

论文用工业流式数据训练 100B examples（DLRM 等价），并用 NE 评估排序任务（0.001 的 NE 降低被视作显著）。

在小规模固定 encoder 配置下（l/n/d 给定）：

- Transformers：检索 log pplx 4.069；排序出现 NaN（训练不稳定）
- HSTU（最好版本）：检索 log pplx 3.978；排序 NE = 0.4937（E-Task）、0.7805（C-Task）

并且作者称在该设置下 HSTU 可做到 **1.5×–2× 更快 wall-clock**、**HBM 省 50%**。

### 5.4 与工业 DLRM 的端到端对比：离线更好，线上 A/B 大幅提升（Table 6/7）

**召回（Table 6）**：

- DLRM：HR@100/500 = 29.0% / 55.5%
- GR（new source）：36.9% / 62.4%，并带来线上 **+6.2% / +5.0%**（E/C 任务）
- 只用交互序列（GR interactions only）虽然离线也不差（35.6% / 61.7%），但论文认为忽略其他特征会伤最终效果
- 纯内容（content-based）很差（11.6% / 18.8%），强调“高基数用户动作”在推荐里比内容更关键

**排序（Table 7）**：

- DLRM：NE = 0.4982（E）/ 0.7842（C）
- GR：NE = 0.4845（E）/ 0.7645（C），线上 **+12.4% / +4.4%** uplift

### 5.5 SL vs 其他长度外推：几乎不掉点（Table 16）

在 2048（52% sparsity）与 4096（75% sparsity）两档上，SL 的平均 NE 相对全序列基线差异只有：

- 0.098%（2048）
- 0.64%（4096）

而 RoPE 等长度外推方法在 zero-shot / finetune 下的差异要大一个数量级（6%–11% / ~2%）。

## 6. 我认为最重要的启示

1. **“Actions speak louder than words”不是口号**：推荐里最强的信号往往是高基数行为动作（点击/完播/跳过/分享……），把它们当成 token 放进统一序列，能让模型用更统一的方式“吃进原始信号”。
2. **生成式推荐的关键不是“像 NLP 一样输出句子”，而是建模联合过程**：系统选择内容 + 用户产生动作，这个联合分布一旦写清楚，ranking 与 retrieval 反而更像“同一个问题的不同切片”。
3. **点注意力形式要服务于推荐的归纳偏置**：softmax 的“总量=1”不一定适合表达偏好强度；pointwise 聚合 + 归一化能更自然地表达“相关历史出现越多→信号越强”。
4. **想要推荐的 scaling law，必须同时 scale 序列长度**：论文明确指出相比语言建模，GR 更依赖长序列，长度与宽度/深度要一起扩。

如果你正在做工业推荐系统，这篇论文值得反复读的部分不是“某个 trick”，而是一整套“从任务表述到模型与系统设计”的连贯路线：**先把问题写成能 scale 的形式，再为这个形式设计能 scale 的算子与训练/推理算法**。

