---
title: "GPR 论文解读：广告推荐的“生成式一体化”范式，如何用一个模型替代级联系统？"
description: "解读 GPR（arXiv:2511.10138v2）：用统一的 U/O/E/I Token 输入与多级语义 ID（RQ-Kmeans+）把广告与内容对齐；用 HHD（HSD+PTD+HTE）做“理解→生成→估值”的层级解码；再用 MTP+VAFT+HEPO（含层级过程奖励与 PPO）完成从预训练到价值对齐与策略优化。"
pubDate: 2026-01-13
tags: ["论文解读", "推荐系统", "广告推荐", "生成式推荐", "RLHF", "PPO", "Beam Search"]
category: "论文解读"
draft: false
---

> 论文：GPR: Towards a Generative Pre-trained One-Model Paradigm for Large-Scale Advertising Recommendation（arXiv:2511.10138v2）  
> 链接：<https://arxiv.org/abs/2511.10138>

广告推荐系统的传统形态，是一个“非常工程化”的级联：召回 → 粗排 → 精排 → 重排/预算/风控 → 拍卖。  
每一层都有自己的模型、特征、目标函数和策略。这个体系能跑起来，但也带来两个长期痛点：

1. **目标不一致（objective misalignment）**：上游在优化某个 proxy（比如召回命中、CTR），下游在优化 eCPM/GMV；各层之间并非在同一个目标上做全局最优。
2. **误差传播（error propagation）**：上游漏掉的候选，下游再强也救不回来；上游分布漂移还会让下游训练/推理分布不一致。

这几年“生成式推荐（Generative Rec）”开始尝试用一个生成模型统一召回与排序，但在工业广告场景里又会撞上更硬的现实约束：**超长用户历史、强异构特征、强业务约束（定向/预算/合规）、低延迟高 QPS、以及“价值”比“相关性”更复杂**。

这篇论文提出 **GPR（Generative Pre-trained Recommender）**：把广告推荐重定义为端到端生成任务，并给出一套从表示、结构、训练到推理的完整落地路线，最终在腾讯视频号广告系统上线，带来 GMV、CTCVR 等指标提升（Table 4/5）。

下面按 **问题 → 表示 → 模型 → 推理 → 训练（含公式）→ 实验** 逐层拆开。

## 1. 先把输入“写成生成模型能吃的样子”：U/O/E/I Token + 多级语义 ID

广告平台的数据既“杂”又“长”：用户既刷自然内容（短视频/文章），也会看到广告并产生点击/转化；同时还有请求上下文（位置、时间、场景）与用户画像。论文把用户全旅程统一序列化为四类 token（Sec. 2.1）：

- **U-Token**：用户属性与偏好（profile / long-term）
- **O-Token**：用户自然内容行为（organic content，如短视频/图文）
- **E-Token**：一次广告请求的环境上下文（environment，如场景、位置等）
- **I-Token**：用户与广告交互的 item（ad item）

其中 O-Token 与 I-Token 的“内容本体”会被离散化为 **多级语义 ID**（coarse→fine），使得“推荐生成”变成“生成离散 token 序列”，更贴近语言模型范式。

### 1.1 为什么要把 item embedding 量化成离散语义 ID？

直觉上有两个收益：

1. **离散化让生成更自然**：生成模型擅长输出 token；直接生成连续 embedding 往往不稳定，还需要额外检索/量化步骤。
2. **多级 ID 提供层级结构**：可以把一个 item 表示成 $L$ 级路径 $(z_1,z_2,\dots,z_L)$，类似“类目→子类→簇→具体 item”。这对后续“层级解码 + 层级策略优化”很关键。

### 1.2 RQ-Kmeans+：解决 codebook collapse，让语义空间真的“用起来”

论文指出常见的 RQ-VAE / RQ-Kmeans 量化会出现两个问题：**codebook collapse（死码）**与 **latent space 不够鲁棒**，导致语义空间利用率低、表达力受限。

他们提出 **RQ-Kmeans+**（Fig. 3）：先用 RQ-Kmeans 得到更好的 codebook 初始化，再用 VAE 风格的损失去更新 codebook 适配可学习 latent；并在 encoder 侧加残差，稳定早期训练、加速收敛。

为了衡量“量化质量”，论文给了三个指标（Sec. 4.1）：

- **Collision Rate**：不同 item 映射到同一 code 的比例（越低越好）
- **CUR\_{L1}（level-1 Code Usage Rate）**：第 1 层活跃 code 占比（越高越好，代表不“死码”）
- **PAS（Path Average Similarity）**：共享 code 的 item embedding 相似度均值（越高越好，代表“碰撞也更像”）

Table 1 的结论很清晰：RQ-Kmeans+ 在不牺牲 code 使用率的前提下降低碰撞，并让剩余碰撞更“语义一致”（PAS 更高）。

## 2. 一个模型怎么同时“懂用户、会生成、还能对齐价值”？HHD = HSD + PTD + HTE

GPR 的主干是一个 decoder-only 的生成式架构 **HHD（Heterogeneous Hierarchical Decoder）**（Sec. 2.2），由三块组成：

- **HSD（Heterogeneous Sequence-wise Decoder）**：吃下超长异构序列，产出“意图表征”（intent embeddings）
- **PTD（Progressive Token-wise Decoder）**：在意图表征条件下，逐级生成下一条广告的多级语义 ID
- **HTE（Hierarchical Token-wise Evaluator）**：对每一级 token / 最终 item 估值（final_value），用于剪枝、拍卖与 RL critic

你可以把它类比为：**先“读懂上下文”（HSD），再“写出答案”（PTD），最后“给答案打分”（HTE）**。

### 2.1 HSD：改造 HSTU 的“混合注意力”与 token-aware 模块

HSD 基于 HSTU block（HSTU 是上一篇论文里的高性能注意力编码器），但为了适配“prompt + 生成”的结构，论文做了三类增强（Sec. 2.2）：

1. **Hybrid Attention**：对 prefix（U/O/E）放开因果约束，允许双向注意力
2. **Token-Aware Norm/FFN**：不同 token 类型用不同 LN/FFN，减少异构干扰
3. **MoR + 外部知识**：Mixture-of-Recursions 增加有效深度；再注入 LLM 生成的“思考文本”增强语义推理

#### (1) Hybrid Attention 的公式与直觉

论文把 HSD 的注意力写成（Eq. 1）：

$$
\mathtt{HybridAttn}(\cdot)=
\mathrm{Softmax}\!\left(\frac{QK^{\top}}{\sqrt{d}}+M^{\text{hybrid}}\right)V\odot U
$$

其中 $U$ 是额外的门控 embedding，用来调制注意力输出（继承自 HSTU 的“用门控增强表达/抑制噪声”的思路）。

关键是 $M^{\text{hybrid}}$：它不是纯 causal mask，而是“**因果 + prefix 双向**”。论文原文给了一个分段定义（Eq. 2 的 HTML 渲染不完整），但思想可以写成一个更好读的版本：

设 $X_i$ 表示第 $i$ 个 token 的类型，prefix 集合 $\mathcal{P}=\{\text{U-Token, O-Token, E-Token}\}$，则

$$
M^{\text{hybrid}}_{ij}=
\begin{cases}
0, & j\le i \ \ \text{（标准因果：看过去）} \\\\
0, & X_i\in\mathcal{P}\ \text{且}\ X_j\in\mathcal{P}\ \ \text{（prefix 内双向）} \\\\
-\infty, & \text{其他情况}
\end{cases}
$$

直觉：U/O/E 更像“提示词”，它们在推理时是已知的，不需要像生成 token 一样严格因果；让它们彼此双向交互，可以构造更强的上下文表征，然后再去生成/预测广告语义 ID。

#### (2) Token-Aware LN/FFN：为什么要“分开算”？

U/O/E/I 这四类 token 的统计分布与语义空间差异很大。如果所有 token 共享同一套 LN/FFN，容易出现“某一类 token（比如高频 O-token）主导更新、压制其他 token”的问题。

论文做法是：**每种 token 类型有独立的归一化与 FFN 子网络**，把它们投影到各自语义子空间，再通过注意力/残差融合。

### 2.2 PTD：Thinking-Refining-Generation，把“生成路径”拆成可控步骤

HSD 给出的 intent embeddings 信息量很大，但也可能冗余。PTD（Sec. 2.2）用一个二级 decoder 在其上做层级生成，并显式加入推理过程：

1. **Thinking**：先生成 $K$ 个 thinking tokens，从 intent embeddings 中提炼关键信息
2. **Refining**：用扩散（diffusion）范式做“去噪反推”，进一步精炼 thinking 的结果（条件是对 thinking tokens 做 `Sum_Pooling`）
3. **Generation**：基于 thinking + refining token，逐级生成多级语义 ID $(z_1,\dots,z_L)$

这里最值得注意的是：他们把“生成前的隐式推理”显式化了，这在广告场景里可能有用——因为很多约束不是“相似度”就能解决的（比如定向、预算、曝光频控），需要更强的条件推理能力。

### 2.3 HTE：把多目标（CTR/CVR/eCPM…）聚合成 final_value，并用于剪枝与 RL critic

广告不是只看相关性，还要看商业价值。论文把 CTR/CVR/eCPM 等多指标聚合成一个标量 **final_value**（Sec. 2.2），作为全链路优化目标，并在 RL 段给出形式（Eq. 5）：

$$
R=\texttt{final\_value}(s,\{z_{\ell}\}_{\ell=1}^{L})
=\mathrm{eCPM}(s,\{z_{\ell}\})
\frac{1}{N}\sum_{i=1}^{N}\alpha_i\,\mathrm{target}_i(s,\{z_{\ell}\}).
$$

HTE 在推理时为“每一级语义 code / 最终 item”估值，用于：

- **生成时早剪枝**（价值引导 beam search）
- **最终拍卖排序**（value-aware）
- **作为 RL critic**（HEPO 的 $V_\phi$）

## 3. 推理：Value-Guided Trie Beam Search，把“业务约束”塞进解码过程

直接做 beam search + 后过滤（预算/定向/无效 item）+ 后排序（按价值）会非常慢。

论文提出 **Value-Guided Trie-Based Beam Search**（Sec. 2.3）做两件事：

1. **Trie 约束**：把用户定向策略（年龄/性别等）构造成一棵 Trie，只允许解码到“对该用户合法”的路径上，实现早期过滤。
2. **价值引导动态 beam**：利用 HTE 对每个语义 code 的预测 value，动态调整 beam width：价值越高，保留越多分支，让搜索预算倾向高潜力路径。

直觉上就是：把“候选是否合法”和“候选值不值钱”提前到解码阶段做，而不是生成完再补救。

## 4. 训练：从“生成预训练”到“价值对齐”再到“层级策略优化”（重点公式）

GPR 的训练是三段式（Sec. 3）：

1. **MTP 预训练**：解决广告数据稀疏 + 多兴趣并行
2. **VAFT 微调**：把 MTP 的学习重心对齐到 eCPM/动作价值
3. **HEPO 后训练（RL）**：在高保真仿真环境里做策略优化，并解决层级解码的 credit assignment

### 4.1 MTP：Multi-Token Prediction，用多个 head 同时刻画多兴趣（Eq. 3）

广告场景下用户兴趣经常“多线程”：既可能对美妆感兴趣，也可能在看数码、又可能在关注育儿。单 head 的 next-token prediction 容易把这些兴趣平均掉。

论文用 **$N$ 个并行 head**（默认 $N=4$）来同时预测 $N$ 条语义路径，每个 head 生成完整的 $L$ 级 code 序列；并用 head 权重 $\omega_j^H$（满足 $\sum_j \omega_j^H=1$）聚合损失（Eq. 3）：

$$
L_{\text{MTP}}
=-\sum_{j=1}^{N}\sum_{t=1}^{T}\sum_{\ell=1}^{L}
\omega_{j}^{H}\cdot\log P_{j}\!\left(I_{j,t,\ell}\mid S,C,I_{j,t,1:\ell-1}\right).
$$

含义拆开看：

- $t$ 是序列位置，$\ell$ 是语义层级（从 coarse 到 fine）
- $P_j(\cdot)$ 是在“该层合法候选集合”上的条件概率（masked decoding）
- $\omega_j^H$ 让模型可以“更信任”更高质量的兴趣线程

### 4.2 VAFT：把“动作层级 + eCPM”写进 loss 权重（Eq. 4）

MTP 仍然是似然训练：它默认每个样本贡献差不多的梯度。但广告里不同曝光的商业价值差很大，而且动作价值有明显层级：**转化 > 点击 > 曝光**。

于是 VAFT 引入位置权重 $\omega_{j,t}^{V}$，把动作类型与 eCPM 融合进 loss（Eq. 4）：

$$
L_{\text{eCPM-MTP}}
=-\sum_{j=1}^{N}\sum_{t=1}^{T}\sum_{\ell=1}^{L}
\big(\omega_j^H\,\omega_{j,t}^{V}\big)\,
\log P_j\!\big(I_{j,t,\ell}\mid S,C,I_{j,t,1:\ell-1}\big).
$$

论文还给了一个很实用的“动作价值归一化”写法（Sec. 3.2）：用不同分母让三类动作落在可比尺度上，例如

$$
\omega^V \propto
\begin{cases}
\mathrm{eCPM}, & \text{impression} \\\\
\frac{\mathrm{eCPM}}{\text{pCTR}}, & \text{click} \\\\
\frac{\mathrm{eCPM}}{\text{pCTR}\cdot \text{pCVR}}, & \text{conversion}
\end{cases}
$$

直觉：转化动作稀缺且价值高，如果不做尺度校正，要么梯度被长尾曝光淹没，要么被极少数高 eCPM outlier 扯飞。

### 4.3 HEPO：Hierarchical Enhanced Policy Optimization（Eq. 5–10）

仅靠监督学习，策略只能模仿历史曝光；但历史策略并不保证全局最优，且无法覆盖“没被系统展示过”的反事实候选。

论文用 RL，在一个高保真的仿真环境中让策略探索与评估。关键是：这里的“动作”是**层级的**——模型要按层级依次选出 $z_1,z_2,\dots,z_L$，最终落到具体广告。

#### (1) 终止奖励：仿真环境计算 final_value（Eq. 5）

仿真环境用生产的 pCTR/pCVR 排序模型评估每个候选生成序列，得到终止奖励 $R$（上面已给）。

#### (2) 难点：层级信用分配（credit assignment）

如果只在最后一步给奖励，中间层（比如类目层、簇层）学习信号会很弱、方差很大。

论文给了一个非常直观的反例（Sec. 3.3）：

> 用户喜欢“智能手机”，但不喜欢“品牌 A”。如果策略生成了“手机→品牌 A→某款机型”而被拒绝，只有终止负奖励会把“手机”这个正确的粗粒度偏好也一起打压，导致上层决策学歪。

#### (3) 过程奖励：用“用户历史成功偏好”给每一层直接反馈（Eq. 6–7）

对每一层 $\ell$，从用户历史正反馈里统计一个 token 流行度/偏好分数 $P_\ell(t)\in[0,1]$。对当前选择的 $z_\ell$，与该层合法集合 $\mathcal{S}_\ell$ 的均值做差（Eq. 6）：

$$
\Delta_\ell
=P_\ell(z_\ell)
-\frac{1}{|\mathcal{S}_\ell|}\sum_{t\in\mathcal{S}_\ell}P_\ell(t).
$$

再得到层级步进奖励（Eq. 7）：

$$
r_\ell=
\begin{cases}
\alpha_\ell\max(0,\Delta_\ell), & \ell<L \\\\
R, & \ell=L
\end{cases}
$$

其中 $\alpha_\ell$ 是小系数，确保过程奖励“指路”但不压过终止商业奖励。

#### (4) Advantage：粗层用 GAE，终层用候选内 z-score 稳定训练（Eq. 8）

粗层 $\ell<L$ 用 GAE：

- TD error：$\delta_\ell=r_\ell+\gamma V_\phi(s,z_{1:\ell})-V_\phi(s,z_{1:\ell-1})$
- Advantage（Eq. 8 上半部分）：

$$
A_\ell=\sum_{l=0}^{L-\ell-1}(\gamma\lambda)^l\delta_{\ell+l}.
$$

终层 $\ell=L$ 的 $R$ 跨请求方差太大，论文改用“同一请求下 $K$ 个候选”的 z-score（Eq. 8 下半部分）：

$$
A_L=\frac{R-\mu_K}{\sigma_K+\epsilon}.
$$

#### (5) PPO 风格的策略目标 + 价值回归（Eq. 9–10）

策略更新目标（Eq. 9）：

$$
\mathcal{L}_\theta
=\mathbb{E}\left[
\sum_{\ell=1}^{L}c_\ell
\min\Big(
\rho_\ell A_\ell,\;
\text{clip}(\rho_\ell,1-\epsilon,1+\epsilon)A_\ell
\Big)\right],
$$

其中重要性采样比率

$$
\rho_\ell=\frac{\pi_\theta(z_\ell)}{\pi_{\theta_{\text{old}}}(z_\ell)}.
$$

价值函数（HTE/critic）用 MSE 拟合各层 return（Eq. 10）：

$$
\mathcal{L}_\phi
=\mathbb{E}\left[\sum_{\ell=1}^{L}\big(V_\phi(s,z_{1:\ell-1})-G_\ell\big)^2\right].
$$

### 4.4 ARR：Anticipatory Request Rehearsal，让策略“提前适应”明天的流量

广告生态变化快：用户兴趣、广告素材、预算都在日更。论文提出 ARR 生成“面向未来”的合成请求样本（Sec. 3.3）：

- 用用户最近自然内容重建 O-Token
- 用实时系统状态更新 E-Token
- 按用户活跃度设定不同的合成频率（高活跃 2h/4h；低活跃更稀疏）

并把这些合成请求走同样的仿真评估与 HEPO 更新流程，使策略不至于只“追着历史跑”。

## 5. 实验：离线消融、对齐指标、线上 A/B（关键数字）

### 5.1 Tokenization：RQ-Kmeans+ 的语义空间更可用（Table 1）

Table 1 显示 RQ-Kmeans+：

- Collision：20.60%（最低）
- CUR\_{L1}：99.36%（接近饱和）
- PAS：0.992（碰撞更“像”）

这意味着：即使存在碰撞，也更可能是“同类 item 的合理合并”，对生成式推荐更友好。

### 5.2 用户行为建模：HHD 全量提升 HitR@100（Table 2）

在“从百万级 catalog 里生成 top-100 命中”的任务上：

- HSTU（decoder-only）：18.98
- OneRec（encoder-decoder）：19.85（+4.6%）
- **GPR Full**（HSD+PTD+HTE + 训练组件）：**27.32（+43.9%）**

并且 Table 2 的消融也解释了每个组件为何存在：Hybrid Attention、Token-Aware FFN/LN、MTP、Thinking、HTE 等都能贡献可观增益。

### 5.3 价值对齐：VAFT + HEPO 在 nDCG/OPR/final_value 上持续提升（Table 3）

从 MTP → VAFT → DPO → HEPO：

- nDCG：0.3868 → 0.3925 → 0.4383 → 0.4413
- OPR：0.5292 → 0.5348 → 0.5463 → 0.5509
- avg/max final_value（归一化）：max 从 0.6201 → 0.7619（HEPO）

说明“把价值写进 loss”能带来收益，但“在仿真里做层级 RL 并解决 credit assignment”能进一步拉开差距。

### 5.4 线上 A/B：多轮增量上线，GMV 与 CTCVR 均提升（Table 4/5）

Table 4 给出 5 次连续线上评估（每次在上一版本基础上增量）：

- v0.1（HSD+NTP+DPO）：GMV +2.11%
- v0.2（+HEPO w/o ARR）：GMV +0.70%
- v0.3（+MTP+Thinking）：GMV +0.63%
- v0.4（+PTD）：GMV +0.71%
- v0.5（+HEPO w/ ARR）：GMV +0.58%

Table 5 的分层分析也说明收益更“广谱”：低活跃用户组 UG1/UG2 提升更明显；新广告（new）提升也更大（CTCVR +4.02%）。

## 6. 总结：GPR 最值得记住的三个“方法论”

1. **先统一表示，再谈统一模型**：U/O/E/I schema + 多级语义 ID 让广告与自然内容进入同一 token 空间，这是“一个模型”成立的前提。
2. **生成不是终点，价值对齐才是广告的核心**：HTE 把多目标聚合成 final_value，并贯穿推理剪枝与 RL critic。
3. **层级生成要配套层级信用分配**：HEPO 用过程奖励 + GAE + PPO，把“类目/簇/细粒度”每一层都训练得更稳定、更贴合用户偏好，同时把最终目标留给商业奖励。

如果你在做工业广告推荐，这篇论文的贡献不仅是“一个新模型”，更像是一套完整的“生成式一体化落地手册”：从 tokenization 到解码，再到对齐与上线验证，链条是闭环的。

