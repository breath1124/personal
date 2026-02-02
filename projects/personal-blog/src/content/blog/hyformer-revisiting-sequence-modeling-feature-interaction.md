---
title: "HyFormer 论文解读：重新理解长序列建模与特征交互在 CTR 里的分工"
description: "解读 HyFormer（arXiv:2601.12681v2）：用全局查询 token 把长序列建模与异构特征交互统一到单一骨干；通过 Query Decoding/Query Boosting 交替优化实现双向信息流；并给出与 LONGER、RankMixer、MTGR/OneTrans 等方法的系统对比与实验结果。"
pubDate: 2026-01-18
tags: ["论文解读", "推荐系统", "CTR 预测", "长序列", "特征交互", "HyFormer", "Douyin", "ByteDance"]
category: "论文解读"
draft: false
---

> 论文：HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction（arXiv:2601.12681v2）  
> 链接：<https://arxiv.org/abs/2601.12681>（HTML：<https://arxiv.org/html/2601.12681v2>）

这篇论文来自字节跳动，解决的问题看起来“老”，但掰开看非常现实：**工业级 CTR 系统里，长序列建模（sequence modeling）和异构特征交互（feature interaction）到底应该怎么协同？**

传统做法是“先压缩长序列，再做特征交互”，典型流水线就是 **LONGER → RankMixer**。这种分工清晰，但副作用也明显：**长序列被过早压缩，后续特征交互无法反过来影响序列建模**。HyFormer 的主张非常直接：**把两者放进一个统一的骨干里，形成“交替优化”的双向信息流**。

下面按“问题 → 方法 → 公式 → 对比 → 实验 → 启示”来拆。

## 1. 问题背景：CTR 在工业场景为什么难？

CTR 预测的基本形式很简单：

$$
P(y=1\mid S,u,v)\in[0,1],
$$

训练目标是标准的二分类对数损失：

$$
\mathcal{L}=-\frac{1}{|\mathcal{D}|}\sum_{(S,u,v,y)\in\mathcal{D}}\Big[y\log\hat{y}+(1-y)\log(1-\hat{y})\Big],
$$

其中 $S$ 是用户行为序列，$u$ 是用户/查询特征，$v$ 是候选内容特征。

难点在于两点同时存在：

1. **序列很长**：长序列里包含“长期兴趣”和“冷门偏好”，但直接用 Transformer 做全自注意力成本太高。
2. **特征很杂**：非序列特征（用户属性、query、文档属性、交叉特征等）维度高、类型多，需要强交互能力。

工业系统里经常采用“先序列建模、后特征交互”的流水线，但它天然带来信息流的单向性：**序列压缩后就定型了，后面的交互再强也无法补救早期的信息损失。**

HyFormer 的核心就是针对这个“结构性瓶颈”。

## 2. 现有方法的两大范式

论文把基线分成两类，我们也沿着这个框架理解：

### 2.1 传统两阶段模型（Two-Stage）

- **序列建模**：LONGER、Full Transformer
- **特征交互**：RankMixer、Full Transformer、Wukong

特点：先压缩序列，再在压缩后的 query token 上做特征交互。

问题：

- **序列建模与特征交互割裂**，交互模块无法反向影响序列表示；
- query token 通常“太窄”，信息容量受限；
- 为了效率，序列压缩往往过早、过狠。

### 2.2 统一架构（Unified-Block）

代表是 **MTGR / OneTrans**：把序列和非序列都 token 化后，放进一个 Transformer 式的统一模块。

问题：

- 自注意力在特征交互阶段成本高；
- query token 数量增加会明显拖垮在线效率；
- 统一 block 里 sequence modeling 与 feature interaction 被“绑死”，缺少灵活性。

HyFormer 的目标不是简单“更统一”，而是**让序列建模与特征交互双向协作，同时保持可控的计算成本**。

## 3. HyFormer 的核心理念：全局查询 + 交替优化

HyFormer 把模型看作一个交替优化过程，包含两个模块：

- **Query Decoding**：用“全局查询 token”去解码长序列的 key/value 表示；
- **Query Boosting**：用轻量 token mixing 强化查询 token 与非序列特征的交互。

核心关键词是 **Global Tokens**：它们是连接长序列与非序列特征的“语义接口”。

### 3.1 Query Generation：全局查询从哪里来？

HyFormer 先把非序列特征和序列全局摘要拼成 Global Info：

$$
\mathrm{Global\ Info}=\mathrm{Concat}\big(F_{1},\ldots,F_{M},\;\mathrm{MeanPool}(Seq)\big).
$$

然后用多个轻量 FFN 生成 $N$ 个全局查询：

$$
Q=\big[\mathrm{FFN}_{1}(\mathrm{Global\ Info}),\ldots,\mathrm{FFN}_{N}(\mathrm{Global\ Info})\big]\in\mathbb{R}^{N\times D}.
$$

直观理解：

- **非序列特征**提供“静态语义”（用户/内容/上下文）；
- **MeanPool(Seq)** 提供“序列全局概览”；
- 每个 FFN 产生一个“语义角度不同”的查询 token。

为了在线效率，HyFormer还支持 query 压缩，保持 token 数量稳定；在深层不再重新生成，而是复用前一层的 query，使其逐层增强语义。

这一段背后还有一个细节：**输入 token 的组织方式**。论文比较了“语义分组”和“自动拆分”，最终选择语义分组（例如用户、上下文、候选内容、交叉特征等）。原因很朴素：推荐特征本身具有强语义结构，如果一股脑 flatten 再切片，会让模型丢掉结构性归纳偏置。HyFormer 保留这种结构，等于提前把“谁是谁的特征”告诉模型，让 query 的生成更加可解释、也更稳定。

可以把这些 query tokens 想象成“多视角提问”。同一份 Global Info 同时喂给多个 FFN，每个 FFN 形成一组不同的 query，等于从不同角度去“问”长序列：有的偏长期兴趣，有的偏短期语境，有的更关注 query 意图。这也是 HyFormer 能在不暴涨 token 数的情况下提升 query 容量的关键。

### 3.2 Sequence Representation Encoding：序列如何编码？

HyFormer允许不同序列编码策略，覆盖性能-效率的折中：

1. **Full Transformer**（最强表达）：

$$
H_{l}=\mathrm{TransformerEnc}_{l}(S),
$$

2. **LONGER 风格高效编码**（用短序列 query 对长序列做 cross-attn）：

$$
H_{l}=\mathrm{CrossAttn}(S_{\text{short}},\;S,\;S),
$$

将复杂度从 $\mathcal{O}(L_{S}^{2})$ 降到 $\mathcal{O}(L_{H}L_{S})$。

3. **Decoder-style 轻量编码**（高延迟场景）：

$$
H_{l}=\mathrm{SwiGLU}_{l}(S),
$$

几乎不含注意力，代价更小。

无论哪种方式，都会生成每层的 key/value：

$$
K_{l}=H_{l}W^{K}_{l},\qquad V_{l}=H_{l}W^{V}_{l}.
$$

### 3.3 Query Decoding：让全局 token 直接“读序列”

有了 $K_l, V_l$，Query Decoding 用 cross-attention 解码：

$$
\tilde{Q}_{(l)}=\mathrm{CrossAttn}\!\left(Q_{(l)},\,K_{(l)},\,V_{(l)}\right),
$$

直观理解：**全局查询 token 把“序列信息”吸收进来**，得到“带序列语义的 query 表示”。

这一步把“序列建模”与“非序列语义”在 query 层面融合起来，而不是让序列先单独压缩成一个固定表示。

### 3.4 Query Boosting：让 query 和非序列特征充分交互

解码得到的 query 还需要和非序列特征做更深交互，于是有 Query Boosting：

先拼成统一 token 序列：

$$
Q=[\tilde{Q}_{(l)},F_{1},\ldots,F_{M}]\in\mathbb{R}^{T\times D}.
$$

然后用 RankMixer / MLP-Mixer 风格的 token mixing：

1. 每个 token 分块：

$$
q_{t}=[\,q_{t}^{(1)}\|q_{t}^{(2)}\|\cdots\|q_{t}^{(T)}\,],\quad q_{t}^{(h)}\in\mathbb{R}^{D/T}.
$$

2. 对每个 subspace 做“跨 token 拼接”：

$$
\tilde{q}_{h}=\mathrm{Concat}\big(q_{1}^{(h)},q_{2}^{(h)},\ldots,q_{T}^{(h)}\big)\in\mathbb{R}^{D}.
$$

3. 聚合得到混合表示：

$$
\hat{Q}=[\tilde{q}_{1},\tilde{q}_{2},\ldots,\tilde{q}_{T}]\in\mathbb{R}^{T\times D}.
$$

4. 再经过 PerToken-FFN 并残差：

$$
\widetilde{Q}=\mathrm{PerToken\text{-}FFN}(\hat{Q}),
$$

$$
Q_{\mathrm{boost}}=Q+\widetilde{Q}.
$$

这一步的意义是：**让 query token 不仅“读序列”，还要与非序列特征充分交互，从而提升 query 的语义容量。**

### 3.5 HyFormer 层堆叠：交替优化的“循环”

每一层都是：

1. Query Decoding：

$$
\widehat{Q}^{(l)}=\mathrm{CrossAttn}\big(Q^{(l-1)},K^{(l)},V^{(l)}\big).
$$

2. Query Boosting：

$$
\widetilde{Q}^{(l)}=\mathrm{QueryBoost}\big(\mathrm{Concat}(\widehat{Q}^{(l)},\mathrm{NS\ Tokens})\big).
$$

交替执行意味着：**序列信息与非序列信息在每一层都相互影响，而不是只在某一层“交汇一次”。**

### 3.6 为什么不直接增加 query tokens？

很多人第一反应是：既然 query token 太少，那就多加一些不就行了？论文明确指出这条路在工业系统里走不通。

- 查询 token 数量增加会显著放大 cross-attention 成本，尤其在 KV-Cache 与 M-Falcon 机制下会直接拖慢在线延迟；
- MTGR/OneTrans 这类“全 token 化”方法在统一 block 里虽能增加 token 数，但自注意力的 FLOPs 会爆炸；
- 更重要的是，**仅仅增加 token 数并不会改变“信息流单向”的结构问题**，它只是把瓶颈变大而不是打破瓶颈。

HyFormer 的思路是：**不把 token 数量当唯一的解法，而是通过“Query Decoding/Boosting 的交替”让有限的 token 在每层都能吸收新信息**，用结构换容量。

### 3.7 结构直觉：两条“信息循环”

如果把 HyFormer 的信息流画成两条循环，会更直观：

1. **序列 → Query**：Query Decoding 让全局 token 反复“读序列”，把长序列的细节逐层吸进 query。
2. **Query → 非序列特征**：Query Boosting 让全局 token 与非序列特征做深度交互，再把增强后的 query 送回下一层的序列解码。

这就像两个人对话：一个负责“读材料”（序列），一个负责“理解语境”（非序列特征），他们每一轮交流都会让语义更完整。与传统 pipeline 最大的区别是：**信息在每一层都有机会重塑，而不是在第一步就被压缩定型。**
## 4. 多序列建模：不合并，而是“并行解码”

工业场景里往往有多条序列：

- 长期搜索点击序列
- 搜索序列（Top-50）
- Feed 行为序列（Top-50）

MTGR/OneTrans 通常把它们拼成一个大序列再建模，但 HyFormer 实验表明这种“强行合并”会损失语义差异，AUC 下降 0.06%。

HyFormer 的策略是：

- 每条序列各自拥有独立 query tokens；
- Query Decoding 在序列内部完成；
- 跨序列交互在 Query Boosting 阶段通过 token mixing 实现。

这既保留了序列独特性，也避免了“对齐特征维度”的工程复杂度。

## 5. 训练与部署优化：能跑起来才是关键

### 5.1 GPU Pooling for Long Sequence

长序列 token 数极大，Host-to-Device 传输成本高。论文发现长序列里 **真正唯一的特征 ID 只占 25%**。于是他们做了两件事：

- **离线去重 + 压缩 embedding table**
- 在线时在 GPU 上重建完整序列

这样减少了传输和主存占用，同时保持训练效果。

### 5.2 异步 AllReduce

为了减少通信瓶颈，引入异步 AllReduce，把 step k 的梯度同步与 step k+1 的前后向计算重叠。

代价是 dense 参数会有一步“滞后”：

$$
W_{k}=W_{k-1}+g_{k-1}
$$

而 sparse 参数仍可即时更新：

$$
W_{k}=W_{k-1}+g_{k}
$$

论文实验显示这种“轻微不一致”不会影响收敛。

## 6. 实验设置

### 6.1 数据集

- **Douyin Search CTR 任务**
- 70 天日志
- 约 **3 billion samples**
- 特征包含用户、query、文档、交叉特征以及多条序列
- 序列定义：
  - 长期序列：最长 3000
  - 搜索序列：Top-50
  - Feed 序列：Top-50

### 6.2 评价指标

- **Query-level AUC**（按 query 分组平均）
- 参数量与 FLOPs（batch size 2048）

### 6.3 实现细节

- MLP-Mixer 的 token 数量固定为 16：13 个非序列 token + 3 个 global tokens（对应三条序列）
- 64 GPU 训练

## 7. 整体结果：HyFormer 在精度和效率上同时领先

### 7.1 Overall Performance（Table 1）

关键结果：

- **LONGER + RankMixer（基线）**：AUC 0.6478
- **Full Transformer + RankMixer**：AUC 0.6481（+0.05%），但 FLOPs 6.6
- **MTGR/OneTrans (Full Transformer)**：AUC 0.6483（+0.08%），FLOPs 高达 21.9
- **HyFormer**：AUC 0.6489（+0.17%），FLOPs 3.9

换句话说：**HyFormer 既比基线更准，又比“重型统一模型”更省算力。**

如果把参数量和 FLOPs 放在一起看，差距更直观：

- LONGER + RankMixer：Params 386M，FLOPs 3.5
- Full Transformer + RankMixer：Params 388M，FLOPs 6.6
- MTGR/OneTrans（Full Transformer）：Params 450M，FLOPs 21.9
- **HyFormer：Params 418M，FLOPs 3.9**

HyFormer 的参数规模并不最小，但 **AUC 却是最高，FLOPs 仍接近基线水平**。这说明它的优势来自“结构性信息流设计”，而不是单纯堆参数或算力。
论文还解释了 MTGR/OneTrans 的不足：

- 它们用自注意力做特征交互，成本大；
- Global Tokens 与 Seq Tokens 混合做 key/value 时，Global Tokens 容易“自己关注自己”；
- HyFormer 强制“先吸收序列，再做交互”的两步流程，避免信息流混乱；
- 混合式结构允许独立调节序列建模深度与特征交互深度，更利于后续 scaling。

### 7.2 Ablation（Table 2）

- **去掉序列 pooling tokens**：AUC -0.05%  
- **只保留 target tokens（去掉非序列 + 序列 summary）**：AUC -0.08%

说明：**Query 的语义来源越丰富，越能提升序列解码质量**。

Query Boosting 消融：

- HyFormer w/o Global Tokens：AUC -0.08%
- BaseArch w/ Global Tokens：AUC -0.14%

结论：**单加 Global Tokens 不够，必须配合 Query Boosting 才能释放价值**。

多序列消融：

- 合并序列（merge）导致 AUC -0.06%

说明 HyFormer 的“分序列解码 + query 交互”确实更合理。

### 7.3 Scaling with Sequence Sparse Dim（Table 3）

扩大序列侧信息（sparse dim）带来的增益：

- 序列长度 1k：
  - BaseArch +0.09%
  - HyFormer +0.12%（多出 +0.03%）

- 序列长度 3k：
  - BaseArch +0.06%
  - HyFormer +0.12%（多出 +0.06%）

说明：**序列越长，HyFormer 对“序列信息密度”的吸收能力越强。**

### 7.4 在线 A/B（Table 4）

在 Douyin 上线上指标提升：

- **Average Watch Time / user**：+0.293%
- **Video Finish Play Count / user**：+1.111%
- **Query Change Rate**：-0.236%

Query Change Rate 的定义是：

$$
\mathrm{query\ change\ rate}=\frac{N_{\mathrm{reform}}}{N_{\mathrm{total}}}
$$

下降意味着用户更少“反复修改搜索词”，体现更强的结果匹配性。

## 8. HyFormer 的创新点到底在哪里？

归纳下来是三条主线：

1. **Global Tokens 作为统一接口**  
   把序列信息与非序列语义“绑”在同一层的 query 上，而不是先压缩后融合。

2. **Query Decoding + Query Boosting 交替执行**  
   让序列信息与特征交互形成双向流，不再是单向“先序列后交互”。

3. **多序列建模的“并行解码”策略**  
   不强行拼接序列，保留语义独立性，再通过 query 层面交互融合。

更重要的是，这种设计不是“把 Transformer 堆大”，而是在**信息流路径上做结构性调整**。

## 9. 与其他方法的对比（更直观的理解）

### 9.1 vs LONGER + RankMixer

- LONGER 把序列压缩成 query token 后，交互阶段只看到压缩结果；
- HyFormer 每层都让 query 再次“读序列”，并与非序列特征混合；
- **本质区别：单向信息流 vs 交替优化。**

### 9.2 vs Full Transformer

- Full Transformer 在长序列上成本极高；
- HyFormer 在序列编码上允许 LONGER 或轻量解码，避免高 FLOPs；
- 结果是 **相近或更好 AUC，但计算成本低很多**。

### 9.3 vs MTGR / OneTrans

- MTGR/OneTrans 统一建模，但自注意力开销大；
- 容易“global tokens 自我关注”，序列信息吸收不足；
- HyFormer 强制“先读序列，再交互”，保证信息流合理。

这也是为什么 HyFormer 在 Table 1 里能同时压过基线和统一架构模型。

## 10. 可能的局限与思考

- **Global Tokens 数量如何设定？** 论文中固定为 3（对应三条序列），但在更复杂的场景下，如何动态分配仍是问题。
- **MeanPool(Seq) 的摘要是否足够？** 如果序列兴趣高度多样，单一 MeanPool 可能过粗。
- **异步 AllReduce 带来的“时序不一致”** 是否会在极端大模型下放大，还需要更多实证。

从工程落地角度，一个更实际的问题是“迁移成本”。HyFormer 不是简单替换某个模块，而是会牵动 query token 设计、序列编码策略、token mixing 模块三条链路。它适合“已经在 LONGER + RankMixer 路线上走到瓶颈”的团队，或者在统一架构上遇到效率墙的团队。对资源有限的小团队来说，可以先尝试：

1) 保留原有序列编码（比如 LONGER）
2) 引入 Query Decoding + Query Boosting 的交替框架
3) 在不改变 token 数的前提下逐层加深

这样能以较小代价验证“交替信息流”是否带来收益。

## 11. 总结

HyFormer 的价值不在于“更复杂的注意力”，而在于**重新定义序列建模与特征交互的协作方式**。

它用 Global Tokens 作为接口，通过 Query Decoding 把序列信息吸到 query 中，再用 Query Boosting 加强异构交互，形成一种交替优化的双向信息流。这种结构性调整让模型在精度、效率、扩展性三方面同时受益。

如果你正在搭建工业级 CTR 模型，HyFormer 的最大启发可能是：

- **序列建模与特征交互不是“先后关系”，而是“协同关系”**；
- **真正的增益来自信息流的设计，而不只是堆更大的模型**。

这也是这篇论文“重新理解两者角色”的真正价值。
