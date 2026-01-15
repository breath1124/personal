---
title: "LEMUR 论文解读：工业级端到端多模态搜索推荐怎么做（Memory Bank + SQDC）"
description: "解读字节跳动 LEMUR（arXiv:2511.10962v2）：把多模态 Transformer 与排序模型真正端到端联合训练；用 Session-masked Query-Doc Contrastive（SQDC）对齐表示；用 Memory Bank 让超长历史序列的多模态建模在工业成本下可训练、可部署，并给出显著离线/在线收益。"
pubDate: 2026-01-15
tags: ["论文解读", "推荐系统", "搜索排序", "多模态", "端到端", "对比学习", "Memory Bank", "RankMixer", "LONGER"]
category: "论文解读"
draft: false
---

> 论文：LEMUR: Large scale End-to-end MUltimodal Recommendation（arXiv:2511.10962v2）  
> 链接：<https://arxiv.org/abs/2511.10962>（HTML：<https://arxiv.org/html/2511.10962v2>）

很多工业推荐/搜索系统都知道“多模态很重要”：文本（title）、画面文字（OCR）、语音转写（ASR）、图片/视频视觉信号……这些内容特征能显著缓解冷启动、提升泛化。

但你一旦想把“内容编码器”（比如 Transformer/视觉语言模型）**端到端**拉进排序/CTR 训练里，马上会遇到一串现实问题：

1. **两阶段训练的错位（misalignment）**：上游多模态模型预训练学到的 embedding，未必对 CTR/排序目标最有用；下游排序模型又会把大部分信号塞进 ID embedding，让内容 embedding “边缘化/欠拟合”。
2. **更新频率不一致**：排序模型在线高频更新，而多模态编码器往往预训练后就冻结，越到后面越跟不上分布漂移。
3. **长序列的算力爆炸**：用户历史序列动辄上百上千条。如果每次训练都把“历史里的每个 doc”都过一遍 Transformer，成本会呈数量级上升。
4. **线上传输瓶颈**：如果把多模态 embedding 当成独立服务输出，再塞进超长序列，带宽和延迟也会成为硬瓶颈。

LEMUR 的价值在于：它不是“提出一个更强的编码器”，而是把“端到端多模态 + 超长序列 + 工业训练/部署”这三个互相打架的目标，用一套工程与建模组合拳捏到一起，并给出明确的线上收益。

下面按 **动机 → 公式/方法 → 工程关键 → 实验与消融** 把论文拆开讲清楚。

## 1. 任务与目标：CTR 预测（Douyin Search）

论文以抖音搜索为例，优化目标是 CTR（二分类）。预测函数：

$$
\hat{y}=f_{\theta}(x)
$$

训练使用二分类交叉熵（对数据集 $\mathcal{D}$ 求平均）：

$$
\ell_{\text{CTR}}=-\frac{1}{|\mathcal{D}|}\sum_{(x,y)\in\mathcal{D}}\Big[y\log\hat{y}+(1-y)\log(1-\hat{y})\Big]
$$

接下来所有“多模态/序列/对比学习”的设计，本质都是为了让 $f_{\theta}$ 在大规模工业数据上学到更强的表示，从而提升 CTR 预测与排序质量。

## 2. 总体思路：把多模态编码器拉进排序闭环

LEMUR 的主干可以概括成三件事：

1. **Raw 特征直接进 Transformer**：对 query、doc 的 raw 内容特征分别过双向 Transformer，得到向量表示。
2. **SQDC 对比学习对齐表示**：利用真实点击信号做 in-batch Query-Doc 对比学习，同时用 session/query 级 mask 避免“同一 query 的样本互相当负例”导致训练不稳定。
3. **Memory Bank + 轻量序列建模**：把 doc 的多模态表示缓存进 memory bank，历史序列直接按 doc id 查表；然后用一个计算友好的 Decoder + 相似度模块建模超长序列，最后与其他特征一起喂给 RankMixer 输出 logit。

这三点里，前两点更偏“建模”，第三点更偏“工程可落地”。但它们必须配合：没有 Memory Bank，端到端会算不动；没有端到端梯度回传，多模态表示会与排序目标错位；没有 SQDC，query-doc 内容表示难以可靠对齐。

## 3. Raw Features Modeling：query/doc 各自 Transformer 编码

对第 $i$ 个样本，分别对 query raw 特征 $q_{\text{raw}}^{i}$ 与 doc raw 特征 $d_{\text{raw}}^{i}$ 编码：

$$
q_i=\text{Transformer}(q_{\text{raw}}^{i}),\qquad d_i=\text{Transformer}(d_{\text{raw}}^{i})
$$

实现细节上，论文用了几个很“工业实用”的小设计：

- 在 query 与 doc 序列开头插入独立的 `cls` token，用其输出作为整体表示。
- doc 输入里，不同字段（例如 title/OCR/ASR/封面 OCR 等）之间插 `sep` token 作为边界。
- 除了 position embedding，还加 **type embedding** 区分“不同字段/特征类型”。

得到 $q_i,d_i$ 后，不是单独做一个“两塔召回”，而是 **把它们当作新特征**，与原有用户特征、ID 特征、交叉特征、序列特征等拼接，输入 RankMixer 得到最终 CTR logit。

这点很关键：LEMUR 并不试图“用内容特征替换 ID 特征”，而是要让内容表示在端到端训练中被排序目标驱动，真正成为 ranker 的一部分。

## 4. SQDC：Session-masked Query-Doc Contrastive（重点公式）

### 4.1 为什么需要对比学习？

在搜索推荐里，query 和 doc 的内容表示天然应该“对齐”：同一个 query 下被点击的 doc（正样本）应该更接近。

仅靠 CTR loss 也能学到这种关联，但它更像“把一切都丢进大模型去拟合”。对比学习提供了一个更直接的几何约束：**把正对拉近、把负对推远**，让表示空间更稳定、可泛化。

### 4.2 相似度：余弦相似

$$
\text{sim}(q,d)=\frac{q^{T}d}{\|q\|\|d\|}
$$

### 4.3 In-batch 对比学习（InfoNCE 风格）

设 batch size 为 $K$。对每个正样本 $i$（$\text{label}_i=1$ 表示真实点击），希望 $(q_i,d_i)$ 的相似度高于 $(q_i,d_j)$（$j\neq i$）。论文写法里，用一个缩放系数 $T$（命名为 temperature）：

$$
\ell_{\text{contrastive}}
=\sum_{i,\text{label}_{i}=1}-\log\frac{\exp\left(\text{sim}(q_{i},d_{i})*T\right)}{\sum_{j=0}^{K}{\exp\left(\text{sim}(q_{i},d_{j})*T\right)}}
$$

### 4.4 “同 query 样本”不是好负例：引入 session/query 级 mask

搜索日志里，batch 往往按 query/session 组织：同一个 query 下会出现多条样本（对应不同候选 doc，或不同曝光/点击）。

如果你把同一 query 的其他 doc 都当负例，会出现一个非常反直觉的效果：

- query = “坦克 300”
- doc A：介绍越野车坦克 300（点击）
- doc B：也是坦克 300 的试驾/测评（可能也点击，或者未点击但语义强相关）

把 B 当“强负例”去推远，会把“同语义簇”的 doc 表示撕裂，训练会很不稳定（论文也强调同 query 的样本高度相关）。

于是 LEMUR 定义 SQDC，把同 query 的样本从负例集合里 mask 掉。SQDC 损失：

$$
\ell_{\text{SQDC}}
=\sum_{i,\text{label}_{i}=1}-\log\frac{\exp\left(\text{sim}(q_{i},d_{i})*T\right)}{\sum_{j=0}^{K}{A_{ij}*\exp\left(\text{sim}(q_{i},d_{j})*T\right)}}
$$

其中 $A_{ij}$ 是 mask，$\text{QID}$ 是 query identifier：

$$
A_{ij}=
\begin{cases}
0 & \text{if } i\neq j \text{ and } \text{QID}_{i}=\text{QID}_{j}\\
1 & \text{else}
\end{cases}
$$

直觉总结：**“同 query 的别的样本”要么是潜在正例，要么是“难负例但语义相关”，直接当负例会污染对比学习信号；mask 掉它们能让 in-batch 对比更稳。**

## 5. Memory Bank：让超长多模态序列可训练、可部署

### 5.1 痛点：历史序列全过 Transformer 算不动

LEMUR 不仅要对“当前候选 doc”做多模态表示，还要对“用户历史 doc 序列”建模多模态表示。

但联合训练的算力成本极高。论文给了一个很直观的量级例子（batch=2048）：

- base ranking model：**1.6 TFLOPs**
- doc Transformer：**2.3 TFLOPs**
- 若历史序列长度到 1024：计算量会 **上千倍** 增长

这还没算上多机、多任务、在线训练等工程约束。

### 5.2 解法：把 doc 表示缓存为“按 doc id 可查”的向量

Memory Bank 的操作方式很简单但很关键：

1. 每个 batch 里，对“目标 doc”（当前候选）跑 Transformer，得到 $d_i$，**写入 memory bank**（key 是 doc id）。
2. 对用户历史序列里的 doc，不再跑 Transformer，而是 **按 doc id 从 memory bank 取出其多模态表示**，拼成序列表示 $(d_1,\dots,d_N)$。

由于只考虑“一月窗口”的用户历史，而训练数据覆盖两个月以上，memory bank 在训练中会逐步积累到足够的覆盖率（论文也在实验里分析了 coverage 与 staleness）。

这一步直接解决了两类工业问题：

- **训练成本**：历史序列的多模态表示从“算出来”变成“查出来”。
- **线上传输**：如果把 memory bank 放在训练/Serving 的基础设施里，就不需要把长序列的 embedding 当作外部服务结果在系统里到处传（论文在引言明确把它作为解决传输瓶颈的关键）。

### 5.3 代价与副作用：staleness 与 coverage

缓存会带来两个经典问题：

- **staleness（陈旧）**：模型参数在更新，但 memory bank 里的旧表示没有实时刷新。
- **coverage（覆盖率）**：历史序列里的 doc 有没有被写入过 memory bank？没写入就取不到。

论文给出的结论是：训练过程中“当前模型输出 vs 缓存输出”的相似度逐步升到约 0.95 并稳定，表明 staleness 可控；长短序列的覆盖率最终都能超过 90%，当前 doc 的覆盖率长期保持 98%+，说明 serving 侧可用。

## 6. 多模态序列建模：改版 LONGER + Decoder + Similarity Module（含公式）

Memory bank 解决了“序列里每个 doc 的表示怎么来”。接下来还需要一个序列模型把这些表示用起来。

LEMUR 的序列建模由三部分组成：

1. 一个改版 LONGER（长序列友好）
2. 一个 Decoder（跨注意力 + FFN，刻意做轻量）
3. 一个 similarity module（目标 doc 与历史 doc 的相似度特征）

### 6.1 Decoder：用 query token 跨注意力读历史

Decoder 第 1 层先把“其他特征”变成一个 query token $Q_1$，对历史表示 $d_1,\dots,d_N$ 做 cross-attention，然后仅用 FFN 生成下一层 token：

$$
Q_{i+1}=\text{FFN}\left(\text{CrossAttention}(Q_{i},d_{1},d_{2},\dots,d_{N})\right)
$$

cross-attention 的输出写成加权和：

$$
\text{CrossAttention}(Q_{i},d_{1},d_{2},\dots,d_{N})=\sum_{j=1}^{N}a(Q_{i},d_{j})d_{j}
$$

注意力权重是 softmax（为避免符号冲突，我把分母求和下标写成 $k$）：

$$
a(Q_{i},d_{j})=\frac{\exp(Q_{i}^{T}d_{j})}{\sum_{k=1}^{N}\exp(Q_{i}^{T}d_{k})}
$$

论文强调 Decoder 在 cross-attn 后只用 FFN（不再堆更多 attention），以显著降低计算量；并且 baseline（RankMixer+LONGER）里也用同样的 Decoder 结构作为序列建模组件。

### 6.2 Similarity module：目标 doc 与历史 doc 的“内容相似”特征

在推荐里，“用户看过的内容”和“当前候选内容”之间的相似度通常很重要，尤其在内容理解更强的多模态场景。

LEMUR 额外计算目标 doc 与历史 doc 的余弦相似度向量，并维护其 **ranked version**（把相似度排序后的版本）来稳定训练。最终：

- Decoder 输出
- 相似度特征（含排序版本）
- 其他稠密/离散特征

一起输入 RankMixer 输出最终 logit。论文里模型包含多条序列，最长一条能到 **1000 items**。

## 7. 训练效率：flash-attn、混精、采样、去重

即使用了 memory bank，Transformer 依然可能比 ranker 更贵，所以论文还做了四类优化：

- flash attention + 混合精度训练
- **样本采样**：只对一部分样本跑 Transformer 的 forward/backward，其余直接复用 memory bank 的 embedding
- **跨 worker 的 doc 去重**：同一个 batch 里可能有大量重复 doc；先去重只保留一份再过 Transformer，可显著省算力（细节在附录）
- 推理时直接查 memory bank，不激活 Transformer

采样策略论文用 $p,q$ 表示 forward/backward 的采样比例，并在实验里给了 FLOPs 与效果的权衡（见后文 Table 4）。

## 8. 实验设置：数据、指标与线上口径

### 8.1 数据与多模态输入

离线数据来自抖音搜索推荐日志：70 天、约 30 亿样本；包含用户特征、query/doc 特征、交叉特征、用户搜索历史特征；涉及 **数十亿用户 ID** 与 **数亿 doc ID**。

多模态“raw 文本”来自多源内容（论文 Table 1）：

| source | feature | max length | avg length |
|---|---|---:|---:|
| query | query | 10 | 4 |
| doc | title | 31 | 22 |
| doc | OCR | 124 | 81 |
| doc | ASR | 12 | 2 |
| doc | cover image OCR | 12 | 6 |

### 8.2 Baselines 与指标

论文对比了生产环境的两个 SOTA 组件：

- **RankMixer**：高 MFU 的推荐模型（token mixing + per-token FFN）。
- **LONGER**：面向长序列、GPU 友好的 Transformer 结构。

离线指标：AUC、QAUC（按 query 分组算 AUC 再平均）。

在线指标：query change rate（用户把查询改得更具体通常代表体验较差），定义为：

$$
\text{query change rate}=
\frac{\text{number of distinct UID-query pairs with query reformulation}}{\text{total number of distinct UID-query pairs}}
$$

## 9. 实验结果：离线提升 + 线上收益 + 优于两阶段

### 9.1 离线：相对 RankMixer+LONGER，LEMUR 还能再挖出 0.81% QAUC

论文 Table 2（数值为相对提升，baseline 为 DLRM-MLP，同时给出对线上 baseline 的额外增益）：

| 模型 | AUC | QAUC |
|---|---:|---:|
| DLRM-MLP | 0.74071 | 0.63760 |
| +RankMixer（相对 DLRM） | +0.49% | +0.59% |
| +LONGER（累计相对 DLRM） | +0.89% | +1.04% |
| +LEMUR-SQDC（累计相对 DLRM） | +1.22% | +1.51% |
| +LEMUR-SQDC-MB（累计相对 DLRM） | +1.44% | +1.85% |
| 相对 RankMixer+LONGER 的增益 | +0.55% | +0.81% |

把它翻译成一句话：**在已经是线上生产 SOTA（RankMixer+LONGER）的情况下，端到端多模态 + memory bank 仍能带来 +0.81% 的 QAUC，相当可观。**

### 9.2 线上：部署 1 个月，QAUC +0.81%，query change rate 下降

论文在摘要中给出线上结论：部署在 Douyin Search 1 个月后，

- **QAUC +0.81%**
- **query change rate decay -0.843%**（更少的 query 改写，代表更好的搜索体验）

同时在 Douyin Advertisement 的多个离线关键指标上也有显著收益（论文未在正文给出具体数值）。

### 9.3 对比两阶段：端到端明显更强

论文还做了一个很关键的对照：用同样的 SQDC loss 先预训练 Transformer，再冻结表示做下游推荐（典型两阶段）。Table 3：

| 方法 | QAUC | ΔQAUC |
|---|---:|---:|
| baseline | 0.64393 | – |
| Two-Stage | 0.64470 | +0.12% |
| LEMUR | 0.64914 | +0.81% |

因此 LEMUR 相对两阶段的额外提升约 **0.69%** QAUC。这个结果很“工业直觉”：如果你把表示学习和排序目标拆开，多模态表示就很难被真实的点击/排序信号“拧到位”。

## 10. 消融：哪些组件真的在贡献增益？

### 10.1 联合训练的梯度很重要

Table 5 显示：如果在 RankMixer 里对 Transformer 表示 **stop gradient**，最终 QAUC 下降 **0.31%**。这直接说明：

> 排序模型中的特征交互与 CTR 信号，能反过来“教会”多模态编码器学到更有用的表示；把梯度掐断，端到端就失去了关键优势。

### 10.2 长短序列、相似度特征都有效

同样来自 Table 5：

- 去掉 short multimodal sequence：-0.09%
- 去掉 long multimodal sequence：-0.25%
- 去掉 cosine similarity：-0.12%

也就是说，**把多模态表示真正放进超长历史序列去用**，是 LEMUR 重要的增益来源之一。

### 10.3 SQDC 与 session mask 的贡献是“可量化”的

还是 Table 5：

- 去掉 SQDC loss：-0.17%
- 去掉 session-level mask：-0.10%

换句话说：SQDC 总贡献约 +0.17%，其中“同 query mask”就占了 +0.10%。这基本符合我们在第 4 节的直觉分析：batch 里同 query 样本相关性太强，不处理就会污染对比学习。

### 10.4 温度与负采样：50 最好，负采样差别不大

Table 6（仅列关键结论）：

- temperature=50 最优（QAUC 0.64914，对应 +0.81%）
- 用 “in-batch positive samples” 或 “in-batch samples” 做负例，差别极小（+0.81% vs +0.80%）

### 10.5 采样策略：把 FLOPs 从 4303G 拉回到 2439G，收益只掉 0.05%

Table 4 给出不同训练采样下的成本与效果（QAUC 为相对 baseline 的提升）：

| 方法 | params | FLOPs | QAUC |
|---|---:|---:|---:|
| baseline | 92M | 1673G | – |
| LEMUR, no-sampling | 139M | 4303G | +0.81% |
| LEMUR, p=100,q=20 | 139M | 3232G | +0.78% |
| LEMUR, p=20,q=20 | 139M | 2439G | +0.76% |
| LEMUR, p=10,q=10 | 139M | 2262G | +0.75% |

论文最终采用 $p=20,q=20$：在 ROI 可接受的前提下，还能更好覆盖长序列场景。

## 11. 复用 LEMUR 的三条工程启示

1. **端到端不是口号，是梯度通路**：内容编码器必须吃到排序任务的梯度（Table 5 的 -0.31% 非常说明问题）。
2. **超长历史序列一定要“查表化”**：memory bank 把“历史多模态表示”从算力瓶颈变成存储/一致性问题，才有机会工业落地。
3. **对比学习别忽略日志结构**：搜索/推荐日志天然带 session/query 的相关性；对比学习若不做 mask，很容易把“同语义簇”当负例推开，得不偿失。

## 12. 小结

LEMUR 这篇论文最值得带走的不是某个单点技巧，而是一个工业化模板：

- 用 raw 多源内容信号做多模态表示（而不是把内容 embedding 当外部服务）
- 用对比学习 + session mask 把 query/doc 表示对齐
- 用 memory bank + 轻量序列模型把超长历史吃进去
- 再用一套采样/去重/混精把成本压到可上线

如果你正在做“端到端多模态 + 长序列”的搜索/推荐系统，LEMUR 基本就是一份很扎实的工程路线图。

