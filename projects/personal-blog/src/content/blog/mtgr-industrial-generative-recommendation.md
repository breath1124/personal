---
title: "MTGR 论文解读：美团工业级“生成式”推荐框架如何做到可扩展又不丢特征？"
description: "解读 MTGR（arXiv:2505.18654v4）：用“按用户聚合候选 + Transformer(HSTU) 编码 + 动态 Mask + GLN”把 DLRM 的 cross feature 和 GRM 的可扩展性揉到一起，并给出离线/线上实验与训练系统工程细节。"
pubDate: 2026-01-12
tags: ["论文解读", "推荐系统", "生成式推荐", "Transformer", "工业实践"]
category: "论文解读"
draft: false
---

> 论文：MTGR: Industrial-Scale Generative Recommendation Framework in Meituan（arXiv:2505.18654v4）
> 链接：<https://arxiv.org/abs/2505.18654>

这篇文章想解决一个非常“工业界”的矛盾：

- **DLRM（Deep Learning Recommendation Model）**在真实业务里效果强，原因之一是它能吃下大量精心设计的特征，尤其是 **cross feature（交叉特征）**；但它的推理/训练计算常常和候选数近似线性相关，规模化很难。
- **GRM（Generative Recommendation Model）**把推荐问题组织成 token 序列，用 Transformer 做“下一 token”预测，天然更像“可扩展的序列建模”；但很多 GRM 的落地形式会**弱化或移除 cross feature**，在真实场景里性能会明显退化。

**MTGR**的核心主张是：在不放弃 DLRM 的特征体系（尤其 cross feature）的前提下，用更像 GRM 的“token 化 + Transformer”组织方式，把“候选数线性成本”的问题打掉，从而让推荐排序模型也能吃到类似 scaling law 的红利。

下面我按“背景 → 创新点 → 数学与方法 → 工程系统 → 实验效果”把它讲清楚，尽量通俗。

## 1. 背景：工业级排序模型为什么难 scale？

一个典型的工业排序（ranking）阶段，每次请求会拿到一个用户以及一批候选商品（几十到上千不等），目标是为每个候选打分。

DLRM 范式通常把**每个 (user, item) 对**视为一个独立样本来做二分类/多任务预测（例如 CTR、CTCVR），包含：

- 用户画像特征（年龄、性别、城市……）
- 用户历史行为序列（长期序列）
- 用户近实时行为序列（短期/实时序列）
- 候选商品特征（item id、品牌、类目……）
- **cross feature**（用户与候选的交互统计、上下文交叉、时空交叉等）

问题在于：当你想把模型做大（更深的 MLP、更复杂的交互、更强的序列建模），**很多计算发生在“用户×候选”的交互上**，这部分往往需要对每个候选都跑一次，于是：

- 候选越多，推理成本越高、延迟越难控；
- 训练样本也会被“同一用户的多候选曝光”放大，计算冗余大；
- 现实里只好在序列长度、交互模块复杂度上做妥协，导致“想 scale 但 scale 不动”。

论文把这个矛盾概括为：**user module**（只跟用户相关）扩展相对划算，而 **cross module**（需要对每个候选算交互）扩展会带来近线性成本。

## 2. 预备知识：论文如何描述“传统数据与模型”？

### 2.1 传统样本表示

对一个用户，以及对应的 $K$ 个候选，传统 DLRM 会把第 $i$ 个候选的样本写成：

$$
\mathbb{D}_{i}=[\mathbf{U},\overrightarrow{\mathbf{S}},\overrightarrow{\mathbf{R}},\mathbf{C}_{i},\mathbf{I}_{i}]
$$

其中（论文 Sec. 3.1）：

- $\mathbf{U}=[\mathbf{U}^1,\dots,\mathbf{U}^{N_\mathbf{U}}]$：用户画像标量特征（年龄/性别等）
- $\overrightarrow{\mathbf{S}}=[\mathbf{S}^1,\dots,\mathbf{S}^{N_\mathbf{S}}]$：历史交互 item 序列（每个 $\mathbf{S}^t$ 本身是由若干 item 属性组成的向量）
- $\overrightarrow{\mathbf{R}}$：近实时交互序列（与 $\overrightarrow{\mathbf{S}}$ 类似，但时间窗口更近）
- $\mathbf{C}_{i}$：用户与候选 $i$ 的 cross feature（例如该用户对该候选的历史 CTR、曝光次数等）
- $\mathbf{I}_{i}$：候选 $i$ 的 item 特征（ID、brand、tag……）

### 2.2 典型的序列处理：Target Attention

推荐里常见的一种做法是对行为序列做 target attention：用候选 item 表示做 query，去“检索”用户序列中和该候选更相关的兴趣（论文 Sec. 3.2）：

$$
\mathbf{Feat}_{\overrightarrow{\mathbf{S}}}
=\text{Attention}(\mathbf{Emb}_{\mathbf{I}},\mathbf{Emb}_{\overrightarrow{\mathbf{S}}},\mathbf{Emb}_{\overrightarrow{\mathbf{S}}})
\in\mathbb{R}^{K\times d_\mathbf{S}}
$$

直觉上：每个候选都要在用户历史里“找一遍相似兴趣”，因此 $\mathbf{Feat}_{\overrightarrow{\mathbf{S}}}$ 的 shape 会带上 $K$，这也暗示了**候选数对计算的影响**。

最后把各类特征拼起来得到 dense 表示（论文式子(2)）：

$$
\mathbf{Feat}_{\mathbb{D}}=
[\mathbf{Emb}_{\mathbf{U}},
\mathbf{Feat}_{\overrightarrow{\mathbf{S}}},
\mathbf{Feat}_{\overrightarrow{\mathbf{R}}},
\mathbf{Emb}_{\mathbf{C}},
\mathbf{Emb}_{\mathbf{I}}]
$$

再过 MLP / cross module 输出 CTR、CTCVR 等任务的 logits。

### 2.3 Scaling 困境

如果你把交互模块（cross module）做得更强，往往需要对每个候选都跑一遍，计算会随候选数近似线性增长，导致线上延迟无法接受（论文 Sec. 3.3）。

所以 MTGR 的目标很明确：**要一个“候选数更不敏感”的结构**，同时又要把 cross feature 这类强信号保留下来。

## 3. MTGR 的关键创新点（我认为最重要的 4 件事）

### 3.1 按用户聚合候选：把“多次前向”变成“一次前向”

MTGR 的第一刀，是把原来“一个候选一个样本”的组织方式，改成“一个用户聚合一批候选”的组织方式（论文 Sec. 4.1）。

把同一用户在一个窗口内的 $K$ 个候选合并成一个样本：

$$
\mathbb{D}=[\mathbf{U},\overrightarrow{\mathbf{S}},\overrightarrow{\mathbf{R}},[\mathbf{C},\mathbf{I}]_{1},\dots,[\mathbf{C},\mathbf{I}]_{K}]
$$

直觉：

- 原本要对 $K$ 个候选分别算一遍“用户表示 + 交互表示”，现在变成**一次编码**输出 $K$ 个候选的表示/分数；
- 推理时对一个请求里的所有候选做一次前向，成本不再随 $K$ 线性飙升（至少大幅缓解），这给模型变大留出了空间。

> 论文强调：聚合让训练样本量从“所有候选”更接近“所有用户请求”，减少冗余；推理也从“候选次”变成“请求次”。

### 3.2 Token 化：把异构特征变成可做 self-attention 的序列

聚合后，$\mathbb{D}$ 同时包含标量特征、序列特征、候选 item（每个候选又有 cross feature + item feature）。为了用 Transformer 统一处理，MTGR 把它们都转成 token：

- 用户画像：每个标量特征一个 token（维度统一到 $d_{\text{model}}$）
- 历史/实时序列：每个交互 item 一个 token（item 内部多特征先 embedding+concat，再 MLP 投到 $d_{\text{model}}$）
- 候选：每个候选一个或多个 token（论文实现里把 cross feature 作为候选 item token 的一部分）

然后把所有 token 拼成一个长序列送入 self-attention 编码器。

这一点非常“GRM”：**把推荐问题转成 token 序列建模**，但 MTGR 不做“下一 token 生成”，而是保留“打分/判别”的目标（discriminative loss）。

### 3.3 HSTU 风格的 Encoder：GLN + Q/K/V/U 四投影 + gating

论文在编码器里借鉴了 HSTU（Zhai et al., 2024），并做了面向推荐特征的改造（论文 Sec. 4.2）。

关键点一：**Group Layer Norm（GLN）**  
推荐特征是异构的：用户画像 token、序列 token、候选 token 语义空间不同。MTGR 把“同域 token”分组后做归一化，让不同域的 token 在进入 attention 前分布更可对齐：

$$
\tilde{\mathbf{X}}=\text{GroupLN}(\mathbf{X})
$$

关键点二：self-attention 的计算与更新（论文式子(5)(6)）  
对归一化后的 $\tilde{\mathbf{X}}$，分别用 MLP 投影到四组表示：

$$
\mathbf{K},\mathbf{Q},\mathbf{V},\mathbf{U}
=\text{MLP}_{\mathbf{K}/\mathbf{Q}/\mathbf{V}/\mathbf{U}}(\tilde{\mathbf{X}})
$$

然后用带 mask 的 attention 更新 value（论文(5)）：

$$
\tilde{\mathbf{V}}
=
\frac{\text{silu}(\mathbf{K}^{T}\mathbf{Q})}
{(N_\mathbf{U}+N_{\overrightarrow{\mathbf{S}}}+N_{\overrightarrow{\mathbf{R}}}+N_\mathbf{I})}
\mathbf{M}\mathbf{V}
$$

最后用 $\mathbf{U}$ 做 gating，再过一次 GLN+MLP，并残差回加（论文(6)）：

$$
\mathbf{X}=\text{MLP}(\text{GroupLN}(\tilde{\mathbf{V}}\odot\mathbf{U}))+\mathbf{X}
$$

直觉上可以把它理解为：**attention 负责信息路由，$\mathbf{U}$ 类似门控/调制项**，再配合分组归一化，让不同域信息更稳定地融合。

### 3.4 Dynamic Mask：防止“信息泄漏”，同时保留必要的可见性

把“候选 token”和“实时交互 token”放在同一个序列里，最怕的问题是 **causality 错误 / 信息泄漏**：比如晚上的实时交互不应该被下午的候选看到，但简单的 causal mask 可能会漏。

MTGR 的做法是按 token 类型制定可见性规则（论文 Sec. 4.2）：

1. **静态序列**（用户画像 $\mathbf{U}$、历史序列 $\overrightarrow{\mathbf{S}}$）对所有 token 可见（full attention）
2. **动态序列**（实时序列 $\overrightarrow{\mathbf{R}}$）遵守因果：一个 token 只能看“发生在它之后”的 token（并可见候选 token，取决于时间先后）
3. **候选 token**（包含 cross+item）只对自身可见（diagonal mask），避免候选之间互相“抄答案”

这套规则既保证模型能利用历史信息，又避免把未来信息泄露给过去的候选，从而更适合线上时序语义。

## 4. 训练系统：没有工程体系，就没有“工业级”

论文有一整节（Sec. 5）专门讲训练框架，这也是工业论文很有价值的部分。

他们没有沿用老的 TensorFlow 工业框架，而是基于 PyTorch 生态重构，并对 TorchRec 做了扩展优化。论文给出的几个关键点：

- **Dynamic Hash Table**：用可动态扩容/淘汰的 hash embedding table 替代固定大小的静态表，适配流式训练里不断出现的新 user/item id
- **Embedding Lookup 通信优化**：All-to-all 前后做 ID 去重，减少跨卡重复传输
- **Load Balance（动态 batch size）**：序列长度长尾会导致不同 GPU 负载不均；他们用“动态 BS + 梯度按 BS 加权聚合”来平衡
- **Pipeline 三流并行**：copy/dispatch/compute 三个 stream 重叠，隐藏 I/O 与查表开销
- **bf16 混合精度 + 自研 attention kernel（cutlass）**：进一步提升吞吐

论文声称：相比原 TorchRec，吞吐提升 **1.6×–2.4×**，并且在 **100+ GPU** 上可扩展；相比 DLRM baseline，前向 FLOPs/样本可达 **65×**，但训练成本“几乎不变”，推理成本还可下降（文中提到 -12%）。

## 5. 实验效果：离线、消融、可扩展性、线上 A/B

### 5.1 数据规模（论文 Table 1）

论文构造了来自美团真实工业系统日志的 10 天数据（强调 cross feature 的重要性）。数据量级如下：

| Split | #Users | #Items | #Exposure | #Click | #Purchases |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 0.21B | 4,302,391 | 23.74B | 1.08B | 0.18B |
| Test | 3,021,198 | 3,141,997 | 76,855,608 | 4,545,386 | 769,534 |

这张表本身就说明了：这是“工业规模”的排序问题，不是一个小而美的学术数据集。

### 5.2 模型大小与计算量（论文 Table 2）

论文给了 3 个 MTGR 尺寸（只列关键信息）：

| Model | Encoder | $d_{\text{model}}$ | heads | GFLOPs / example |
| --- | --- | ---: | ---: | ---: |
| UserTower-SIM | - | - | - | 0.86 |
| MTGR-small | 3 layers | 512 | 2 | 5.47 |
| MTGR-medium | 5 layers | 768 | 3 | 18.59 |
| MTGR-large | 15 layers | 768 | 3 | 55.76 |

可以看到 large 的单样本前向计算约是 baseline 的 $\sim 65\times$，但论文强调：由于“按用户聚合候选”，系统总成本依然可控，从而让这种规模成为可能。

### 5.3 总体效果（论文 Table 3）

论文用 CTR 与 CTCVR 两个任务，指标是 AUC/GAUC。这里摘取关键结论（数值来自 Table 3）：

- 最强 DLRM baseline 是 **UserTower-SIM**
- **MTGR-small** 已超过最强 DLRM baseline
- 随模型变大（small→medium→large）离线指标持续提升，呈现可扩展性

以 GAUC 为例（与 UserTower-SIM 对比）：

- CTR GAUC：0.6792 → 0.6865（+0.0073，约 +1.07%）
- CTCVR GAUC：0.6550 → 0.6646（+0.0096，约 +1.47%）

论文还提到他们的经验：离线指标提升 0.001 在该场景已属显著，因此这属于很可观的增益。

### 5.4 消融实验（论文 Table 4）

在 MTGR-small 上做消融（只列 GAUC）：

| Variant | CTR GAUC | CTCVR GAUC |
| --- | ---: | ---: |
| MTGR-small | 0.6826 | 0.6603 |
| w/o cross features | 0.6689 | 0.6514 |
| w/o GLN | 0.6809 | 0.6585 |
| w/o dynamic mask | 0.6810 | 0.6587 |

解读：

- **cross feature 非常关键**：去掉后明显掉点，甚至可能“抹掉 scale 带来的收益”
- GLN 与 dynamic mask 都是有效组件：去掉会有稳定退化

### 5.5 线上 A/B（论文 Table 5）

论文在美团外卖场景做了 **2% 流量**的 A/B，对比线上最强 DLRM（UserTower-SIM）。表 5 给了离线 GAUC 提升与线上业务指标提升（PV_CTR、UV_CTCVR）：

| Model | Offline: CTR GAUC diff | Offline: CTCVR GAUC diff | Online: PV_CTR | Online: UV_CTCVR |
| --- | ---: | ---: | ---: | ---: |
| MTGR-small | +0.0036 | +0.0154 | +1.04% | +0.04% |
| MTGR-medium | +0.0071 | +0.0182 | +2.29% | +0.62% |
| MTGR-large | +0.0153 | +0.0288 | +1.90% | +1.02% |

此外论文在引言里还提到，MTGR-large 在他们的业务里带来转化量、CTR 的提升，并已在真实系统中服务海量用户。

### 5.6 可扩展性（论文 Fig. 3）

论文展示了随着：

- HSTU block 数量增加
- $d_{\text{model}}$ 增大
- 输入序列长度增加

离线指标平滑提升，并呈现某种“性能-计算量”幂律关系（power law）。这与他们“把推荐也做成可 scale 的序列模型”的目标一致。

## 6. 我对 MTGR 的理解与启示

1. **“数据重排”可能比“换个网络结构”更关键**  
   按用户聚合候选，直接砍掉了最核心的成本项（候选数线性）。没有这一步，后面的 Transformer 再强也很难上线。

2. **工业推荐的“强信号”往往来自 cross feature**  
   学术数据集里 cross feature 稀缺，但线上系统里它们非常重要；MTGR 的一个核心贡献就是把 cross feature 重新塞回“token 序列”范式里。

3. **把异构特征做成 token 后，稳定训练变成第一要务**  
   GLN、dynamic mask、以及训练系统的各种优化，本质上都在解决“工业规模 + 异构特征 + 长序列”带来的不稳定和高成本。

4. **MTGR 更像“用 Transformer 做可扩展的判别式排序”**  
   虽然论文叫 generative recommendation，但它并不是让模型生成 item ID（那会遇到百万级词表 softmax、以及 cross feature 缺失等问题），而是借鉴 GRM 的组织方式来服务 ranking。

## 参考

- MTGR 论文（arXiv:2505.18654v4）：<https://arxiv.org/abs/2505.18654>
