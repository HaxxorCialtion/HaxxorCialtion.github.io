---
title: K-ON: Stacking Knowledge On the Head Layer of Large Language Model
date: "2025-10-03"
tags: ["KG Completion", "Multi-token Prediction", "LLM", "Contrastive Learning"]
description: "提出K-ON方法，通过多头层预测和实体级对比学习解决LLM在知识图谱补全任务中的粒度不匹配问题"
---

## 概括

**核心矛盾与时效性警示**：本文提出的K-ON方法基于multi-token prediction机制来解决知识图谱（KG）实体预测问题。然而，这一核心创新可能已与当前主流LLM架构的演进方向产生冲突——目前业界正在从传统的next-token prediction向multi-token prediction范式转变（如Meta的multi-token prediction模型）。这意味着本文的技术方案可能面临与原生多token预测能力的竞争，其独特性和必要性需要重新评估。

**问题定义**：LLM采用token级别的next-token prediction，而KG中实体是基本单位，需要多个token才能表示一个实体，存在**粒度不匹配**（granularity mismatch）问题。传统方法要么简化任务（如候选集验证），要么效率低下（逐token生成无法并行化）。

**核心创新**：K-ON在LLM的head layer集成KG知识，使用K个head层进行next k-step prediction，实现一步生成实体级结果，并支持实体级对比学习。

## 1. 要解决的问题

### 1.1 核心挑战
- **Token-Entity粒度不匹配**：实体（如"Matt Damon"）需要多个token表示，但LLM按token预测
- **无法并行化**：传统方法逐token生成实体，无法跨实体并行计算
- **Out-of-KG问题**：LLM缺乏对KG实体集合的感知，直接优化序列预测会生成不存在的实体
- **现有LLM方法的局限**：
  - 仅用于简化任务（如triplet验证、候选集排序）
  - token级优化而非entity级优化
  - 无法利用KG表示学习中最强大的对比学习

### 1.2 具体场景示例
给定不完整三元组：`(The Bourne Identity (2002 film), starring, ?)`
- **目标**：预测实体"Matt Damon"
- **传统问题**：需要多步生成"Matt" → "Damon"，无法并行处理其他候选实体

## 2. 核心创新点

### 2.1 K-ON架构
**多头并行预测**：
- 使用K个head layer，第k个head预测所有实体的第k个token
- 实体"Matt Damon"被tokenize为t₀（"Matt"）到t_{K-1}（"Damon"或padding）
- 从第1个head提取第1个token概率，从第K个head提取第K个token概率

**关键优势**：
- 一步生成完整实体
- 跨实体并行化计算
- 支持实体级对比学习

### 2.2 实体级对比损失（Entity-level Contrastive Loss）
```
p_e = Σ α_k · p_k  （加权聚合K步概率）
L_NCE = -log(p_e) + (1/N)Σ log(p_{e_j})  （对比损失）
```
- 将K步预测视为整体，计算联合概率
- 直接在实体空间优化，而非token空间
- 负样本从实体集E中随机采样

### 2.3 Head Trajectory Tuning (HTT)
解决两大风险：

**风险1：过度优化（Over-optimization）**
- 问题：优化第1个token "Matt"时，负样本包含大量非实体成分的token
- 解决：实体级对比损失强制K步预测作为整体优化

**风险2：分布损坏（Distribution Corruption）**
- 问题：原始LLM中第2个token "Damon"条件依赖第1个token "Matt"，但K步预测缺失这种依赖
- 解决：HTT对齐分布轨迹
  - **Supervised Fine-Tuning (SFT)**：用LoRA微调LLM，优化单步预测
  - **Token Distribution Tuning (TDT)**：最小化K-ON与原始LLM的KL散度

## 3. 核心技术栈

### 3.1 模型组件
1. **Head MLPs**：K个独立的MLP层处理LLM输出
   ```
   h^h_{0:K-1} = {L^h_k(σ(W^h_k · h^m_0))}^{K-1}_{k=0}
   ```
   - 使用SiLU激活函数和LlamaRMSNorm

2. **Conditional Attention**：小型Transformer捕获序列依赖
   ```
   h^a_k = M_s(h^h_{0:k}, M) + h^m_0
   ```
   - 因果mask确保第k步只能看到前k-1步
   - 残差连接允许从零初始化逐步学习

3. **LoRA Score Layer**：低秩适配预测概率
   ```
   W^S_k = W^S + A_k·B_k
   p_k = W^S_k · h^a_k
   ```

4. **K-step Gathering**：高效提取实体相关概率
   - 实体tokenize + padding/truncation至长度K
   - 从概率矩阵P中按索引提取

### 3.2 损失函数
```
L_total = L_NCE + L_sft + L_tdt
```
- **L_NCE**：实体级对比损失（核心）
- **L_sft**：token级监督微调（保持原始能力）
- **L_tdt**：分布对齐（KL散度）

### 3.3 实现细节
- 基座模型：Llama-2-chat-7B
- 硬件：8×A100 GPU
- 超参数：lr=1e-4, batch_size=12×8(梯度累积), K=8, N=128

## 4. Baseline & Benchmark

### 4.1 数据集
| Dataset | #Entity | #Relation | #Train | #Valid | #Test | #Text | #Image |
|---------|---------|-----------|--------|--------|-------|-------|--------|
| **DB15K** | 12,842 | 279 | 79,222 | 9,902 | 9,904 | 12,842 | 12,818 |
| **MKGW** | 15,000 | 169 | 34,196 | 4,276 | 4,274 | 14,123 | 14,463 |

**特点**：包含结构、文本、图像多模态信息（比纯结构数据集更全面）

### 4.2 Baseline方法分类

**1. 结构方法（Structure-only）**
- **TransE**：平移模型，h+r≈t
- **RotatE**：复数空间旋转模型
- 仅使用三元组结构，忽略实体自身特征

**2. 文本方法（Structure+Text）**
- **KG-BERT**：用BERT编码文本描述
- **FLT-LM**：微调语言模型以更好编码文本
- **KGLM**：集成语言模型进行链接预测

**3. 多模态方法（Structure+Text+Image）**
- **MMKRL**：多模态知识表示学习
- **MANS**：模态感知负采样
- **AdaMF**：自适应模态融合
- 利用实体的文本描述和图像信息

**4. LLM方法**
- **KG-Llama-7b**：在triplet验证任务上微调
- **GPT-3.5**：用于KG补全（效果不佳）
- 局限：token级优化，任务简化

### 4.3 评估指标
- **MRR** (Mean Reciprocal Rank)：平均倒数排名
- **Hits@K**：前K个预测中包含正确答案的比例
- 使用filtered ranks（过滤已知正确三元组）

## 5. 实验结果

### 5.1 主要结果（DB15K）
| 方法 | MRR | Hits@1 | Hits@3 | Hits@10 |
|------|-----|--------|--------|---------|
| RotatE (S) | 29.28 | 17.87 | 36.12 | 49.66 |
| FLT-LM (S+T) | 33.45 | 24.56 | 37.67 | 50.12 |
| AdaMF (S+T+I) | 32.51 | 21.31 | 39.67 | 51.68 |
| **K-ON (S+T)** | **38.10** | **30.13** | **42.77** | **53.59** |

**关键发现**：
- 仅用文本信息超越所有多模态方法
- 相比最佳baseline提升13.9% MRR

### 5.2 消融实验
| 变体 | MRR | Hits@1 |
|------|-----|--------|
| K-ON | 38.10 | 30.13 |
| w/o L_nce | 14.09 | 10.40 | ⬇️ **性能暴跌** |
| w/o L_tdt | 37.48 | 28.43 |
| w/o Conditional Attention | 37.20 | 27.69 |
| Shared Score Layer | 37.54 | 28.64 | ⬇️ 轻微下降 |
| Shared Head MLP | 28.01 | 19.57 | ⬇️ **严重下降** |

**结论**：
- **L_nce最关键**：移除后性能崩溃（验证实体级对比学习的核心作用）
- Conditional Attention对Hits@1影响显著（精确识别目标实体）
- 每步需要独立的Head MLP，但可共享Score Layer

### 5.3 超参数分析

**K值（head数量）影响**：
- K≥8时性能饱和（大多数实体<8 tokens）
- 计算成本线性增长
- 选择K=8平衡性能与效率

**负样本数量N影响**：
- N=128效果最佳
- 过多负样本反而降低性能
- 计算成本相对稳定（复用概率分布）

**联合概率函数**：
- 加权求和（learnable +）最优：MRR=38.10
- 乘法（*）效果差：MRR=23.24（梯度消失问题）

## 6. 应用场景

### 6.1 知识图谱补全
**任务**：预测缺失的头实体或尾实体
- 输入：(The Bourne Identity, starring, ?)
- 输出：Matt Damon
- 优势：无需枚举候选集，直接从实体空间预测

### 6.2 效率提升
- 训练epoch：从1000降至5（相比传统方法）
- 训练时间：<1小时（DB15K, 8×A100）
- 并行化：跨实体并行计算对比损失

### 6.3 实际部署优势
- 基于开源LLM（Llama-2-7B）
- 仅修改head layer，保持LLM主体frozen
- LoRA轻量级微调，参数效率高

## 7. 作者提出的缺点

### 7.1 灵活性受限
**问题**：不支持任意大的K值
- 实体名称长度有上限约束
- K过大导致计算成本显著增加

**未来方向**：探索滑动窗口机制处理K步预测

### 7.2 缺少多模态支持
**问题**：当前仅处理文本，未利用图像等其他模态
- 实验表明文本信息已足够强大
- 但图像等模态可能提供补充信息

**未来方向**：整合大型视觉-语言模型（LVLMs）到K-ON

### 7.3 隐含局限（分析补充）

**a) 实体粒度假设**
- 假设实体可表示为固定K个token
- 对复杂实体（如长描述）可能不适用

**b) 对比学习的负样本质量**
- 随机采样负样本可能不是最优策略
- 硬负样本挖掘可能进一步提升性能

**c) 泛化能力未知**
- 仅在KG补全任务验证
- 对其他实体相关任务（如实体链接、关系抽取）的迁移能力未测试

## 8. 关键术语解释

### 8.1 KG相关
- **知识图谱（KG）**：实体-关系-实体的三元组集合，如(人物, 出演, 电影)
- **KG补全**：预测缺失的头实体或尾实体
- **Filtered Ranks**：评估时过滤掉已知正确的三元组，避免惩罚正确预测

### 8.2 LLM技术
- **LoRA**：低秩适配，通过小参数量矩阵微调大模型
- **Causal Mask**：因果掩码，确保位置i只能看到位置≤i的信息
- **KL散度**：衡量两个概率分布差异的指标

### 8.3 对比学习
- **Contrastive Learning**：通过拉近正样本、推远负样本优化表示
- **NCE Loss**：噪声对比估计损失，常用于对比学习

### 8.4 评估指标
- **MRR**：正确答案排名倒数的平均值，排名越靠前分数越高
- **Hits@K**：Top-K准确率，正确答案在前K个预测中的比例

## 9. 技术洞察

### 9.1 设计哲学
K-ON的核心思想是**将KG集成到LLM的输出层而非输入层**，这种设计：
- 避免修改LLM主体结构
- 充分利用LLM的语言理解能力
- 在输出空间约束到KG实体集

### 9.2 与传统方法对比
| 维度 | 传统KG方法 | LLM方法 | K-ON |
|------|-----------|---------|------|
| 优化单位 | 实体向量 | Token | 实体（K-token组合）|
| 对比学习 | ✅ | ❌ | ✅ |
| 文本理解 | 有限 | 强大 | 强大 |
| 并行化 | ✅ | ❌ | ✅ |

### 9.3 理论创新
**分布轨迹对齐**：HTT通过对齐K步预测与原始单步预测的分布轨迹，解决了多步预测破坏条件依赖的问题，这是首次在多头LLM中系统性解决该问题的工作。

---

**总结**：K-ON通过巧妙的多头设计和实体级对比学习，在保持LLM文本理解能力的同时，实现了高效的知识图谱补全。然而其与正在兴起的原生multi-token prediction范式的关系，以及在更广泛实体相关任务上的适用性，仍需进一步探索。
