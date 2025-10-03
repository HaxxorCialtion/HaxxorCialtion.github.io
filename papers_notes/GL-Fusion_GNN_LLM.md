---
title: "GL-Fusion: Rethinking GNN-LLM Integration"
date: "2025-10-03"
tags: ["GNN", "LLM", "Graph-Learning",  ]
description: "深度融合GNN和LLM的创新架构，通过Structure-Aware Transformers、Graph-Text Cross-Attention和Twin Predictor实现图结构与文本语义的协同建模"
---

## 概览

**GL-Fusion**是Peking大学2024年提出的GNN-LLM深度集成架构，解决了现有方法的三大问题：
1. **信息压缩损失**：GNN-centered方法将文本压缩为固定向量，LLM-centered方法压缩图结构
2. **任务单一性**：GNN无法生成语言，LLM无法并行预测节点
3. **独立编码**：图结构和文本语义分别编码，缺乏交互
4. **复杂度**: 围绕LLM的模型结构做了非常大的调整，模型较为复杂。

**核心贡献**：
- 在OGBN-Arxiv达到**78.20% SOTA**（vs GLEM 76.12%）
- 在OGBG-Code2文本生成任务F1提升至**40.97%**（vs 最佳22.22%）
- 统一架构支持节点分类、链接预测、图生成、QA等多任务

---

## 核心创新

### 1. Structure-Aware Transformers

**突破传统因果mask限制**，引入图感知的注意力机制：

```python
# 标准Llama-3: 严格下三角mask
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]

# GL-Fusion: 图内全连接 + 文本因果
[[1, 0, 0, ... | 0, 0, ... | 0],  ← 文本token
 [1, 1, 0, ... | 0, 0, ... | 0],
 [1, 1, 1, 1,  | 1, 1, ... | 0],  ← 图节点（内部全连接）
 [1, 1, 1, 1,  | 1, 1, ... | 1]]  ← 后续文本（看全部）
```

**关键设计**：
- **共享位置编码**：图节点使用相同PE，保证置换不变性
- **Message Passing嵌入**：在第0,4,8,...层执行MPNN，用三聚合器（mean/max/std）
- **门控融合**：`t ← t + tanh(W·t)·h_gnn`，初始化W=0，渐进引入GNN信息

### 2. Graph-Text Cross-Attention

**核心思想**：节点文本不压缩，通过侧路保留完整语义，按需提取。

**两阶段注意力**：
```python
# Stage 1: 聚合节点内部256个tokens
query = hidden[token_i]  # [4096]
keys = node_text[node_j]  # [256, 4096]
weights = softmax(query @ keys.T)  # [256]
node_repr = weights @ keys  # [4096]

# Stage 2: 聚合50个节点
all_nodes = [node_repr_0, ..., node_repr_49]  # [50, 4096]
weights2 = softmax(query @ all_nodes.T)  # [50]
output = weights2 @ all_nodes  # [4096]
```

**复杂度优化**：O(nLn·Lt) vs 朴素方法O((nLn+Lt)²)

**访问策略**：
- 图节点token：只访问自己的完整文本（避免节点混淆）
- 后续文本token：访问所有节点的完整文本

### 3. GNN-LLM Twin Predictor

**双输出架构**：
```python
# GNN分支（位置20-69的图节点）
gnn_logits = Linear(hidden[target_pos])  # [num_classes]
# 优势：并行预测所有节点，适合分类任务

# LLM分支（位置-1的最后token）
lm_output = AutoRegressive_Generate(hidden[-1])
# 优势：生成自然语言，适合文本任务

# Ensemble
final_pred = combine(gnn_logits, lm_output)
```

**训练策略**：`Loss = CE(GNN) + CE(LM)`，联合优化

---

## 模型架构

### 三条并行Pipeline

```
Pipeline 1: 主序列（Llama-3 Tokenizer）
"Classify paper <graph_start><node>×50<graph_end> Answer:"
→ [73, 4096]  进入Transformer主干

Pipeline 2: 节点文本（LLM2Vec）
["Attention Is All You Need. The dominant...", ...]
→ [50, 256, 4096]  侧路，Cross-Attention时使用

Pipeline 3: 边特征（Edge Encoder）
["cites", "cites", "collaborated", ...]
→ [120, 4096]  侧路，Message Passing时使用
```

### 完整前向传播

```python
# Layer i (i=0..31)
hidden = structure_aware_self_attention(hidden)  # 全序列[73,4096]

if i in [0,4,8,12,16,20,24,28]:
    graph_part = hidden[:, 20:70, :]
    graph_part = message_passing(graph_part, edge_embeds)  # 用Pipeline 3
    hidden[:, 20:70, :] = graph_part  # 覆盖更新

if i in [3,7,11,15,19,23,27,31]:
    for pos in range(73):
        cross_out = cross_attention(hidden[pos], node_texts)  # 用Pipeline 2
        hidden[pos] += cross_out

hidden = ffn(hidden)
```

### 参数分配

| 模块 | 参数量 | 训练策略 |
|------|--------|---------|
| Llama-3 Base | 7.984B | **冻结** |
| LoRA (主模型) | 16M | 可训练 (rank=64) |
| LLM2Vec Base | 7.984B | **冻结** |
| Message Passing | 1.6B | 从头训练 |
| Cross-Attention | 536M | 从头训练 |
| GNN Classifier | 164K | 从头训练 |
| **总可训练** | **~2.15B (27%)** | |

---

## 数据流详解

### OGBN-Arxiv数据集

**基本信息**：
- 169,343篇CS论文，116万引用关系
- 40个学科类别（cs.AI, cs.CL, cs.CV, ...）
- 时间划分：训练(2007-2017) / 验证(2018) / 测试(2019+)

**样本结构**：
```python
paper_12345 = {
    "title": "Attention Is All You Need",
    "abstract": "The dominant sequence transduction...",
    "label": 5,  # cs.CL (Computation and Language)
    "citations": [2341, 5678, 8901],  # 2跳采样→50节点子图
    "year": 2017
}
```

### 输入构造

```python
# 动态子图采样
subgraph = sample_k_hop_neighbors(target=12345, k=2, max_nodes=500)
# → 50个节点，120条边

# 三条Pipeline的输入
input_data = {
    "main_sequence": [
        "Classify paper into 40 categories. Graph:",
        "<graph_start>", "<node>"×50, "<graph_end>", "Answer:"
    ],  # [73] tokens
    
    "node_texts": [
        "Attention Is All You Need. The dominant...",  # 256 tokens
        "[Label: cs.CL] Neural Machine Translation...",  # 训练集加标签
        ...
    ],  # [50, 256]
    
    "edge_data": [
        (0, 1, "cites"), (0, 2, "cites"), ...
    ]  # 120条边
}
```

### 训练流程

```python
# 超参数
config = {
    "base_model": "Llama-3-8B",
    "lora_rank": 64,
    "learning_rate": 3e-5,
    "batch_size": 1,  # 子图大小不同
    "epochs": 1,  # 单epoch足够
    "optimizer": "AdamW",
    "weight_decay": 0.1
}

# 训练循环
for target_node in train_set:  # 90,941个样本
    # 1. 采样子图
    subgraph = sample_subgraph(target_node)
    
    # 2. 三路编码
    main_embeds = llama3_embedding(main_seq)
    node_embeds = llm2vec_encoder(node_texts)
    edge_embeds = edge_encoder(edges)
    
    # 3. 前向传播（32层Transformer）
    hidden = model(main_embeds, node_embeds, edge_embeds)
    
    # 4. 双损失
    gnn_loss = CE(gnn_classifier(hidden[20]), label)
    lm_loss = CE(lm_head(hidden[-1]), "Computation and Language")
    total_loss = gnn_loss + lm_loss
    
    # 5. 反向传播（只更新可训练部分）
    total_loss.backward()
    optimizer.step()
```

**防过拟合策略**：
- 参数冻结（99%预训练参数）
- Dropout=0.1, Weight Decay=0.1
- 子图随机采样（数据增强）
- 早停（patience=3）

---

## 关键技术细节

### 1. Self-Attention的因果Mask

```python
# 输入: "The cat sat"
tokens = [CLS, The, cat, sat]

# 标准因果mask
mask = [[1, 0, 0, 0],
        [1, 1, 0, 0],  # "cat"不能看到"sat"
        [1, 1, 1, 0],
        [1, 1, 1, 1]]

# Attention计算
scores = Q @ K.T / √d  # [4, 4]
masked = scores + (1-mask)*(-1e9)  # 上三角→-∞
weights = softmax(masked)  # [4, 4]，上三角→0
output = weights @ V  # [4, dim]
```

**作用**：防止训练时"看到未来"，确保自回归生成的一致性。

### 2. Message Passing执行时机

```python
# Layer 0完整流程
def layer_0(hidden):
    # 时刻1: Self-Attention（全部[73,4096]参与）
    hidden = self_attention(hidden)
    
    # 时刻2: Message Passing（仅20:70参与）
    for node_u in range(50):
        neighbors = [node_v for v in adjacency[u]]
        msgs = [hidden[20+v] * edge_embeds[u,v] for v in neighbors]
        agg = concat([mean(msgs), max(msgs), std(msgs)])  # [12288]
        h_new = Linear(concat([hidden[20+u], agg]))  # [4096]
        hidden[20+u] = hidden[20+u] + gate*h_new  # 门控更新
    
    # 时刻3: FFN（全部参与）
    hidden = ffn(hidden)
    
    return hidden
```

**边信息作用**：`msg = h_neighbor * e_edge`，边特征调制邻居信息。

### 3. LLM2Vec vs 主Transformer

| 特性 | LLM2Vec | GL-Fusion主模型 |
|------|---------|----------------|
| 基础 | Llama-3-8B | Llama-3-8B |
| Mask | **双向**（无因果限制） | 特殊mask（图双向+文本因果） |
| 训练 | Sentence Similarity | 节点分类+文本生成 |
| 输出 | [256, 4096] 每个token | [73, 4096] 主序列 |
| 参数 | 独立，最后一层LoRA | LoRA全层 |

**对齐机制**：通过Cross-Attention的W_Q和W_K学习空间对齐。

### 4. Cross-Attention的W_Q和W_K

```python
# 可学习的投影矩阵
W_Q1 = nn.Linear(4096, 4096)  # 投影query（主序列）
W_K1 = nn.Linear(4096, 4096)  # 投影key（节点文本）

# 计算过程
query = hidden[70, :]  # "Answer:"的表示 [4096]
keys = node_texts[0, :, :]  # Node 0的256个tokens [256, 4096]

Q = W_Q1(query)  # [4096]
K = W_K1(keys)   # [256, 4096]

scores = Q @ K.T / √d  # [256]
weights = softmax(scores)  # 关注"Attention", "Transformer"等关键词
output = weights @ keys  # [4096]
```

**作用**：将主序列空间和节点文本空间映射到同一"语义比较空间"。

---

## 实验结果

### 节点分类（OGBN-Arxiv）

| 模型 | Test Accuracy | 说明 |
|------|--------------|------|
| GCN | 71.47% | 传统GNN |
| GLEM | 76.12% | GNN-centered |
| GraphGPT | 75.11% | LLM-centered |
| **GL-Fusion** | **78.20%** | **SOTA** |

### 少样本学习（OGBN-Arxiv）

| #shots | GIANT | PLM-sparse | G-Prompt | **GL-Fusion** |
|--------|-------|------------|----------|---------------|
| 10 | 51.4% | 51.17% | 52.48% | **56.44%** |
| 100 | 61.26% | 58.65% | 61.67% | **68.18%** |

### 知识图谱补全（FB15k-237-ind v1）

| 模型 | MRR | Hits@1 | Hits@10 |
|------|-----|--------|---------|
| NBFNet | 0.625 | 0.519 | 0.834 |
| UniLP | 0.754 | 0.672 | 0.921 |
| **GL-Fusion** | **0.856** | **0.731** | **0.983** |

### 图到文本生成（OGBG-Code2）

| 模型 | Test F1 | 任务 |
|------|---------|------|
| GraphTrans | 18.30% | 5000类分类 |
| SAT++ | 22.22% | 5000类分类 |
| **GL-Fusion** | **40.97%** | **直接生成函数名** |

### 常识问答（CommonsenseQA）

| 模型 | Accuracy |
|------|----------|
| GPT-3 | 73.0% |
| QA-GNN | 76.1% |
| **GL-Fusion** | **81.79%** |

---

## 消融实验（OGBN-Arxiv）

| 配置 | GNN Acc | LLM Acc | Ensemble |
|------|---------|---------|----------|
| 完整模型 | 77.09% | 76.43% | **78.20%** |
| w/o Cross-Attn | 75.50% | 74.97% | 76.35% |
| w/o Gate | 75.88% | 73.33% | 76.20% |
| w/o Multiple Aggr | 75.40% | 75.57% | 76.48% |
| w/o GNN Pred | - | 72.45% | - |
| w/o Text Pred | 75.80% | - | - |

**结论**：所有模块都对性能有贡献，Cross-Attention影响最大。

---

## 局限性

1. **任务覆盖不全**：未测试分子生成、轨迹预测等其他图任务
2. **缺乏统一预训练**：每个任务单独训练，未建立foundation model
3. **可靠性风险**：继承LLM的幻觉问题，可能生成错误信息
4. **计算成本**：8B模型+大图处理，推理需要40GB GPU

---

## 核心洞察

### 为什么不损害文本能力？

1. **残差连接保护**：`hidden += cross_out`，初始时cross_out≈0
2. **选择性激活**：图前的文本完全不受影响
3. **零初始化**：Cross-Attention的输出投影初始化为0
4. **实验验证**：纯文本任务（MMLU）仅下降0.3%

### 为什么三路模型不冲突？

```python
# 梯度流分离
gnn_loss → gnn_classifier → message_passing → graph_nodes[20:70]
                                            ↘ main_transformer_lora

lm_loss → lm_head → text_tokens[70:73] → cross_attention → llm2vec
                                       ↘ main_transformer_lora

# 只在main_transformer_lora处汇合，梯度叠加但不冲突
```

### Message Passing vs Residual Connection

```python
# 错误理解
hidden[20:70] = hidden[20:70] + mp_output  # ✗

# 正确理解
hidden[20:70] = mp_output  # ✓ 覆盖更新

# 但MP内部有软残差
h_final = h_original + gate * h_new  # gate初始≈0
```

---

## 设计原则总结

1. **模块化交互**：三条Pipeline通过主序列hidden states通信，避免直接耦合
2. **渐进式融合**：门控机制让GNN信息缓慢引入，稳定训练
3. **无损信息传递**：Cross-Attention保留完整文本，避免压缩损失
4. **灵活输出**：Twin Predictor兼顾效率（GNN）和表达力（LLM）

**核心哲学**：不是简单拼接GNN和LLM，而是让Transformer **同时成为**GNN和LLM。

---

## 参考资源

- 论文：arXiv:2412.06849v1 [cs.LG] 8 Dec 2024
- 代码：github.com/PKU（待发布）
- 数据集：OGBN-Arxiv, FB15k-237-ind, OGBG-Code2
- 基础模型：Llama-3-8B, LLM2Vec