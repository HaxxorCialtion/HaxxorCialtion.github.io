---
title: "WISE: 基于侧记忆的终身模型编辑方法"
date: "2025-10-11"
tags: ["Model Editing", "Memory Architecture", "LLM", "Parameter Isolation"]
description: "在特定FNN层中，编辑双参数层，类似分离lora结果层和原层，实现对记忆机制（主记忆+侧记忆）和激活驱动路由"
---

# WISE: 侧记忆终身编辑方法核心笔记
https://arxiv.org/abs/2210.03629
## 📌 一句话总结

**WISE通过复制FFN的down_proj参数创建侧记忆，用激活差异的L2范数做门控路由，实现了参数化知识编辑+检索式知识隔离的双重优势。**

---

## 🎯 核心问题

### 不可能三角
现有终身模型编辑方法无法同时满足：
- **Reliability**：记住所有历史编辑
- **Generalization**：理解知识而非记忆query-target
- **Locality**：不影响无关预训练知识

### 两类方法的困境
| 方法类型 | 代表 | 问题 |
|---------|------|------|
| 编辑长期记忆（参数） | ROME, MEMIT | ❌ Locality差（与预训练知识冲突） |
| 编辑工作记忆（检索） | GRACE | ❌ Generalization差（检索表示无法泛化） |

**关键发现**：GRACE的generalization失败是因为**自回归LM的最后token表示无法有效表达语义**，导致paraphrase无法命中codebook。

---

## 💡 WISE核心创新

### 双记忆架构
```
Main Memory (Wv):  冻结的预训练参数，保护原知识
Side Memory (Wv'): 可编辑的参数副本，存储新知识
                   ↑ 这是"mid-term memory"
```

### 三大技术模块

#### 1️⃣ Memory Routing（门控路由）- 核心机制

**物理位置**：第26层FFN的down_proj矩阵
- `Wv`: (4096, 11008) 主记忆，冻结
- `Wv'`: (4096, 11008) 侧记忆，可编辑
- 仅0.18GB，占模型0.64%

**路由公式**：
```python
# 计算激活指示器
activated = gate ⊙ up  # FFN中间激活，shape: (seq_len, 11008)
Δact(x) = ||activated @ (Wv' - Wv)||₂  # L2范数，标量

# 推理时路由决策
if Δact(x) > ε:
    output = activated @ Wv'  # 使用侧记忆（编辑知识）
else:
    output = activated @ Wv   # 使用主记忆（预训练知识）
```

**关键特点**：
- 不是独立神经网络，而是**计算规则**
- 11008维激活 → 1维标量（信息压缩率0.009%）
- ε是训练中记录的最小编辑激活值

#### 2️⃣ Knowledge Sharding（知识分片）

**目的**：避免知识密度过高导致冲突

```python
# 生成k个随机mask（比例ρ）
k = 2, ρ = 0.2
M_i ~ Bernoulli(ρ), i ∈ [k]

# 在子空间中编辑
for shard_i in range(k):
    grad = compute_gradient(loss)
    Wv' -= lr * (M_i ⊙ grad)  # 只更新mask=1的参数
```

**定理2.1**：k个随机mask的重叠参数 = ρᵏ × |Wv'|
- k=2, ρ=0.2 → 重叠4%参数
- 这4%充当"锚点"(anchor)，帮助后续合并

#### 3️⃣ Knowledge Merging（知识合并）

**Ties-Merge三步骤**：
1. **Trim**：修剪小幅度更新
2. **Elect Sign**：投票选择参数更新方向（+/-）
3. **Disjoint Merge**：同向参数求平均

```python
Wv' ← Wv + Ties(τ₁, τ₂, ..., τₖ; Wv)
# τᵢ = Wvᵢ' - Wv（第i个分片的权重偏移）
```

---

## 🔥 门控机制详解（重点）

### 训练阶段：如何让门控自涌现

**损失函数**：
```python
L_edit = -log P(y_e | x_e)  # 标准自回归损失

L_routing = max(0, Δact(x_irr) - α) +        # 无关样本激活 < α
            max(0, β - Δact(x_edit)) +       # 编辑样本激活 > β  
            max(0, γ - (Δact(x_edit) - Δact(x_irr)))  # 保持margin γ

L_total = L_edit + L_routing
```

**超参数**（α=5, β=20, γ=10）：
- α：无关样本激活上界
- β：编辑样本激活下界  
- γ：两者最小间隔

**训练过程**：
```python
optimizer = SGD([Wv'], lr=1.0)
epsilon = float('inf')

for (x_e, y_e), x_irr in zip(edit_data, irrelevant_data):
    # 1. 前向传播（同时计算两个memory）
    a = activation(x_e @ Wk)
    out_main = a @ Wv
    out_side = a @ Wv'
    
    # 2. 计算激活指示器
    delta_edit = torch.norm(a @ (Wv' - Wv), p=2)
    delta_irr = compute_delta(x_irr)
    
    # 3. 联合损失优化
    loss = L_edit + L_routing
    loss.backward()
    
    # 4. 子空间梯度更新
    grad = Wv'.grad
    Wv' -= lr * (M ⊙ grad)  # mask掉不需要更新的参数
    
    # 5. 动态更新阈值
    epsilon = min(epsilon, delta_edit)
```

**关键洞察**：
- 路由器不需要单独训练
- 通过L_routing，Wv'自动学会：
  - 对编辑样本产生大激活差异
  - 对无关样本产生小激活差异
- ε是"见过的最保守边界"

### 推理阶段：实际路由流程

```python
def forward_layer_26(x):
    # 标准FFN计算
    gate = F.silu(x @ Wgate)
    up = x @ Wup
    activated = gate * up  # shape: (seq_len, 11008)
    
    # ⚠️ 关键：必须同时计算两个输出
    out_main = activated @ Wv.T   # 45M FLOPs
    out_side = activated @ Wv'.T  # 45M FLOPs
    
    # 计算激活指示器
    delta = torch.norm(out_side - out_main, p=2)  # 标量
    
    # 路由决策（硬路由，非加权）
    if delta > epsilon:
        return out_side  # 编辑知识路径
    else:
        return out_main  # 预训练知识路径
```

**开销分析**：
- 理论：2倍矩阵乘法
- 实际：+3% latency（GPU并行优化）
- 原因：两个矩阵乘可高度并行

**两种变体**：
```
WISE-Merge:   合并所有分片 → 1个侧记忆 → 恒定开销+3%
WISE-Retrieve: 保留多个侧记忆 → Top-1激活检索 → 开销随编辑数增长+7%@3K
```

---

## 📊 实验结果

### 主要数据集
| 数据集 | 任务 | 样本 | 特点 |
|--------|------|------|------|
| ZsRE | 问答 | 1K | 标准QA |
| SelfCheckGPT | 去幻觉 | 600 | 长文本(17-254 tokens) |
| Temporal | OOD泛化 | 100 | 新兴实体(2019后) |

### 核心结果（T=1000, LLaMA-2-7B, ZsRE）

| 方法 | Rel. | Gen. | Loc. | Avg. |
|------|------|------|------|------|
| GRACE | 0.97 | 0.08 | 1.00 | **0.68** |
| MEMIT-MASS | 0.69 | 0.65 | 0.62 | 0.65 |
| **WISE** | **0.77** | **0.72** | **1.00** | **0.83** |

**关键发现**：
- WISE比GRACE平均高15%
- 首次在三指标上达到相对均衡
- Locality始终保持1.00（知识隔离成功）

---

## ⚠️ 局限性（重点关注）

### 1. 门控泛化性差

**根本原因**：
```
11008维激活 → 1维标量Δ
信息压缩率 = 0.009%
→ 不同语义的query可能有相似Δ
```

**实验证据**：
```
WISE-Retrieve路由准确率：
T=1000: 95%  ✓
T=2000: 75%  ⚠️
T=3000: 60%  ❌

即使加replay优化，仍有12%误差
```

**失效场景**：
- ❌ 多领域混合编辑（ε无法适配多个分布）
- ❌ OOD query（激活模式超出训练分布）
- ❌ 罕见实体（参数容量不足）
- ❌ 对抗样本（可构造Δ ≈ ε）

### 2. 参数容量有限

**理论容量**：45M参数 / 100参数/编辑 = 450K编辑
**实际瓶颈**：3K编辑已出现性能下降

**Case Study失败类型**：
- 罕见实体（如"Persian"）：完全失败
- 部分token错误：占多数bad cases
- 泛化不完全：答"English"而非"American English"

### 3. 推理双重计算

**不可避免**：必须同时计算Wv和Wv'才能得到Δ
- WISE-Merge: +3% latency（可接受）
- WISE-Retrieve: +7% latency @ 3K edits

### 4. 跨域泛化弱

**Temporal OOD实验**：
```
WISE OOD Gen = 0.37 (绝对值低)
训练：简单规范描述
测试：复杂自然文本
→ ε分布不匹配
```

---

## 🆚 与其他方法对比

### 技术维度

| 维度 | ROME/MEMIT | GRACE | WISE |
|------|-----------|-------|------|
| 编辑对象 | 直接改Wv | codebook | 复制的Wv' |
| 推理路由 | 无 | 高维检索 | 1维阈值 |
| 泛化能力 | ★★★★★ | ★☆☆☆☆ | ★★★☆☆ |
| 知识隔离 | ✗ | ✓ | ✓ |
| 开销 | +0% | +5% | +3% |

### PEFT方法对比

```
LoRA:  x → (Wv + A×B)(x)  # 低秩，所有输入共享
WISE:  x → Wv(x) or Wv'(x)  # 全秩，动态选择
```

**WISE不是传统PEFT**：
- 目标：持续编辑（非任务适配）
- 机制：路由选择（非参数叠加）
- 定位：参数隔离 + 动态路由

---

## 💭 核心启示

### 1. Mid-term Memory的本质

**不是新模块，是参数管理策略**：
- 空间：物理隔离（Wv vs Wv'）
- 时间：分片编辑（避免密度极端）
- 路由：激活驱动（动态选择空间）

### 2. 为什么门控是瓶颈

**三重困境**：
1. **信息瓶颈**：11008维 → 1维压缩
2. **分布固化**：ε是训练时的最小值，无法适应新domain
3. **容量限制**：45M参数需编码多个领域知识

### 3. WISE的适用边界

**✓ 适用场景**：
- 单领域、<1K编辑
- 格式化query（如"Who/What/Where"）
- 高频实体

**✗ 不适用场景**：
- 多领域混合
- >3K编辑
- OOD泛化需求高

---

## 🔧 可能的改进方向

### 1. 学习的路由器
```python
class LearnedRouter(nn.Module):
    def __init__(self):
        self.fc = nn.Linear(11008, 2)  # 分类器
    
    def forward(self, activated):
        return self.fc(activated.mean(0))  # 输出main/side概率
```

### 2. 动态阈值
```python
epsilon = {
    "geography": 8.5,
    "biology": 12.3,
    ...
}
```

### 3. 分层路由
```python
if delta < ε1: return Wv_main
elif delta < ε2: return Wv_side_1
elif delta < ε3: return Wv_side_2
else: return Wv_side_3
```

---

## 📚 相关论文

### GRACE（对比基线）
**论文**: "Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors"  
**链接**: https://arxiv.org/abs/2211.11031  
**发表**: NeurIPS 2023

**核心思想**：
- 维护离散codebook存储编辑
- 推理时检索最近邻KEY替换激活
- 问题：检索表示无法泛化（尤其在自回归LM）

### ROME & MEMIT
**ROME**: https://arxiv.org/abs/2202.05262  
**MEMIT**: https://arxiv.org/abs/2210.07229  
直接定位编辑FFN参数，但缺乏知识隔离

---

## 🎓 快速回忆检查清单

看到这份笔记时，问自己：

- [ ] 不可能三角是什么？（Reliability/Generalization/Locality）
- [ ] WISE的核心是什么？（双记忆+激活路由）
- [ ] Wv和Wv'指的是什么？（第26层FFN的down_proj矩阵）
- [ ] 门控公式是什么？（Δ = ||a@(Wv'-Wv)||₂，与ε比较）
- [ ] 训练时如何让门控有效？（L_routing的margin约束）
- [ ] 推理时计算开销如何？（必须算两次，+3%）
- [ ] 最大的局限是什么？（门控泛化性差，3K编辑路由准确率60%）
- [ ] 适用场景是什么？（单领域<1K编辑，格式化query）

---

**论文信息**  
标题: WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models  
机构: Zhejiang University & Alibaba Group  
发表: arXiv:2405.14768v1 [cs.CL] 23 May 2024  
代码: https://github.com/zjunlp/EasyEdit