---
title: "多头解码与推测解码研究调研汇总"
date: "2025-10-22"
tags: ["LLM", "DeepResearch", "MultiHead", "Decode",]
description: "多头解码和相关研究汇总，如果给了详细链接，则说明已经看过。"
---
# 多头解码与推测解码研究调研汇总

## 概述
本文档整理了关于大语言模型（LLM）中多头解码和推测解码的研究方法。这些方法主要用于解决自回归解码的速度瓶颈问题，通过并行生成多个token来加速推理过程。

---

## 一、经典多头推测解码方法

### 1. Medusa
**简介**: 在LLM最后一层添加多个解码头，每个头预测未来不同位置的token，使用树状注意力机制并行验证。这是多头推测解码的开创性工作之一。

**链接**: 搜索关键词 "Medusa speculative decoding" 可找到相关论文

---

### 2. Hydra
**简介**: 改进Medusa的顺序依赖多头结构，每个头会考虑前面候选token的信息，通过引入序列依赖性来提升预测准确率。

**链接**: 搜索关键词 "Hydra speculative decoding" 可找到相关论文

---

### 3. PyTorch 多阶段预测头
**简介**: 使用层级化的多头结构，每个阶段预测一个token并传递给下一阶段，采用级联方式进行预测。

---

## 二、基于特征级预测的方法

### 4. EAGLE (v1)
**简介**: 在特征层（倒数第二层）进行自回归预测而非token层，结合提前一步的token序列来解决特征级预测的不确定性问题。

**链接**: 搜索 "EAGLE speculative decoding" 或访问相关研究机构网站

---

### 5. EAGLE-2
**简介**: 引入动态草稿树结构，在推理过程中动态构建树而非使用固定结构，提高了灵活性和效率。

---

### 6. EAGLE-3
**简介**: 进一步改进EAGLE系列，使用训练时测试和多层融合技术来提升整体性能。

---

## 三、树结构优化方法

### 7. Sequoia
**简介**: 使用动态规划算法寻找最优的推测token树结构，引入无替换采样的验证方法，在不同温度参数下都表现良好。

**链接**: 搜索 "Sequoia speculative decoding"

---

### 8. SpecInfer
**简介**: 使用多个小模型的集成作为草稿模型，将它们的草稿聚合成树结构，使用树注意力进行并行验证。

**链接**: 搜索 "SpecInfer speculative decoding"

---

### 9. OPT-Tree
**简介**: 引入自适应分支策略，根据草稿置信度动态调整树结构，优化树的生成效率。

---

### 10. BiTA (Bidirectional Tree Attention)
**简介**: 通过双向调整实现无损加速，使用自执行树结构进行推测和验证。

---

## 四、基于检索的方法

### 11. REST (Retrieval-Based Speculative Decoding)
**简介**: 不使用独立的草稿模型，而是通过检索数据语料库直接构建草稿token序列，适合知识密集型任务。

**链接**: 搜索 "REST retrieval speculative decoding"

---

### 12. N-gram 推测解码
**简介**: 直接从输入提示和之前生成的输出中复制n-gram作为草稿token，特别适用于摘要、文档问答等有重复模式的场景。

---

### 13. Lookahead Decoding
**简介**: 使用两个并行计算分支：lookahead分支使用固定大小的2D窗口生成n-gram，验证分支验证n-gram候选。

**链接**: 搜索 "Lookahead Decoding LLM"

---

## 五、早期退出和自推测方法

### 14. LayerSkip
**简介**: 使用早期退出机制，在早期层退出进行草稿生成，用剩余层进行验证和修正，实现自推测解码，无需额外模型。

**链接**: 搜索 "LayerSkip speculative decoding"

---

### 15. Kangaroo
**简介**: 使用固定的浅层子网络作为自草稿模型，其余层作为目标模型，通过双重早期退出实现无损推测。

---

## 六、知识蒸馏和对齐方法

### 16. DistillSpec
**简介**: 使用知识蒸馏来改善草稿模型与目标模型的对齐，采用on-policy数据生成和针对任务定制的散度函数。

**链接**: 搜索 "DistillSpec knowledge distillation speculative"

---

### 17. ReDrafter
**简介**: 使用循环预测器，每个草稿token依赖于前一个token，在TensorRT引擎内执行logits预测、束搜索和草稿token接受。

**链接**: 搜索 "ReDrafter speculative decoding"

---

## 七、并行和流水线方法

### 18. SpecExec
**简介**: 大规模并行推测解码，专门针对消费级设备和RAM卸载场景进行优化，适合资源受限环境。

---

### 19. PEARL
**简介**: 提出预验证（在草稿阶段提前验证第一个token）和后验证（在验证阶段生成更多草稿token），实现自适应草稿长度。

---

## 八、验证策略改进

### 20. Block Verification
**简介**: 改进传统的逐token验证方式，提出联合验证整个块的算法，可提供5-8%的额外加速。

---

### 21. Mirror-SD
**简介**: 通过早期退出信号并行启动分支完整的rollout，打破了延迟与接受率之间的权衡关系。

---

## 九、其他创新方法

### 22. Multi-Candidate Speculative Decoding
**简介**: 使用多个候选序列而非单一序列来提高接受率，增加了推测的鲁棒性。

---

### 23. Speculative Streaming
**简介**: 无需辅助模型的快速推测解码方法，简化了部署流程。

---

### 24. ProPD
**简介**: 动态token树剪枝和生成方法，实时调整树结构以优化性能。

**链接**: 搜索 "ProPD speculative decoding"

---

### 25. Ouroboros
**简介**: 使用大模型增强的草稿生成，通过更强的模型提供更高质量的草稿。

---

### 26. TriForce
**简介**: 层次化推测解码，利用检索作为中间层，结合了检索和生成的优势。

---

### 27. TokenTiming
**简介**: 针对异构词汇表的动态对齐方法，使任何现成的off-the-shelf模型都能用于推测解码。

---

### 28. SpecDec++
**简介**: 通过自适应候选长度优化，使用训练的接受预测头动态决定何时停止推测，提高效率。

---

### 29. Lookahead Reasoning
**简介**: 专门针对推理模型，在步骤级别引入推测，与token级推测正交，可以叠加使用。

---

## 十、架构层面的创新方法

以下方法突破了传统推测解码范式，从模型架构层面进行创新：

### 30. FourierNAT (非自回归Transformer)
**简介**: 在解码器中集成基于傅里叶变换的混合层，使用频域门控在整个序列维度上混合token embeddings，实现全局上下文的即时传播，无需显式的自回归步骤。

**链接**: 搜索 "FourierNAT non-autoregressive transformer"

---

### 31. DisCo Transformer (Disentangled Context)
**简介**: 基于注意力掩码的模型，给定不同的上下文同时生成所有token，训练模型预测每个输出token时基于其他参考token的任意子集。

**链接**: 搜索 "DisCo Disentangled Context Transformer"

---

### 32. Paraformer
**简介**: 使用连续的integrate-and-fire预测器来预测token数量并生成隐藏变量，再通过Glancing Language Model (GLM)采样器生成语义嵌入来增强NAR解码器建模token间依赖的能力。

**链接**: 搜索 "Paraformer non-autoregressive"

---

### 33. NARVL (Non-Autoregressive Vision-Language)
**简介**: 使用可学习的查询token序列（而非顺序生成），并行生成所有token的输出，配合Query-CTC损失函数。

**链接**: 搜索 "NARVL non-autoregressive vision language"

---

### 34. MDLM (Masked Diffusion Language Models)
**简介**: 通过迭代去噪过程生成文本，从完全掩码的序列开始，使用双向注意力机制，可以并行处理整个序列。这是扩散模型在文本生成中的应用。

**链接**: 搜索 "MDLM Masked Diffusion Language Models"

---

### 35. Whisfusion
**简介**: 融合预训练Whisper编码器与文本扩散解码器，这种NAR架构通过在每个解码步骤并行处理整个声学上下文来解决AR延迟瓶颈。

**链接**: 搜索 "Whisfusion diffusion decoder"

---

### 36. Soft-Masked Diffusion
**简介**: 改进标准的二元掩码，通过将掩码token与之前预测的top-k进行叠加，为下一步解码提供更丰富的反馈。

---

### 37. HDLM (Hierarchical Diffusion LM)
**简介**: 构建层次化词汇表，低层token具有详细语义，高层token具有粗粒度含义，实现时变的下一语义尺度预测。

**链接**: 搜索 "HDLM Hierarchical Diffusion Language Model"

---

### 38. BPDec (BERT Pretraining Decoder)
**简介**: 为BERT的编码器模型设计增强的解码器用于预训练，关注被低估的增强masked language modeling decoder的设计和研究。虽然主要用于理解任务，但其双向架构可以启发多头生成设计。

**链接**: 搜索 "BPDec BERT decoder"

---

### 39. Parallel Prompt Decoding (PPD)
**简介**: 采用提示调优进行非自回归LLM推理的新型框架，使用成本效益高的prompt tokens实现长距离token预测的高接受率。

**链接**: 搜索 "Parallel Prompt Decoding PPD"

---

### 40. 插入Transformer & Levenshtein Transformer
**简介**: 在灵活性和质量之间取得平衡，同时实现部分并行性，属于Masked Generation范式。

**链接**: 搜索 "Insertion Transformer Levenshtein Transformer"

---

### 41. CTC-based 方法
**简介**: 使用连接时序分类（Connectionist Temporal Classification）算法，在非自回归生成中引入依赖关系。

---

## 十一、架构对比与分析

### 架构类型对比表

| 架构类型 | 核心创新 | 多头/并行方式 | 与推测解码的区别 |
|---------|---------|--------------|----------------|
| **NAT系列** | 完全非自回归 | 所有位置同时预测 | 不需要验证步骤，一次性生成 |
| **扩散模型** | 迭代去噪 | 每步并行处理整个序列 | 不是草稿-验证，而是渐进refinement |
| **BERT式** | 双向编码+MLM | Masked位置并行预测 | 预训练架构，不是为生成设计 |
| **推测解码** | 草稿+验证 | 多头预测未来token | 需要验证阶段，仍依赖AR模型 |

### 建模假设对比

- **传统AR**: P(x₁)P(x₂|x₁)P(x₃|x₁x₂)...
- **NAT**: P(x₁,x₂,x₃,...|context) 
- **扩散**: P(x_clean|x_noisy) 迭代多次

---

## 十二、关键问题讨论

### 多头上下文构建问题

**问题**: 多轮对话中，如何构建上下文？多头的上文完全一样，是否会导致无法合理地让多头分配不同的预测任务？

**主流做法**: 大多数方法确实使用相同的上下文给所有预测头

**差异化实现方式**:
1. **独立参数**: 不同的线性层/MLP
2. **预测位置不同**: 预测 t+1, t+2, t+3...
3. **Hydra的顺序依赖**: 让每个头看到前面头的预测
4. **EAGLE的特征级预测**: 使用倒数第二层+提前一步的token
5. **树状结构**: 天然引入路径差异

### 改进方向

1. **引入显式的角色分化**: 给不同的头添加可学习的role embedding
2. **使用不同的上下文窗口**: 不同头使用不同长度或筛选策略的上下文
3. **Staged Decoding**: 级联式预测，使用上一头的预测作为下一头的输入
4. **对比学习 + 多样性正则**: 在训练时加入diversity loss

---

## 总结

本调研涵盖了41种多头解码和推测解码的研究方法，从经典的推测解码（如Medusa、EAGLE）到架构创新（如扩散模型、非自回归Transformer），展现了该领域的广泛探索。

**主要发现**:
1. 推测解码方法在保持模型质量的同时能显著提升推理速度
2. 架构层面的创新（NAT、扩散模型）提供了更本质的并行化方案
3. 相同上下文的多头设计已被证明有效，但仍有优化空间
4. 树结构、检索、知识蒸馏等技术可进一步提升性能

**研究趋势**:
- 从单纯的加速到加速与质量的平衡
- 从依赖草稿模型到自推测方法
- 从固定结构到动态自适应结构
- 从token级到特征级、步骤级的推测

---

## 参考资源

由于篇幅限制，具体论文链接建议通过以下方式获取：
- Google Scholar: 搜索对应方法名称
- arXiv.org: 搜索相关关键词
- GitHub: 搜索项目实现代码
- Papers with Code: 查找基准测试结果

**搜索建议**: 在搜索引擎中输入方法名称 + "speculative decoding" 或 "non-autoregressive" 等关键词即可找到相关论文。
