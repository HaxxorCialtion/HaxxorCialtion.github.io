---
title: "MemLLM: Finetuning LLMs to Use Explicit Read-Write Memory"
date: "2025-09-26"
tags: ["LLM", "Memory Systems", "RAG增强", "知识图谱", "微调", "三元组"]
description: "构造RAG数据库结构，微调LLM适配以提高benchmark score"
---

# MemLLM: Finetuning LLMs to Use Explicit Read-Write Memory

**日期：** 2025-9-26  
**链接：** [OpenReview](https://openreview.net/forum?id=dghM7sOudh)  
**标签：** `LLM` `Memory Systems` `RAG增强` `知识图谱` `微调`

## 核心观点

构造RAG数据库结构，微调LLM适配以提高benchmark score

## 技术架构

### 并非直接存储
不是参数化记忆，而是通过LLM微调提升与一个**外部的、结构化的动态知识库**交互的benchmark scores

### 知识库构建
知识库通过**三元组 (subject, relation, object)** 的形式存储事实，并使用特殊标记符号（如 `MEM_WRITE` 和 `MEM_READ`）进行操作

## 微调的核心作用

- 通过微调 **Mistral-7B**，让模型能够**自发判断**在生成某些关键信息之前，主动发起 `MEM_READ` 查询请求
- 训练模型**在关键节点生成有效查询的决策能力**，以及从文本中提取关系并写入知识库的能力 (`MEM_WRITE`)

## 技术特点

✅ **优势：**
- 将外部知识库与LLM有机结合
- 通过微调提升交互效率
- 结构化知识存储

⚠️ **局限：**
- 需要微调特定模型
- 知识库构建成本较高
- 依赖特定的标记符号系统

## 应用场景

- 需要频繁更新知识的问答系统
- 企业知识库管理
- 学术研究辅助工具
