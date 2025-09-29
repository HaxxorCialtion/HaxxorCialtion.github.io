---
title: "RAISE Agent: From LLM to Conversational Agent"
date: "2025-09-26"
tags: ["传统方法", "上下文工程", "蒸馏学习", "ReAct", "对话系统"]
description: "ReAct + Context工程的记忆增强架构，展示微调在Agent领域的蒸馏学习价值"
---

# RAISE Agent: From LLM to Conversational Agent

**日期：** 2025-9-26  
**链接：** [ArXiv](https://arxiv.org/abs/2401.02777)  
**标签：** `传统方法` `上下文工程` `蒸馏学习`

## 核心观点

传统方法：ReAct + Context工程的记忆增强架构

## 架构设计 (RAISE)

### 短期记忆 (Scratchpad)
动态更新的 `Context`，记录当前对话状态，避免重复工作。

### 长期记忆 (Examples)
基于向量检索的 `Few-shot RAG`，用于模仿回答风格和内容。

## 数据工程策略

### 数据生成
使用 GPT-4 为真实对话场景生成高质量的 CoT (思维链) 数据

### 场景增强
构造负样本来微调模型，教其"拒绝与"承认不知"，以抑制幻觉

## 技术评估

这是一套传统的 **Prompt 和 Context 工程**。其核心技术点在于展示了 **LLM 微调在 Agent 领域的意义在于蒸馏学习**。

## 技术特点

✅ **优势：**
- 实现简单，工程成熟
- 数据工程策略清晰
- 蒸馏学习效果明显

⚠️ **局限：**
- 方法相对传统
- 缺乏创新突破
- 依赖大量标注数据

## 应用价值

- 为Agent系统提供基础架构参考
- 展示微调在对话系统中的作用
- 提供数据工程的最佳实践
