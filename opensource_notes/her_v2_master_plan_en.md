---
title: "24-Hour AI Companion System Overview Plan"
date: "2025-10-22"
tags: ["AI", "OpenSource", "AI-Companion", "Alpha"]
description: "Her"
---

# Her-v2 Evolution Plan: Building a 24-Hour AI Companion System

## The Journey from ASR-LLM-TTS to ASR-Agents-TTS

> **Project Vision**: Create a true 24-hour AI companion that not only converses, but truly understands your life, remembers your stories, perceives your emotions, and integrates into your social network.

---

## 🎯 Core Design Philosophy

### Three Levels of Anthropomorphization

**Level 1: Perceptual Anthropomorphization**  
Not just hearing what you say, but understanding "how you say it" — through multi-dimensional VAD and emotion recognition, perceiving your tone, emotions, and intentions, like a true listening friend.

**Level 2: Memory Anthropomorphization**  
Not mechanical database retrieval, but human-like memory — some things remembered clearly, some gradually forgotten, some suddenly recalled in specific contexts. Building an interconnected knowledge network through the A-MEM system.

**Level 3: Interactive Anthropomorphization**  
Not passive responses, but proactive care — morning greetings, weather change reminders, comfort when noticing you're feeling down, even timely humorous banter, making AI part of life rather than just a tool.

### Ultimate Goal of Human-Computer Interaction

**Low Latency**: End-to-end <1.5s for natural, fluid conversation  
**High Intelligence**: Complex task handling through Agents architecture  
**Strong Memory**: Building user social networks, understanding relationship contexts  
**Emotional Resonance**: Recognizing and responding to emotional changes  
**Environmental Awareness**: Passive listening with intelligent intervention

---

## 📐 System Architecture Design

### Overall Architecture: ASR-Agents-TTS

```
┌─────────────────────────────────────────────────────────────┐
│                 Android/Embedded Client                      │
│  [Background Service] → [Audio Stream] → [Smart VAD] → [Trigger Decision]  │
└────────────────────────┬────────────────────────────────────┘
                         │ WebSocket/gRPC
┌────────────────────────┴────────────────────────────────────┐
│                  P2P Penetration Server (Personal Server)    │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Intelligent VAD Decision Engine             │  │
│  │  Silence Detection + Decibel Analysis + Voice Detection + │  │
│  │  Semantic Completeness + Voiceprint                  │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     │                                        │
│  ┌──────────────────▼───────────────────────────────────┐  │
│  │                ASR System                             │  │
│  │ SenseVoice (Multilingual+Emotion+VAD) → Real-time    │  │
│  │ Transcription → Dialogue or Self-talk                │  │
│  └──────────────────┬───────────────────────────────────┘  │
│                     │                                        │
│  ┌──────────────────▼───────────────────────────────────┐  │
│  │           Master Agent (Coordinator)                  │  │
│  │         Qwen2.5-7B / gpt-oss:20b                      │  │
│  │                                                        │  │
│  │  ┌─────────────────────────────────────────────────┐ │  │
│  │  │          Agents Ecosystem                        │ │  │
│  │  │                                                  │ │  │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐     │ │  │
│  │  │  │  Memory  │  │ Emotion  │  │ Context  │     │ │  │
│  │  │  │  Agent   │  │  Agent   │  │  Agent   │     │ │  │
│  │  │  └────┬─────┘  └────┬─────┘  └────┬─────┘     │ │  │
│  │  │       │             │             │            │ │  │
│  │  │  ┌────▼─────┐  ┌────▼─────┐  ┌────▼─────┐     │ │  │
│  │  │  │  Dialog  │  │   Tool   │  │  Social  │     │ │  │
│  │  │  │  Agent   │  │  Agent   │  │  Agent   │     │ │  │
│  │  │  └──────────┘  └──────────┘  └──────────┘     │ │  │
│  │  │                                                  │ │  │
│  │  └─────────────────────────────────────────────────┘ │  │
│  │                     │                                 │  │
│  │            ┌────────┴────────┐                       │  │
│  │            │                 │                       │  │
│  │       [Local Decision]  [Cloud API]                  │  │
│  │     Quick response for    Complex reasoning/         │  │
│  │     simple tasks          tool invocation            │  │
│  └─────────────────┬───────────────────────────────────┘  │
│                    │                                        │
│  ┌─────────────────▼───────────────────────────────────┐  │
│  │         Memory & Personality System (Core)           │  │
│  │  Vector Semantic Matching - Metadata Matching -      │  │
│  │  Multimodal Memory Fusion (Experimental)             │  │
│  │  ChromaDB(Vector) + Neo4j(Graph) + PostgreSQL(Structured) │  │
│  └────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │               TTS System                              │  │
│  │  GPT-SoVITS / Index-TTS → Emotionalized Voice Synthesis │  │
│  └──────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---