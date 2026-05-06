// ============================================================
//  i18n dictionary
//  默认显示中文(HTML 内嵌),切换时把这里的英文塞进 data-i18n 节点
// ============================================================
window.I18N = {
  en: {
    "nav.brand": "Xiaoxin Shi",
    "nav.projects": "Projects",
    "nav.contact": "Contact",

    "hero.name": "Xiaoxin Shi",
    "hero.name_en": "石枭昕",
    "hero.tagline": "LLM Post-Training · Inference Acceleration · On-Device Real-Time Agents",
    "hero.affiliation": "Ph.D. Student · Shanghai Institute of Intelligent Science / SJTU · Advisor: Prof. Zengfeng Huang",
    "hero.download_pdf": "Download CV (PDF)",
    "hero.phone": "+86 173-1671-6363",
    "hero.edu": "B.S. in Chemistry (Computational) 2021–2025<br>Ph.D. in LLM, Sept. 2025 – Present",

    "highlights.title": "Core Highlights",
    "highlights.h1.tag": "Full-Stack Game / Virtual-Human NPC Pipeline",
    "highlights.h1.body": "Independently built a complete end-to-end pipeline (Voice Input → ASR → LLM Intent Understanding → NPC Behavioral Decision → Character Animation &amp; Rendering), including <i>Echo Chronicles</i> (pure on-device iOS tower defense) and <i>SimpleLove</i> (VRM virtual human), with native deployment across iOS / Windows / macOS.",
    "highlights.h2.tag": "ICML 2026 (Accepted, First Author)",
    "highlights.h2.body": "Proposed a multi-head parallel decoding architecture for real-time function calling, achieving end-to-end <b>3–6×</b> speedup (peak 9.6×); Qwen3-4B at <b>61.2 ms / 16 Hz</b> on RTX 4090; <b>528 ms P50</b> on real iPhone 17 Pro Max. Significantly outperforms Google FunctionGemma in both accuracy and speed.",
    "highlights.h3.tag": "Large-Scale Training Practice",
    "highlights.h3.body": "<b>Multi-node distributed training experience (up to 48×H200)</b>; proficient in Packing, Length Bucketing, Gradient Accumulation, FlashAttention-2, FlexAttention, DDP, etc.",
    "highlights.h4.tag": "On-Device Inference Infrastructure",
    "highlights.h4.body": "Customized llama.cpp for cross-platform KV Cache sharing &amp; multi-seq batching tailored for the hybrid SimpleTool model; modified nano-vllm for multi-head parallel decoding; built efficient ONNX-based DiT inference; leveraged AI coding assistants (Claude / Cursor) for solo full-stack module integration.",

    "projects.title": "Project Experience",
    "projects.label.background": "Background",
    "projects.label.method": "Methodology",
    "projects.label.contrib": "Contributions",
    "projects.label.result": "Results",

    "projects.p1.name": "SimpleTool: Parallel Decoding for Real-Time LLM Function Calling (ICML 2026)",
    "projects.p1.date": "Oct. 2025 – Present",
    "projects.p1.background": "LLM function calling exhibits excessive latency in on-device scenarios, failing to meet the 10 Hz+ control requirements of game NPCs, real-time virtual humans, and robotic arms.",
    "projects.p1.method": "Designed 17 special tokens to concurrently serve as \"structural token compressors\" and \"mode selectors,\" compressing the function-call output space by <b>4–6×</b>; proposed a multi-head parallel decoding architecture that exploits idle compute during the decoding phase to generate function names and arguments simultaneously across different heads.",
    "projects.p1.contrib": "Independently led the entire lifecycle: Idea → Data Synthesis Pipeline → Training Framework → Inference Engine Modification → Multi-Platform Deployment → Paper Writing, Submission, and Rebuttal.",
    "projects.p1.result": "Qwen3-4B reaches <b>61.2 ms P50</b> (16 Hz) on RTX 4090 with 93% avg efficiency across 8 parallel heads; RT-Qwen-0.5B achieves <b>86.2%</b> on Mobile Actions Unseen Benchmark (vs. 85.0% for FunctionGemma-270M) while preserving general capabilities (MMLU −0.29%, IFEval +2.78%). <b>Accepted to ICML 2026</b>; 7 model versions (0.5B–30B MoE) released on HuggingFace and ModelScope.",

    "projects.p2.name": "SimpleLove / NPC.exe — On-Device Real-Time AI-Native Virtual Human &amp; Game NPC Engine",
    "projects.p2.date": "Feb. 2026 – Present",
    "projects.p2.background": "Existing virtual-human / game-NPC systems predominantly rely on cloud LLMs, suffering from high latency, poor privacy, and a disjoint between behavior policy and character identity — feeling like traditional cloud voice assistants.",
    "projects.p2.method": "Built an end-to-end local virtual-human engine on top of SimpleTool; used NPC Policy SFT / RL (Action as Cosplay) to make the LLM directly emit character actions; Simple-T2M reuses SimpleTool LLM hidden states as text condition, eliminating the standard 8B-parameter independent text encoder.",
    "projects.p2.contrib": "Independently designed the architecture, trained NPC Policy and a 50M DiT Flow Matching motion-generation model, completed cross-platform native deployment with VRM virtual-human rendering integration.",
    "projects.p2.result": "NPC Policy eval accuracy <b>16.2% → 58.7%</b> (+42.5pp), with RL-friendly output distribution (Top-10 Cov 0.993, directly attachable to PPO/GRPO); Simple-T2M generates <b>motion in ~50 ms</b> on RTX 4090 (Q8); the full system uses only <b>5 GB VRAM</b>; fully running on Linux / Windows (DirectML) / native end-to-end with zero Python dependency (macOS Metal + CoreML in progress) — ready to embed into game clients.",

    "projects.p3.name": "Echo Chronicles — Voice-Driven Tower Defense (SimpleTool Demo)",
    "projects.p3.date": "Dec. 2025 – Feb. 2026",
    "projects.p3.label1": "End-to-End Loop",
    "projects.p3.body1": "Player voice → on-device ASR (Sherpa-onnx Paraformer, 5.9% WER, 115 ms) → SimpleTool 0.5B intent understanding &amp; function calling → in-game tower-defense unit response.",
    "projects.p3.contrib": "Independently developed iOS llama.swiftui + Metal inference integration, PixiJS WebGL rendering layer, five-element tower-defense system, and campaign level design.",
    "projects.p3.result": "Runs entirely locally on iPhone 17 Pro Max with zero cloud dependency, validating the feasibility and latency controllability of the \"LLM as in-game NPC / character controller\" paradigm on edge devices.",

    "skills.title": "Technical Skills",
    "skills.s1.k": "Programming",
    "skills.s1.v": "Proficient in Python; adept at C++ / CUDA / Swift application development with AI coding assistants.",
    "skills.s2.k": "Algorithms &amp; Training",
    "skills.s2.v": "Transformer, SFT / RL, LoRA, Packing / Length Bucketing, Curriculum Learning, multi-node DDP (48-GPU experience), FlashAttention-2, FlexAttention.",
    "skills.s3.k": "Inference &amp; Architecture",
    "skills.s3.v": "PyTorch, vLLM, llama.cpp (with custom modifications), nano-vllm, ONNX Runtime (CUDA / DirectML), GGUF quantization (Q4/Q8), KV Cache optimization.",
    "skills.s4.k": "On-Device Deployment",
    "skills.s4.v": "Linux, Windows (DirectML), macOS (Metal), iOS LLM native development (via ggml &amp; onnx), on-device ASR deployment.",
    "skills.s5.k": "Data Synthesis",
    "skills.s5.v": "Multi-agent collaboration pipelines, LLM-as-Judge; produced 2M+ industrial-grade training samples (game NPCs, virtual humans, robotic arms).",

    "summary.title": "Personal Summary",
    "summary.body": "My research taste favors industry-oriented R&amp;D — I'm used to reverse-engineering infrastructure and algorithm design from application requirements, firmly believing \"a good idea is one that can be deployed.\" My technical interests focus on LLM post-training, inference acceleration, and on-device real-time agents, with a long-term vision of actualizing embodied intelligence — from virtual digital lives to physical robots. Equipped with full-stack solo-development capabilities for translating ideas into cross-platform native demos.",

    "contact.title": "Contact",
    "contact.email": "Email",
    "contact.phone": "Phone",

    "footer.copyright": "© 2026 Xiaoxin Shi · Built with HTML &amp; CSS",
    "footer.source": "Source"
  }
};
