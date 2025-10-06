# Qwen3-8B-Base 性能测试笔记

## 1. 测试环境

- **模型**: Qwen3-8B-Base
- **设备**: NVIDIA GeForce RTX 4090 (48G显存版)
- **词表大小**: 151,643

---

## 2. 实验总结

### 实验 1: 短 Prompt

- **Prompt 长度**: 7 tokens
- **生成 Token 数**: 287 tokens

| 阶段    | 耗时 (ms) | 占比    | 吞吐量 (tokens/s)      |
| :------ | :-------- | :------ | :--------------------- |
| Prefill | 70.87     | 0.47%   | 98.77                  |
| Decode  | 14966.21  | 99.53%  | 19.11 (52.33 ms/token) |
| **总计** | **15037.09** | **100%** | **19.09 (整体)** |

<details>
<summary>点击查看 Prefill 阶段模块分析</summary>

| 模块                | 总时间(ms) | 占比    |
| :------------------ | :--------- | :------ |
| Attention.K_proj    | 12.27      | 24.61 % |
| FFN.down_proj       | 5.23       | 10.48 % |
| FFN.gate_proj       | 5.07       | 10.17 % |
| FFN.up_proj         | 4.84       | 9.71 %  |
| LayerNorm.post_attn | 3.34       | 6.71 %  |
| ...                 | ...        | ...     |

</details>

<details>
<summary>点击查看 Decode 阶段模块分析 (平均每步)</summary>

| 模块                | 平均(ms) | 占比    |
| :------------------ | :------- | :------ |
| FFN.gate_proj       | 0.1309   | 13.27 % |
| FFN.down_proj       | 0.1302   | 13.19 % |
| FFN.up_proj         | 0.1281   | 12.99 % |
| LayerNorm.post_attn | 0.0876   | 8.88 %  |
| LayerNorm.pre_attn  | 0.0852   | 8.64 %  |
| ...                 | ...      | ...     |

</details>

---

### 实验 2: 长 Prompt

- **Prompt 长度**: 293 tokens
- **生成 Token 数**: 55 tokens

| 阶段    | 耗时 (ms) | 占比   | 吞吐量 (tokens/s)      |
| :------ | :-------- | :----- | :--------------------- |
| Prefill | 80.42     | 2.76%  | 3643.30                |
| Decode  | 2838.66   | 97.24% | 19.02 (52.57 ms/token) |
| **总计** | **2919.08** | **100%** | **18.84 (整体)** |

<details>
<summary>点击查看 Prefill 阶段模块分析</summary>

| 模块            | 总时间(ms) | 占比    |
| :-------------- | :--------- | :------ |
| FFN.gate_proj   | 9.46       | 16.20 % |
| FFN.up_proj     | 9.07       | 15.53 % |
| FFN.down_proj   | 9.04       | 15.48 % |
| Attention.Q_proj| 4.34       | 7.43 %  |
| Attention.O_proj| 4.12       | 7.05 %  |
| ...             | ...        | ...     |

</details>

<details>
<summary>点击查看 Decode 阶段模块分析 (平均每步)</summary>

| 模块                | 平均(ms) | 占比    |
| :------------------ | :------- | :------ |
| FFN.gate_proj       | 0.1306   | 13.20 % |
| FFN.down_proj       | 0.1302   | 13.16 % |
| FFN.up_proj         | 0.1281   | 12.95 % |
| LayerNorm.post_attn | 0.0886   | 8.95 %  |
| LayerNorm.pre_attn  | 0.0860   | 8.69 %  |
| ...                 | ...      | ...     |

</details>

---

## 3. 对比分析与结论

### 数据对比

| 指标                       | 实验1 (短prompt) | 实验2 (长prompt) | 比值      |
| :------------------------- | :--------------- | :--------------- | :-------- |
| **Prompt 长度** | 7                | 293              | 41.9 x    |
| **Prefill 时间 (ms)** | 70.87            | 80.42            | 1.1 x     |
| **Decode 平均时间 (ms/token)** | 52.33            | 52.57            | 1.00 x    |
| **生成 token 数** | 287              | 55               | -         |

### 关键结论

1.  **Prefill 阶段是计算密集型 (Compute-Bound)**
    - 当 Prompt 长度从 7 增加到 293 (约42倍)时，Prefill 时间仅增加了 1.1倍。这表明对于短文本，Prefill 的固定开销较大，但随着长度增加，处理时间近似线性增长。长 Prompt 下的高吞吐量 (3643 tokens/s) 也证明了并行计算的效率。

2.  **Decode 阶段是访存密集型 (Memory-Bound)**
    - 无论 Prompt 长短，解码（生成）单个 token 的平均时间几乎恒定 (52.33 ms vs 52.57 ms)。这说明 Decode 阶段的瓶颈在于从显存中读写 KV Cache，而不是在矩阵计算上。

3.  **推理的主要瓶颈是 Decode**
    - 在两个实验中，Decode 阶段的耗时都占了总时间的绝对主导地位（短 Prompt 占 99.5%，长 Prompt 占 97.2%）。这是典型的自回归模型生成长序列时的性能特征。

---

## 4. 附录

<details>
<summary>点击查看测试代码</summary>

```python
#!/usr/bin/env python3
import os
import torch
import time
from collections import defaultdict

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from transformers import AutoModelForCausalLM, AutoTokenizer

print("="*80)
print("加载模型和分词器...")
print("="*80)

model_path = "./Qwen/Qwen3-8B-Base/"
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="cuda:0"
)

print(f"模型加载完成, 设备: {model.device}")
print(f"词表大小: {tokenizer.vocab_size}")

# ============================================
# 修正版：更精确的模块性能分析
# ============================================

class ModuleProfiler:
    """改进版：避免父子模块重复计时"""
    def __init__(self, model):
        self.model = model
        self.module_times = defaultdict(list)
        self.hooks = []
        self.active_modules = []  # 跟踪活动模块栈
        
    def register_hooks(self):
        """只为叶子模块注册 hooks"""
        
        # 识别所有叶子模块（没有子模块的模块）
        leaf_modules = {}
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # 叶子模块
                leaf_modules[id(module)] = name
        
        def pre_forward_hook(module, input):
            if id(module) in leaf_modules:
                torch.cuda.synchronize()
                self.active_modules.append({
                    'name': leaf_modules[id(module)],
                    'start': time.time()
                })
        
        def forward_hook(module, input, output):
            if id(module) in leaf_modules and self.active_modules:
                torch.cuda.synchronize()
                module_info = self.active_modules.pop()
                elapsed = time.time() - module_info['start']
                
                # 简化模块名称
                name = self._simplify_name(module_info['name'])
                self.module_times[name].append(elapsed)
        
        # 注册 hooks
        for name, module in self.model.named_modules():
            if id(module) in leaf_modules:
                h1 = module.register_forward_pre_hook(pre_forward_hook)
                h2 = module.register_forward_hook(forward_hook)
                self.hooks.append(h1)
                self.hooks.append(h2)
    
    def _simplify_name(self, full_name):
        """简化模块名称"""
        if 'embed_tokens' in full_name:
            return 'Embedding'
        elif 'lm_head' in full_name:
            return 'LM_Head'
        elif 'model.norm' in full_name:
            return 'Final_LayerNorm'
        
        # Attention 相关
        elif 'self_attn.q_proj' in full_name:
            return 'Attention.Q_proj'
        elif 'self_attn.k_proj' in full_name:
            return 'Attention.K_proj'
        elif 'self_attn.v_proj' in full_name:
            return 'Attention.V_proj'
        elif 'self_attn.o_proj' in full_name:
            return 'Attention.O_proj'
        elif 'rotary_emb' in full_name:
            return 'Attention.RoPE'
        
        # FFN 相关
        elif 'mlp.gate_proj' in full_name:
            return 'FFN.gate_proj'
        elif 'mlp.up_proj' in full_name:
            return 'FFN.up_proj'
        elif 'mlp.down_proj' in full_name:
            return 'FFN.down_proj'
        elif 'mlp.act_fn' in full_name:
            return 'FFN.activation'
        
        # LayerNorm
        elif 'input_layernorm' in full_name:
            return 'LayerNorm.pre_attn'
        elif 'post_attention_layernorm' in full_name:
            return 'LayerNorm.post_attn'
        
        return full_name
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def reset(self):
        self.module_times = defaultdict(list)
        self.active_modules = []
    
    def report(self, title="模块性能统计", top_n=15):
        print(f"\n{'='*90}")
        print(f"{title}")
        print(f"{'='*90}")
        
        # 聚合统计
        aggregated = {}
        for name, times in self.module_times.items():
            total = sum(times)
            count = len(times)
            aggregated[name] = {
                'total': total,
                'count': count,
                'avg': total / count if count > 0 else 0
            }
        
        total_time = sum(s['total'] for s in aggregated.values())
        
        # 按总时间排序
        sorted_items = sorted(aggregated.items(), key=lambda x: x[1]['total'], reverse=True)
        
        print(f"{'模块':<35} {'总时间(ms)':<12} {'调用次数':<10} {'平均(ms)':<12} {'占比':<8}")
        print("-"*90)
        
        for name, stats in sorted_items[:top_n]:
            pct = (stats['total'] / total_time * 100) if total_time > 0 else 0
            print(f"{name:<35} {stats['total']*1000:<12.2f} {stats['count']:<10} "
                  f"{stats['avg']*1000:<12.4f} {pct:<8.2f}%")
        
        if len(sorted_items) > top_n:
            other_time = sum(s['total'] for _, s in sorted_items[top_n:])
            other_pct = (other_time / total_time * 100) if total_time > 0 else 0
            print(f"{'(其他模块)':<35} {other_time*1000:<12.2f} {'---':<10} "
                  f"{'---':<12} {other_pct:<8.2f}%")
        
        print("-"*90)
        print(f"{'总计':<35} {total_time*1000:<12.2f}")
        print()

# ============================================
# Warmup：消除首次运行的 CUDA 开销
# ============================================

print("\n" + "="*80)
print("Warmup: 预热 GPU（消除 CUDA 编译开销）")
print("="*80)

warmup_prompt = "测试"
warmup_inputs = tokenizer(warmup_prompt, return_tensors="pt").to(model.device)

print("进行 3 次 warmup...")
for i in range(3):
    with torch.no_grad():
        _ = model(**warmup_inputs, use_cache=True)
    print(f"  Warmup {i+1}/3 完成")

torch.cuda.synchronize()
print("Warmup 完成！\n")

# ============================================
# 性能分析函数
# ============================================

def profile_generation(model, tokenizer, prompt, max_new_tokens, profiler, experiment_name):
    """带性能分析的生成"""
    print(f"\n{'='*80}")
    print(f"{experiment_name}")
    print(f"{'='*80}\n")
    
    # Tokenize
    torch.cuda.synchronize()
    t0 = time.time()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    torch.cuda.synchronize()
    tokenize_time = time.time() - t0
    
    input_ids = inputs['input_ids']
    prompt_len = input_ids.shape[1]
    
    print(f"Prompt 长度: {prompt_len} tokens")
    print(f"Max new tokens: {max_new_tokens}\n")
    
    # ============================================
    # Prefill
    # ============================================
    print("--- Prefill 阶段 ---")
    profiler.reset()
    
    torch.cuda.synchronize()
    t_prefill = time.time()
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
    
    torch.cuda.synchronize()
    prefill_time = time.time() - t_prefill
    
    print(f"Prefill 时间: {prefill_time*1000:.2f} ms")
    print(f"吞吐量: {prompt_len/prefill_time:.2f} tokens/s")
    
    profiler.report(f"{experiment_name} - Prefill 模块分析")
    
    # ============================================
    # Decode
    # ============================================
    print("--- Decode 阶段 ---")
    profiler.reset()
    
    past_key_values = outputs.past_key_values
    logits = outputs.logits[:, -1, :]
    next_token_id = torch.argmax(logits, dim=-1).item()
    
    generated_tokens = [next_token_id]
    current_token = torch.tensor([[next_token_id]], device=model.device)
    
    decode_times = []
    
    for step in range(1, max_new_tokens):
        torch.cuda.synchronize()
        t_step = time.time()
        
        with torch.no_grad():
            outputs = model(
                input_ids=current_token,
                past_key_values=past_key_values,
                use_cache=True
            )
        
        torch.cuda.synchronize()
        decode_times.append(time.time() - t_step)
        
        logits = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values
        next_token_id = torch.argmax(logits, dim=-1).item()
        
        generated_tokens.append(next_token_id)
        current_token = torch.tensor([[next_token_id]], device=model.device)
        
        if next_token_id == tokenizer.eos_token_id:
            print(f"遇到 EOS，停止于第 {step+1} 步")
            break
    
    avg_decode = sum(decode_times) / len(decode_times) * 1000
    total_decode = sum(decode_times) * 1000
    
    print(f"生成 token 数: {len(generated_tokens)}")
    print(f"平均 decode 时间: {avg_decode:.2f} ms/token")
    print(f"Decode 吞吐量: {1000/avg_decode:.2f} tokens/s")
    
    profiler.report(f"{experiment_name} - Decode 模块分析（平均每步）")
    
    # 解码文本
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # 总结
    total_time = prefill_time + sum(decode_times)
    print(f"\n{'='*80}")
    print(f"{experiment_name} - 总结")
    print(f"{'='*80}")
    print(f"Tokenization:  {tokenize_time*1000:>10.2f} ms  ({tokenize_time/total_time*100:>5.2f}%)")
    print(f"Prefill:       {prefill_time*1000:>10.2f} ms  ({prefill_time/total_time*100:>5.2f}%)")
    print(f"Decode:        {total_decode:>10.2f} ms  ({sum(decode_times)/total_time*100:>5.2f}%)")
    print(f"{'总计:':<15} {total_time*1000:>10.2f} ms")
    print(f"\n整体吞吐量: {len(generated_tokens)/total_time:.2f} tokens/s")
    print(f"生成文本（前150字符）: {generated_text[:150]}")
    print()
    
    return generated_text, {
        'prefill_time': prefill_time,
        'decode_time': sum(decode_times),
        'avg_decode': avg_decode,
        'prompt_len': prompt_len,
        'generated_len': len(generated_tokens)
    }

# ============================================
# 实验
# ============================================

profiler = ModuleProfiler(model)
profiler.register_hooks()

# 实验 1
prompt1 = "莎士比亚是哪国人?"
text1, stats1 = profile_generation(
    model, tokenizer, prompt1, 500, profiler, "实验 1: 短 Prompt"
)

# 实验 2
prompt2 = prompt1 + text1
text2, stats2 = profile_generation(
    model, tokenizer, prompt2, 500, profiler, "实验 2: 长 Prompt"
)

profiler.remove_hooks()

# ============================================
# 对比分析
# ============================================

print("\n" + "="*80)
print("实验对比分析")
print("="*80)

print(f"\n{'指标':<30} {'实验1 (短prompt)':<20} {'实验2 (长prompt)':<20} {'比值':<10}")
print("-"*80)
print(f"{'Prompt 长度':<30} {stats1['prompt_len']:<20} {stats2['prompt_len']:<20} "
      f"{stats2['prompt_len']/stats1['prompt_len']:<10.1f}x")
print(f"{'Prefill 时间 (ms)':<30} {stats1['prefill_time']*1000:<20.2f} "
      f"{stats2['prefill_time']*1000:<20.2f} "
      f"{stats2['prefill_time']/stats1['prefill_time']:<10.1f}x")
print(f"{'Decode 平均时间 (ms/token)':<30} {stats1['avg_decode']:<20.2f} "
      f"{stats2['avg_decode']:<20.2f} "
      f"{stats2['avg_decode']/stats1['avg_decode']:<10.2f}x")
print(f"{'生成 token 数':<30} {stats1['generated_len']:<20} {stats2['generated_len']:<20}")

print("\n" + "="*80)
print("关键结论")
print("="*80)
print("1. Prefill 是 compute-bound:")
print(f"    - 时间随 prompt 长度线性增长（{stats2['prefill_time']/stats1['prefill_time']:.1f}x）")
print("\n2. Decode 是 memory-bound:")
print(f"    - 平均时间几乎不变（{stats2['avg_decode']/stats1['avg_decode']:.2f}x，接近1.0）")
print(f"    - 瓶颈在内存带宽，不在计算")
print("\n3. 主要耗时：")
print(f"    - 实验1: Decode占 {stats1['decode_time']/(stats1['prefill_time']+stats1['decode_time'])*100:.1f}%")
print(f"    - 实验2: Decode占 {stats2['decode_time']/(stats2['prefill_time']+stats2['decode_time'])*100:.1f}%")
print("="*80)
