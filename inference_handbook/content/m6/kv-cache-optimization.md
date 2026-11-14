<div class="chapter-meta">Module 6 · Chapter 4</div>

# KV Cache Optimization — Quantization, Compression, Offloading

<p class="lede">PagedAttention solves fragmentation. The next layer of KV optimization attacks the raw size of the cache itself. Three techniques: <strong>quantize</strong> KV to fewer bits, <strong>share</strong> it via prefix caching, and <strong>offload</strong> cold parts to CPU or remote storage. Together they routinely 3–10 × effective KV capacity.</p>

## Why optimize the KV cache further

Even with PagedAttention, KV cache for a 70 B model at 128 k context is ~40 GB per sequence. At 16 concurrent sessions, that's 640 GB — bigger than the model weights. Three honest paths:

1. **Make each byte do more** — shrink via quantization.
2. **Share bytes across requests** — prefix caching.
3. **Move cold bytes off-GPU** — offloading.

## KV cache quantization

The cache stores $K$ and $V$ tensors. Quantize them to INT8 or FP8:

$$ K_q = \text{quant}_{b}(K), \quad V_q = \text{quant}_{b}(V) $$

At attention time, dequant on the fly (or run int-attn kernels directly on Hopper+). Trade-offs:

| Precision | Memory | Quality cost |
|---|---|---|
| BF16 (baseline) | 2 bytes/val | — |
| FP8 | 1 byte/val | &lt; 0.5 % on most tasks |
| INT8 | 1 byte/val | Comparable to FP8 |
| INT4 | 0.5 byte/val | Noticeable on long-context retrieval |

FP8 KV cache is now the production default on Hopper and Blackwell. Enable in vLLM with `--kv-cache-dtype=fp8`.

## Prefix caching — the biggest free win

Prefix caching stores the KV of common prefixes and reuses it across requests. If 1000 users share a 500-token system prompt, the system prompt gets prefilled once and reused 999 times. Savings on prefill TTFT can be 50–90 % for chat and RAG workloads.

### Hash-based prefix caching (vLLM)

Blocks are content-addressable — their hash is a function of the token sequence. Two requests with the same prefix naturally share the same physical blocks:

```
Request A:  [sys_blk][sys_blk][ex_blk][user_a_blk]
Request B:  [sys_blk][sys_blk][ex_blk][user_b_blk]
                ↑ shared ↑  ↑shared↑
```

Enabled by default in vLLM. Zero config.

### Radix tree caching (SGLang)

RadixAttention stores all cached prefixes in a radix tree and matches more aggressively — including partial matches within blocks. For workloads with heavy sharing (chatbots with long system prompts, agents, RAG), SGLang wins over vLLM by 20–30%.

### Prompt structure matters

Prefix caching only helps if prompts actually share prefixes. The **right structure** (M5.4): system prompt → few-shot → retrieved context → user input. The wrong structure interleaves per-request data early and defeats the cache.

## KV cache offloading — spilling to CPU

When KV won't fit in HBM, offload less-recently-used blocks to CPU DRAM. On access, swap back. Libraries:

- **LMCache** — vLLM-integrated KV offload with disk+CPU tiering.
- **Mooncake** — distributed KV store across a cluster, used by Kimi.

Trade-off: offloaded fetches add ~10–50 ms depending on bus speed. Worth it when the choice is "offload" vs. "run out of memory and preempt". Not worth it on latency-sensitive paths.

## A practical layered strategy

For a production 70 B inference service:

1. **Always on**: PagedAttention (M6.3).
2. **Always on**: FP8 KV cache (`--kv-cache-dtype=fp8`).
3. **Default on**: prefix caching (`--enable-prefix-caching`).
4. **When KV grows large**: offload to CPU (LMCache) OR truncate conversation history.
5. **When context is huge**: INT4 KV — but measure retrieval quality.

Combined, these routinely 4–8 × effective KV capacity vs. naive bf16 contiguous allocation — the difference between fitting 8 and 64 concurrent long-context sessions on the same hardware.

## Pitfalls

- **Enabling INT4 KV without measuring long-context quality.** The regression shows up on needle-in-haystack, not on chat.
- **Thinking prefix caching is free when it isn't effective.** Measure cache hit rate. If &lt; 20 %, your prompts aren't structured for it.
- **Mixing FP8 weights with BF16 KV (or vice versa).** Fine functionally, but you leave bandwidth on the table. Match precision across both when possible.
- **Treating CPU offload as a scheduler optimization.** It's a memory optimization. Latency gets worse, not better.

<section class="faq">
<h2>FAQ</h2>
<h3>Can I quantize only one of K or V?</h3>
<p>Yes, but rare. In practice K and V are quantized together. Some recent research explores separate schemes.</p>
<h3>Does prefix caching work with LoRA?</h3>
<p>Yes — the KV cache is agnostic to the LoRA adapter. Caching hits across requests using the same adapter.</p>
<h3>How much does FP8 KV hurt long-context retrieval?</h3>
<p>In published benchmarks, &lt; 1 % on 128k needle-in-haystack. INT4 KV is the one to watch — can hit 5–10 % on long-context tasks.</p>
<h3>Is CPU offload worth it for streaming chat?</h3>
<p>Rarely. The latency tax on KV fetch is visible in TPOT. Save it for offline or batch workloads.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://docs.vllm.ai/en/stable/features/automatic_prefix_caching.html" target="_blank" rel="noreferrer">vLLM — Automatic prefix caching documentation</a></li>
<li><a href="https://arxiv.org/abs/2312.07104" target="_blank" rel="noreferrer">Zheng et al. — SGLang RadixAttention</a></li>
<li><a href="https://github.com/LMCache/LMCache" target="_blank" rel="noreferrer">LMCache — KV cache offloading</a></li>
<li><a href="https://arxiv.org/abs/2407.00079" target="_blank" rel="noreferrer">Qin et al. — Mooncake: A KVCache-centric Disaggregated Architecture</a></li>
</ul>
</section>

## What's next

Decoding optimization ([Chapter 5](?module=m6&chapter=decoding-optimization)) tackles the step side. Chapter 6 covers distributed parallelism — how sharding interacts with KV management.
