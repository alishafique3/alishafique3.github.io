<div class="chapter-meta">Module 2 · Chapter 4</div>

# Understanding GPU Memory for LLM Inference

<p class="lede">When you run <code>nvidia-smi</code> and see 78 GB / 80 GB used, that number has structure. Five things live in HBM during inference, each with a different growth law. Being able to tell them apart is the difference between fixing an OOM in five minutes and shrugging at it.</p>

## The five buckets

| Bucket | Grows with | Typical fraction |
|---|---|---|
| Model weights | $P$, precision | 50–80 % |
| KV cache | $B \cdot s \cdot L \cdot n_{kv} \cdot d_h$ | 10–40 % |
| Activations (one layer) | $B \cdot s \cdot d$ | 2–8 % |
| CUDA workspace | fixed per kernel + per request | 2–5 % |
| Allocator fragmentation | unpredictable | 2–10 % |

## 1. Model weights

$P$ parameters at $b$ bytes per parameter:

$$ \text{weights} = P \cdot b $$

Llama 3 70B at bf16: 140 GB. At fp8: 70 GB. At int4: 35 GB. This bucket is fixed for the lifetime of the serving process — it is loaded once at startup.

## 2. KV cache

Per token per sequence, the cache stores keys and values for every layer and every KV head:

$$ \text{KV}_{\text{per token}} = 2 \cdot L \cdot n_{kv} \cdot d_h \cdot b_{\text{elem}} $$

Total KV scales linearly with concurrent sequences and their lengths. This bucket *grows and shrinks with traffic* — it is the memory you must budget most carefully.

## 3. Activations

During inference, only the *current layer's* activations need to be in memory — the previous layer's activations are consumed and discarded. For transformer layer $\ell$, the live activation tensors are approximately:

$$ \text{activations} \approx B \cdot s \cdot (3d + i) \cdot b_{\text{elem}} $$

where $i$ is the intermediate size (typically $\sim 4d$). For Llama 3 70B at batch 8, sequence 2k, bf16: ~5 GB. Small relative to weights, but non-zero.

## 4. CUDA workspace

Per-kernel scratch buffers (cuBLAS workspace, FlashAttention IO buffers), the allocator's bookkeeping, and the per-process CUDA context. Collectively ~1–3 GB. Fixed, non-growing.

## 5. Allocator fragmentation

Classic bug: enough free bytes total, but not enough contiguous bytes for the next allocation. PagedAttention (M6.3) nearly eliminated this for KV, but weights and activations still allocate the old way. Fragmentation is the "where did my 5 GB go" at the end of a long-running process.

## A worked decomposition — Llama 3 70B on 2×H100

Config: TP = 2, bf16 weights, batch 16, mean context 2k.

- Weights, sharded across 2 GPUs: 140 GB / 2 = 70 GB per GPU.
- KV at $L = 80$, $n_{kv} = 8$, $d_h = 128$, bf16: 0.31 MB × 2048 × 16 / 2 (sharded) ≈ 5.1 GB per GPU.
- Activations at $B = 16$, $s = 2048$, $d = 8192$, $i = 28672$: ~1.7 GB per GPU.
- CUDA workspace: ~2 GB per GPU.
- Fragmentation headroom: ~2 GB.
- **Total: ~81 GB per GPU** — overflows 80 GB. Either reduce batch, quantize KV, or move to H200 (141 GB).

## Decomposing an OOM

1. **Print it.** vLLM, SGLang, and TensorRT-LLM all log their memory budget on startup. Read it.
2. **Attribute by formula.** Weights are deterministic. KV scales with your max concurrent sequences × max context. Anything else is activations or workspace.
3. **Reduce the right thing.** OOM from weights → quantize or shard more. OOM from KV → smaller batch, shorter max context, or KV quantization. OOM from fragmentation → set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Pitfalls

- **Over-allocating KV upfront.** Pre-vLLM engines reserved worst-case KV per request; PagedAttention fixed this. If you're not on a paged engine, you are leaving 2–4× KV capacity on the table.
- **Assuming activations scale per-layer.** During inference they don't — only one layer is live. During training they do (all layers retained for backward).
- **Forgetting the CUDA context.** Every new Python process loses ~1 GB to CUDA init. If you fork workers, factor it in.

<section class="faq">
<h2>FAQ</h2>
<h3>Why does vLLM allocate 90% of GPU memory at startup?</h3>
<p>It pre-allocates the entire KV cache pool up front to avoid runtime fragmentation. The number is configurable via <code>--gpu-memory-utilization</code>.</p>
<h3>How do I estimate KV for a GQA model?</h3>
<p>Use <code>num_key_value_heads</code>, not <code>num_attention_heads</code>. Then: $2 \cdot L \cdot n_{kv} \cdot d_h \cdot B \cdot s \cdot b_{\text{elem}}$. GQA with 8 KV heads uses 4–8× less KV than full MHA.</p>
<h3>What is "activation memory" during inference?</h3>
<p>Transient tensors for the <em>current</em> layer: Q, K, V, attention scores, FFN intermediate. Previous layers' activations are discarded. Much smaller than training's activation bill.</p>
<h3>How do I unfragment memory?</h3>
<p>Set <code>PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True</code>, or use a paged-KV engine (vLLM, SGLang) which handles KV fragmentation internally.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://dl.acm.org/doi/10.1145/3600006.3613165" target="_blank" rel="noreferrer">Kwon et al. — Efficient Memory Management for LLM Serving with PagedAttention (SOSP 2023)</a></li>
<li><a href="https://alishafique3.github.io/memory-calculator/" target="_blank" rel="noreferrer">Memory Calculator — interactive VRAM estimation tool</a></li>
<li><a href="https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/" target="_blank" rel="noreferrer">NVIDIA — Mastering LLM Inference Optimization (KV cache formulas)</a></li>
</ul>
</section>

## What's next

Multi-GPU cases introduce [interconnect bandwidth](?module=m2&chapter=gpu-interconnects) as another constraint. For the capacity-planning view of this memory decomposition, head to the [Memory Calculator](https://alishafique3.github.io/memory-calculator/).
