# Memory Calculator (v4)

A transformer VRAM estimator with a clean, light UI, SVG plot output, vibrant
component colors, DDP/ZeRO sharding, KV-cache modeling, GPU VRAM capacity
check, and a Hugging Face `config.json` loader. Formulas derived from an
MIT-licensed open-source estimator.

## Run

```
open index.html
```

No build step. No runtime dependencies (no web fonts, no CDN).

## What's new in v4

- **KV Cache** — new memory category, computed on inference only.
- **Hugging Face loader** — paste a public model ID, fetches `config.json`
  and `model.safetensors.index.json` to auto-fill architecture and param count.
- **Expanded preset list** — Qwen 2.5 (7B/14B/32B/72B), Gemma 2 (9B/27B),
  Phi-3 Mini/Medium, Mistral 7B, DeepSeek-Coder-V2 Lite, plus Llama 2 and GPT-2 families.
- **Proper SVG plot** — stacked horizontal bars with X-axis ticks, dashed gridlines,
  rotated "GPU Index" Y-axis label, and per-GPU totals on the right.
- **GPU VRAM capacity input** (default 80 GiB) — total turns red when usage exceeds it.
- **DDP / ZeRO-1 / ZeRO-2 / ZeRO-3** parallelism selection (replaces single FSDP toggle).

## Inputs

### Runtime
- Mode: `Training` | `Inference`
- Precision: `mixed` / `fp32` (training) or `fp16/bf16` / `fp32` (inference)
- Optimizer: `Adam` | `SGD` (with optional `momentum`)
- Sequence Length
- Batch Size (per-GPU when GPUs > 1)
- Number of GPUs
- GPU VRAM (GiB) — capacity limit for the red-overflow indicator
- Display Unit: `MiB` | `GiB`
- Parallelism (training, GPUs > 1): `DDP` | `ZeRO-1` | `ZeRO-2` | `ZeRO-3 / FSDP`
- Model Parallelism toggle (inference, GPUs > 1)

### Model
- Preset dropdown (open-source ungated models + reference older families)
- Load from Hugging Face: paste any public model ID
- Parameters (B), Layers, Vocab Size, Hidden Size, Attention Heads,
  Intermediate Size, Key/Value Heads (for GQA)

## Memory categories

| Category | Present in | Color |
|---|---|---|
| CUDA Kernels | always | soft slate |
| Parameters | always | vibrant violet |
| Activations | always | electric cyan |
| **KV Cache** | **inference only** | **sky blue** |
| Gradients | training only | hot pink |
| First Moments | training (Adam or SGD+momentum) | warm amber |
| Second Moments | training (Adam only) | zesty orange |
| Output Tensor | always (×2 on training) | fresh emerald |

## Equations

Let:
- `P` = parameters (billions)
- `L` = layers, `V` = vocab, `h` = hidden size
- `a` = attention heads, `kv` = key/value heads, `i` = intermediate size
- `b` = batch size, `s` = sequence length, `G` = number of GPUs

### Bytes per parameter (`bpp_params`)

| Mode | Precision | bpp |
|---|---|---|
| Training | mixed | 6 (fp32 master + fp16 working) |
| Training | fp32 | 4 |
| Inference | fp16/bf16 | 2 |
| Inference | fp32 | 4 |

### ZeRO sharding (training, G > 1)

| Stage | Params | Gradients | Optimizer states |
|---|---|---|---|
| DDP | ×1 | ×1 | ×1 |
| ZeRO-1 | ×1 | ×1 | ÷ G |
| ZeRO-2 | ×1 | ÷ G | ÷ G |
| ZeRO-3 / FSDP | ÷ G | ÷ G | ÷ G |

For inference with Model Parallelism enabled, `params ÷ G`.

### Component formulas

```
cudaKernels    = 1000 MiB                                  (constant)
parameters     = bpp_params × P × 10⁹        / shard.params
activations    = see below
kvCache        = 2 × L × b × s × kv × (h / a) × bpp_act    (inference only)
outputs        = 4 × b × s × V × (2 if training else 1)
gradients      = 4 × P × 10⁹                 / shard.grads (training only)
firstMoments   = 4 × P × 10⁹                 / shard.optim (Adam or SGD+momentum)
secondMoments  = 4 × P × 10⁹                 / shard.optim (Adam only)
```

Where `bpp_act = 2` if (training+mixed) OR (inference+half), else `4`.

### KV Cache (inference)

Autoregressive attention stores K and V per token, per layer, per KV head:

```
kvCache = 2 × L × b × s × kv × head_dim × bytes_per_elem
```

`head_dim = h / a`. `bytes_per_elem = 2` for fp16/bf16, `4` for fp32.
Grows linearly with `s` and `b`.

### Activations (from [arxiv.org/abs/2205.05198](https://arxiv.org/abs/2205.05198))

Attention block (per layer):
```
head_dim           = h / a
attentionInput     = bpp_act × b × s × h
Q                  = bpp_act × b × s × head_dim × a
K                  = bpp_act × b × s × head_dim × kv
softmaxOutput      = bpp_act × b × a × s²
softmaxDropoutMask =     1   × b × a × s²         (boolean mask)
dropoutOutput      = bpp_act × b × a × s²
V                  = bpp_act × b × s × head_dim × kv
outProjInput       = bpp_act × b × s × a × head_dim
attentionDropout   =     1   × b × s × h          (boolean mask)
attentionBlock = sum of above
```

MLP block (per layer):
```
mlpInput        = bpp_act × b × s × h
activationInput = bpp_act × b × s × i
downProjInput   = bpp_act × b × s × i
dropoutMask     =     1   × b × s × h             (boolean mask)
mlpBlock = sum of above
```

Layer norms (two per layer):
```
layerNorms = bpp_act × b × s × h × 2
```

Total activations:
```
layer       = attentionBlock + mlpBlock + layerNorms
activations = layer × L     (training — all layers retained)
activations = layer         (inference — biggest single layer only; KV cache is separate)
```

### Totals

Per-GPU total; GPU 0 holds the `outputs` tensor:
```
total_GPU0  = cuda + params + act + kvCache + grad + m1 + m2 + outputs
total_other = cuda + params + act + kvCache + grad + m1 + m2
```

Missing categories drop out based on the active mode / optimizer.

### Unit conversion

```
MiB = bytes / 2²⁰    (0 decimal places)
GiB = bytes / 2³⁰    (3 decimal places)
```

## Hugging Face loader

- Input accepts `org/name` or a full `https://huggingface.co/...` URL
- Fetches `https://huggingface.co/{id}/resolve/main/config.json` (CORS enabled
  for public models)
- Parses: `num_hidden_layers`, `hidden_size`, `num_attention_heads`,
  `num_key_value_heads` (falls back to attention heads), `intermediate_size`,
  `vocab_size`, optional `torch_dtype`
- For parameter count: tries `model.safetensors.index.json` → `metadata.total_size`,
  divides by precision bytes (2 for fp16/bf16, 4 for fp32) to get `P × 10⁹`
- If safetensors index is missing, estimates from architecture:
  `P ≈ embed + L · (4·h² + 2·h·head_dim·kv + 3·h·i + 2·h)`
  where `embed = V·h · (1 if tie_word_embeddings else 2)` — reads the
  `tie_word_embeddings` flag from `config.json` to avoid double-counting the
  LM head for models that share it with the input embedding (Gemma, Qwen2,
  Llama-3.2 small, Phi-3, …).
- Gated models (Llama, etc.) return a friendly error — use the preset instead.

## Presets

Open-source (ungated):
- Qwen 2.5 — 7B / 14B / 32B / 72B
- Gemma 2 — 9B / 27B
- Phi-3 Mini (3.8B) / Phi-3 Medium (14B)
- Mistral 7B v0.1
- DeepSeek-Coder-V2 Lite (15.7B MoE)

Reference (gated or older):
- Llama 2 — 7B / 13B / 70B
- Phi-2 / Phi-1.5
- GPT-2 — Small / Medium / Large / XL

Default on page load: **GPT-2 Large**.

## Caveats

- Estimates, not exact measurements. Real allocations vary with framework
  version, kernel implementations, memory fragmentation, and activation
  checkpointing.
- CUDA kernel overhead is pinned at 1000 MiB (real range 300 MiB – 2 GiB).
- Dropout masks are modeled as 1 byte per element (boolean tensors stored as uint8).
- FSDP sharding of **activations** is not modeled (only params / grads / optim).
- MoE models' parameter counts reflect total, not active, experts.
- The HF param-count estimate fallback assumes a SwiGLU-style transformer;
  non-standard architectures (MoE, state-space models) may be off.

## Credits & references

**Primary basis** — formulas and original preset configs from
[furiousteabag/vram-calculator](https://github.com/furiousteabag/vram-calculator)
(MIT License), with the author's write-up at
[asmirnov.xyz/vram](https://asmirnov.xyz/vram).

**Activation accounting** — Korthikanti et al., *"Reducing Activation Recomputation
in Large Transformer Models"*, [arxiv.org/abs/2205.05198](https://arxiv.org/abs/2205.05198).

**Cross-checked against**:
- [mshojaei77/vram-calculator](https://github.com/mshojaei77/vram-calculator) — Streamlit-based
  calculator covering LoRA / QLoRA / quantized inference. Useful reference for the user-facing
  framing of quantization options. Note: their KV-cache formula ignores GQA (uses full hidden
  dim instead of `kv_heads × head_dim`), which overstates KV for Llama-3 70B, Mistral, Qwen 2.5
  and similar.
- [alexziskind1/llm-inference-calculator](https://github.com/alexziskind1/llm-inference-calculator) —
  TypeScript/React inference calculator. Uses heuristic ratios (`KV ≈ 20% × weights × ctx/2048`)
  rather than architectural accounting; good for quick estimates but not a per-tensor model.

**ZeRO sharding reference** — Rajbhandari et al., *"ZeRO: Memory Optimizations Toward Training
Trillion Parameter Models"*, [arxiv.org/abs/1910.02054](https://arxiv.org/abs/1910.02054).

**Hugging Face config schema** —
[huggingface.co/docs/transformers/main_classes/configuration](https://huggingface.co/docs/transformers/main_classes/configuration)
(fields used: `num_hidden_layers`, `hidden_size`, `num_attention_heads`, `num_key_value_heads`,
`intermediate_size`, `vocab_size`, `tie_word_embeddings`, `torch_dtype`).

HTML, CSS, SVG plot rendering, DDP/ZeRO stage selection, KV cache formula, GPU VRAM
capacity check, Hugging Face loader (with tied-embedding detection), and expanded
preset list are additions in this fork.
