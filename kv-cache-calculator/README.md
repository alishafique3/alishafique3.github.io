# KV Cache Calculator — v2

Per-device KV cache memory estimator rewritten in the clean `memory_calculator_v5`
aesthetic: white background, centered headline, two-column layout with
indigo accents, GitHub pill top-right.

Reuses upstream calculator logic + model catalog verbatim. Only the UI is new.

## Reused from upstream (`kvcache-ai/kvcache-blog`, MIT)

Both files are adapted from
[`kvcache-ai/kvcache-blog`](https://github.com/kvcache-ai/kvcache-blog/tree/d455485bd1c5265951849f011a60564da1b23cfe)
at commit `d455485bd1c5265951849f011a60564da1b23cfe`:

- `assets/calculator-core.js` — 7 capacity formulas: Standard MHA/GQA, MLA,
  DSA/MLA + Indexer, Qwen Linear/Full Hybrid, Mixed Full/Sliding GQA,
  MiniMax MSA, DeepSeek V4 Hybrid.
- `assets/models.js` — 55 model configs across 10 families (DeepSeek, GLM,
  Kimi, Qwen, Llama, Gemma, Cohere, MiMo, MiniMax, Step) plus speculative
  draft models.

## New in v2

- `index.html` — v5-style HTML (centered heading, two-column config /
  memory-footprint layout, GitHub pill link, system-font stack, white
  background, no borders).
- `assets/app.js` — ~280-line bridge that wires inputs into
  `KVCacheCalculator.calculate()` and renders results into the new DOM.
  Ports the model-specific predicates from upstream `script.js`
  (TP divisor logic, indexer / draft / linear-attention toggles, precision
  defaults).

## Run

```
open index.html
```

or

```
python3 -m http.server 8000
```

then visit `http://localhost:8000`.

## Inputs

- **Model family** + **Model** — cascading dropdowns
- **Tokens per sequence** / **Sequences (batch)**
- **Tensor Parallel size** — auto-filtered to divisors of KV-head count
- **KV precision** — BF16/FP16 · FP8/INT8 · FP4/INT4
- **Indexer precision** (DeepSeek V3.2/V4, GLM-5) — independent precision toggle
- **Speculative model + tokens** (currently used by Qwen 3.8-27B draft)
- **Include draft KV cache** (MTP / draft layers)
- **Include linear-attention state** (Qwen Linear/Full Hybrid convolution + recurrent)

## Outputs

- **Per-device size** (big indigo stat, GiB)
- **Per sequence · Per token · Bytes/element · Total (all devices)** mini stats
- **Cache groups breakdown** — role, local/total heads, per-device + total bytes
- **Formula details** — selected capacity formula + numeric component breakdown
- **Model configuration source link** (HuggingFace config URL)

## Attribution

See [LICENSE](LICENSE) (MIT) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream authors and model
config sources.

Upstream formulas and model catalog:
<https://github.com/kvcache-ai/kvcache-blog/tree/d455485bd1c5265951849f011a60564da1b23cfe>
