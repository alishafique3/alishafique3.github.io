# Model Explorer

Walk through a Hugging Face `config.json` field by field. Each line links to a
plain-English explanation of what the field does and, where it applies, the
exact parameter count it contributes to the model.

The current example is **GPT-2 Small (124M)** —
[openai-community/gpt2](https://huggingface.co/openai-community/gpt2).

## Run

```
open index.html
```

No build step, no dependencies.

## Layout

- **Left column** — the raw `config.json` content with JSON syntax highlighting.
  Each field is a clickable row.
- **Right column** — one explanation card per field. Cards for fields that
  contribute to parameter count are tagged *"affects params"* and show a
  worked formula with numbers plugged in.
- Click either side to highlight the pair.

At the bottom, a **Parameter count summary** reconstructs the full 124,439,808
parameter total from the config:

| Component | Params |
|---|---|
| Token embeddings `V × h` | 38,597,376 |
| Position embeddings `P × h` | 786,432 |
| 12 × transformer block | 85,054,464 |
| Final layernorm `2h` | 1,536 |
| LM head (tied) | 0 |
| **Total** | **124,439,808** |

### Inside one transformer block (7,087,872 params)

| Sub-layer | Formula | Params |
|---|---|---|
| Attention — `c_attn` (Q, K, V combined) | `3·h² + 3·h` | 1,771,776 |
| Attention — `c_proj` (output) | `h² + h` | 590,592 |
| LayerNorm 1 | `2·h` | 1,536 |
| MLP — `c_fc` (up) | `4·h² + 4·h` | 2,362,368 |
| MLP — `c_proj` (down) | `4·h² + h` | 2,360,064 |
| LayerNorm 2 | `2·h` | 1,536 |

Verified against HF's `model.num_parameters()` → 124,439,808.

## Files

```
index.html       — config column + explanation column + summary
assets/app.js    — field metadata + card rendering + highlight sync
```
