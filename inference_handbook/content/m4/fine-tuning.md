<div class="chapter-meta">Module 4 · Chapter 5</div>

# Fine-Tuning Models for Inference

<p class="lede">Fine-tuning specializes a pre-trained model for your task. Done well, a fine-tuned 8 B model beats a generic 70 B on your specific workload at 10 × the throughput. For infra engineers the two questions are: full-parameter fine-tune or LoRA, and how to <em>serve</em> the result efficiently.</p>

## Three fine-tuning styles

| Style | Params updated | Memory | Serving |
|---|---|---|---|
| **Full SFT** | All | Full training memory (~18 × model) | New checkpoint, serve directly |
| **LoRA** | Low-rank adapters only (~0.1–1 % of params) | Few GB | Load base + fuse adapters |
| **QLoRA** | Low-rank adapters on int4 base | ~1 GB for a 7 B model | Base can stay quantized, adapters in fp16 |

## LoRA — the dominant choice

LoRA (Low-Rank Adaptation) freezes the pre-trained weights and injects a small trainable low-rank update $W + BA$ where $B$ and $A$ are rank-$r$ matrices ($r$ typically 8–64). Memory and compute for training drop by 10–100 ×, and inference-time quality typically matches full fine-tuning.

$$ \text{effective weight} = W + \alpha \cdot (B A), \quad B \in \mathbb{R}^{d \times r}, \; A \in \mathbb{R}^{r \times d} $$

## Serving LoRA — fused vs. hot-swap

- **Fused serving.** At load time, merge $BA$ into $W$. Single set of weights, zero inference overhead. Downside: you lose the ability to swap adapters per request.
- **Hot-swap (multi-adapter) serving.** Keep the base model + many adapters in memory; route each request to its adapter. vLLM, SGLang, and TGI all support this via `--enable-lora`. Enables multi-tenant serving of the same base with per-customer specialization.

## A minimal LoRA fine-tune (TRL / PEFT)

```python
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

lora_cfg = LoraConfig(r=16, lora_alpha=32,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                     lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")

model = get_peft_model(model, lora_cfg)

trainer = SFTTrainer(
    model=model, tokenizer=tok,
    args=SFTConfig(output_dir="./out", num_train_epochs=3,
                   per_device_train_batch_size=4, learning_rate=2e-4,
                   bf16=True, logging_steps=10),
    train_dataset=my_dataset,
)
trainer.train()
model.save_pretrained("./qwen-7b-my-task-lora")
```

The output is a tiny (~100 MB) adapter folder. Deploy with vLLM via `--lora-modules name=./qwen-7b-my-task-lora`.

## QLoRA — fine-tune on one GPU

QLoRA quantizes the base model to int4 for training, keeping only the LoRA adapters in bf16. A 7 B model fits full-sequence fine-tuning on a single 24 GB consumer GPU. Quality loss vs. full fp16 LoRA is 0–1 % on most tasks. The pattern that made dev-grade fine-tuning universal.

## When to full-SFT instead

- Shifting the model's style globally (not task-specific).
- Teaching entirely new capabilities beyond what adapters can express.
- Frontier-size models where LoRA's capacity isn't enough.

## Pitfalls

- **Forgetting the chat template.** Fine-tune data must use the model's expected chat template, or serving quality collapses.
- **Catastrophic forgetting.** Aggressive fine-tuning can wipe general capabilities. Mix in generic data or use lower learning rates.
- **LoRA on wrong modules.** Attention modules (`q_proj, k_proj, v_proj, o_proj`) are the usual target. Adding MLP modules helps for domain shift but doubles the cost.
- **Serving adapters without fusing in a single-tenant setup.** Hot-swap has runtime overhead; fuse when you only have one adapter.

<section class="faq">
<h2>FAQ</h2>
<h3>What rank should I pick?</h3>
<p>8–16 for most tasks; 32–64 for heavier fine-tuning. Diminishing returns above 64.</p>
<h3>Can I combine LoRA with quantization?</h3>
<p>Yes — quantize the base to FP8 or int4, serve with LoRA adapters in fp16. Standard production pattern.</p>
<h3>How many examples do I need?</h3>
<p>LoRA needs surprisingly little: 1k–10k high-quality examples is often enough for a focused task.</p>
<h3>Will my fine-tune still work after an upstream model update?</h3>
<p>No. LoRA adapters are tied to the exact base version. If the base updates, re-tune.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://arxiv.org/abs/2106.09685" target="_blank" rel="noreferrer">Hu et al. — LoRA: Low-Rank Adaptation of Large Language Models</a></li>
<li><a href="https://arxiv.org/abs/2305.14314" target="_blank" rel="noreferrer">Dettmers et al. — QLoRA</a></li>
<li><a href="https://huggingface.co/docs/peft/index" target="_blank" rel="noreferrer">PEFT — Parameter-Efficient Fine-Tuning library</a></li>
<li><a href="https://docs.vllm.ai/en/stable/models/lora.html" target="_blank" rel="noreferrer">vLLM — LoRA adapter serving documentation</a></li>
</ul>
</section>

## What's next

Fine-tuning and quantization both change the model. [Chapter 6](?module=m4&chapter=evaluating-optimized) covers how to measure whether the change was worth it — and whether anything regressed quietly.
