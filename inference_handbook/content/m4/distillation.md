<div class="chapter-meta">Module 4 · Chapter 4</div>

# Model Distillation

<p class="lede">Distillation trains a small <strong>student</strong> model to imitate a larger <strong>teacher</strong>. Done well, a 7 B distilled student reaches 85–95% of a 70 B teacher's quality at 10 × the throughput and 10 × less memory. For infra engineers, distillation is the cheapest way to cut serving cost without sacrificing much of what users feel.</p>

## The two flavors

- **Response-level distillation.** The teacher generates synthetic answers to a prompt set; the student is trained on those answers via standard SFT. Simplest to run; no changes to training loop.
- **Logit-level distillation.** Train the student to match the teacher's full next-token distribution, not just the sampled token. Uses KL-divergence loss. More data-efficient; needs access to teacher logits (so works only with open-weight teachers).

## The classical loss

$$ \mathcal{L}_{\text{distill}} = \alpha \cdot T^2 \cdot \text{KL}(p_T \mathbin\| p_S) + (1 - \alpha) \cdot \mathcal{L}_{\text{CE}}(y, p_S) $$

$p_T$ and $p_S$ are the teacher and student softmax distributions temperature-smoothed by $T$. The KL term matches distributions; the cross-entropy term still enforces the ground-truth labels. $\alpha$ typically 0.5–0.9.

## When distillation beats "use a smaller base"

- Your task is narrow (customer service, SQL generation, classification). A distilled student on that distribution often outperforms a bigger generic model.
- Your teacher is behind an API you want to escape. Distillation into an open-weight student gives you cost control and privacy.
- You need speculative decoding draft models. A distilled drafter with high acceptance rate is a straightforward application.

## When it doesn't

- **Reasoning depth.** Distilled 7 B students often lose chain-of-thought depth. If your task leans on multi-step reasoning, test carefully.
- **Long-tail knowledge.** Distillation compresses what's used; obscure facts in the teacher may vanish.
- **Agentic workloads.** Tool-use / plan quality tends to regress more than straight chat quality.

## A minimal distillation sketch (Hugging Face)

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

student = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B",
                                               torch_dtype=torch.bfloat16).cuda()
teacher = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-14B",
                                               torch_dtype=torch.bfloat16).cuda()
teacher.eval(); teacher.requires_grad_(False)

T, alpha = 2.0, 0.7

def distill_step(input_ids, labels):
    with torch.no_grad():
        teacher_logits = teacher(input_ids).logits / T
    student_logits = student(input_ids).logits
    kl = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction="batchmean") * (T ** 2)
    ce = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)),
                         labels.view(-1), ignore_index=-100)
    return alpha * kl + (1 - alpha) * ce
```

Real recipes use HF Trainer, DeepSpeed/ZeRO for the teacher, and specialized libraries like `distillery` or `axolotl`. The core idea is this.

## A production example

DeepSeek-V3 distillation into smaller open students (~32 B) in 2025–2026 reached 85–92% of the teacher's lm-eval average while serving at 4 × the throughput on the same hardware. The DeepSeek-R1-Distill series popularized the pattern for reasoning tasks.

## Pitfalls

- **Distilling on the wrong distribution.** Students take on the character of their training data. If you distill a general chatbot on a narrow eval, you'll miss generalization.
- **Over-trusting MMLU as the quality signal.** MMLU rarely drops much; chain-of-thought and agent tasks regress more.
- **Treating response-level as "free" relative to logit-level.** Logit-level is dramatically more data-efficient when you can access the teacher's logits.
- **Mixing distillation and quantization without evaluating the combination.** Each is small; stacked they can add up.

<section class="faq">
<h2>FAQ</h2>
<h3>Can I distill a closed-weight teacher like GPT-4?</h3>
<p>Response-level, yes — you query the API, keep the answers, and SFT. Check the provider's ToS first; some explicitly forbid using outputs to train competing models.</p>
<h3>Is distillation the same as fine-tuning?</h3>
<p>Related but different. Fine-tuning changes the model to fit your data. Distillation changes a smaller model to mimic a bigger one. You can do both.</p>
<h3>How much data do I need?</h3>
<p>Response-level: typically 10–100k examples for a narrow task; 100k+ for general chat. Logit-level: 3–5 × less data for similar quality.</p>
<h3>What's "self-distillation"?</h3>
<p>Training a model to match its own earlier checkpoint. Used to regularize or compress. Different flavor; less common in LLM serving.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://arxiv.org/abs/1503.02531" target="_blank" rel="noreferrer">Hinton, Vinyals, Dean — Distilling the Knowledge in a Neural Network</a></li>
<li><a href="https://arxiv.org/abs/2306.08543" target="_blank" rel="noreferrer">Hsieh et al. — Distilling Step-by-Step</a></li>
<li><a href="https://huggingface.co/deepseek-ai" target="_blank" rel="noreferrer">DeepSeek — model distillation series on Hugging Face</a></li>
</ul>
</section>

## What's next

Distillation and fine-tuning are closely related. [Chapter 5](?module=m4&chapter=fine-tuning) covers SFT, LoRA, and QLoRA as alternative or complementary approaches.
