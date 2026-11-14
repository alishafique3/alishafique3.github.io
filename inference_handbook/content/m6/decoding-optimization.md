<div class="chapter-meta">Module 6 · Chapter 5</div>
          <h1>Decoding Optimization — Speculative and Long-Context</h1>
          <p class="lede">Two of the biggest decode-side optimizations attack opposite problems. <strong>Speculative decoding</strong> attacks the memory-bound nature of decode by producing multiple tokens per model forward pass. <strong>Long-context techniques</strong> (RoPE scaling, YaRN, sliding window) attack the memory and compute cost of inference at 128k+ tokens.</p>

          <h2>Speculative decoding: the core idea</h2>
          <p>A small, fast <em>draft</em> model proposes the next $K$ tokens. The large <em>target</em> model then verifies all $K$ in a single parallel forward pass. Each accepted token was produced at draft cost; each rejection falls back to the target's prediction. Because decode is memory-bound, the target's forward pass for $K$ candidate tokens costs only slightly more than for one — so every accepted token is near-free.</p>
          <p>Expected speedup $\mathbb{E}[\text{speedup}]$ depends on acceptance rate $\alpha$ and draft/target cost ratio $c$ with $K$ speculative tokens per round:</p>
          <p>$$ \mathbb{E}[\text{speedup}] = \frac{1 - \alpha^{K+1}}{(1 - \alpha)(Kc + 1)} $$</p>
          <p>At $\alpha = 0.85$, $K = 4$, $c = 0.1$: speedup ≈ 2.9× — a realistic number for EAGLE-3 on instruction-following workloads.</p>

          <h2>Drafter flavors</h2>
          <ul>
            <li><strong>Separate small model</strong> — Qwen 2.5 0.5B drafting for Qwen 2.5 32B. Simple, portable, acceptance 50–70%.</li>
            <li><strong>Medusa</strong> — Multiple decoding heads on the target model itself, each predicting a future position. Fast, acceptance 60–80% on general tasks.</li>
            <li><strong>EAGLE-3</strong> — Uses the target's own hidden states to drive a lightweight drafter. Acceptance <strong>80–95%</strong> on instruction-following and coding, now the state of the art. Supported in vLLM and SGLang as a configuration flag.</li>
            <li><strong>n-gram / lookahead</strong> — Draft from the prompt or recent output itself, zero extra model to deploy. Lower acceptance but near-zero deployment overhead; wins for tasks with heavy input copying (code edits, structured outputs).</li>
          </ul>

          <h2>When speculative decoding wins and loses</h2>
          <ul>
            <li>Wins at low batch — decode is most memory-bound there, so accepted tokens are nearly free.</li>
            <li>Loses at high batch — the target is already saturating compute, and verification overhead cuts into aggregate throughput.</li>
            <li>Loses when acceptance drops below ~60% — rejected draft tokens were wasted work.</li>
          </ul>
          <p>Production serving stacks typically enable speculative decoding adaptively — on during low-traffic hours, off when the batch is already full.</p>

          <h2>Long-context inference</h2>
          <p>Two problems scale badly with context length $s$:</p>
          <ul>
            <li><strong>Prefill compute</strong> grows as $O(s^2)$ due to the attention score matrix. A 32k-token prompt is not 4× more expensive than 8k — it's 16×.</li>
            <li><strong>KV cache memory</strong> grows as $O(s)$ per sequence. For Llama-3 70B at 128k context, KV is ~40 GB per sequence.</li>
          </ul>

          <h3>Position encoding and extension</h3>
          <p>Modern LLMs use <strong>RoPE</strong> (Rotary Position Embeddings), which encode position by rotating Q and K vectors by angles that depend on position. Each dimension gets a different frequency; low-frequency dimensions encode long-range position, high-frequency dimensions encode local position.</p>
          <p>Models trained at 4k don't naturally work at 128k because the rotation angles go out of distribution. <strong>YaRN</strong> (Yet another RoPE extensioN) is the dominant fix: it does <em>NTK-by-parts</em> scaling, keeping high-frequency dimensions intact while interpolating low-frequency ones, and applies a temperature correction to attention. YaRN typically requires 10× less fine-tuning compute than earlier context-extension methods, which is why most 128k-context open models in 2025+ ship with YaRN scaling.</p>

          <h3>Sliding-window attention</h3>
          <p>An alternative to extending full attention: just don't attend beyond a window of $W$ recent tokens. Prefill compute drops to $O(s W)$, KV grows to $O(W)$ — constant at long enough $s$. The cost is weaker long-range dependencies; models like Gemma use a hybrid (some full-attention layers, some sliding) to get the best of both.</p>

          <h2>Pitfalls</h2>
          <ul>
            <li><strong>Enabling speculative decoding without measuring acceptance.</strong> Below 60% acceptance you likely lost throughput.</li>
            <li><strong>Treating YaRN as free quality.</strong> Long-context extension always costs quality on long-context retrieval benchmarks — measure with needle-in-haystack before deploying.</li>
            <li><strong>Ignoring prefill quadratic cost.</strong> A product that accepts 100k-token prompts needs chunked prefill plus a big prefill budget, or TTFTs will be in seconds.</li>
          </ul>

          <section class="faq">
            <h2>FAQ</h2>
            <h3>What acceptance rate does EAGLE-3 reach?</h3>
            <p>80–95% on instruction-following and coding tasks in published benchmarks; 50–70% on open-ended generation. Measure on your own workload before shipping.</p>
            <h3>When does speculative decoding hurt throughput?</h3>
            <p>At high batch (target already saturates compute, verification overhead is pure loss) or at low acceptance rate (&lt; 60%, where draft tokens are mostly wasted).</p>
            <h3>Is YaRN free?</h3>
            <p>No — it costs quality on long-context retrieval benchmarks (needle-in-haystack tests). Measure on your own long-context queries before deploying.</p>
            <h3>Should I use sliding-window attention?</h3>
            <p>Yes if your workload has explicit locality (code autocomplete, tail-of-chat follow-ups). No if the model needs long-range dependencies (summarization, RAG with early context).</p>
          </section>

          <section>
            <h2>References</h2>
            <ul class="refs">
              <li><a href="https://arxiv.org/abs/2309.00071" target="_blank" rel="noreferrer">Peng et al. — YaRN: Efficient Context Window Extension of Large Language Models</a></li>
              <li><a href="https://arxiv.org/abs/2503.01840" target="_blank" rel="noreferrer">Li et al. — EAGLE-3</a></li>
              <li><a href="https://arxiv.org/abs/2401.10774" target="_blank" rel="noreferrer">Cai et al. — Medusa: Simple LLM Inference Acceleration</a></li>
              <li><a href="https://docs.vllm.ai/en/stable/features/spec_decode.html" target="_blank" rel="noreferrer">vLLM — Speculative decoding documentation</a></li>
            </ul>
          </section>

          <h2>What's next</h2>
          <p>Speculative decoding interacts with the <a href="?module=m6&chapter=batching-scheduling">scheduler</a> and the <a href="?module=m6&chapter=distributed-parallelism">parallelism layout</a>. Long-context runtime cost connects directly to <a href="?module=m9&chapter=energy-tokens-per-joule">energy</a> — every token takes more joules at 128k than at 2k.</p>
