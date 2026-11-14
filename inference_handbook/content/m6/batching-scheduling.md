<div class="chapter-meta">Module 6 · Chapter 2</div>
          <h1>Batching and Scheduling</h1>
          <p class="lede">The single most important optimization in modern LLM serving is batching — running many sequences through the model together to amortize the memory-bandwidth cost of reading weights. Modern engines go further with <strong>continuous batching</strong>, where new requests join the batch mid-decode, and <strong>chunked prefill</strong>, which prevents long prompts from blocking the decode queue.</p>

          <h2>Three eras of batching</h2>
          <ol>
            <li><strong>Static batching</strong> — Collect N requests, run them together, wait for the slowest to finish, return everyone, start over. Simple, catastrophic for latency variance. A request answering in 20 tokens waits for the one answering in 2,000.</li>
            <li><strong>Dynamic batching</strong> — Pad all sequences to the longest and run together, but admit the next batch once the current finishes. Better but still pads everything to max length and still blocks on stragglers.</li>
            <li><strong>Continuous (iteration-level) batching</strong> — Schedule at the token-generation step, not the request. Every iteration, the scheduler decides: which running sequences get a decode step, which new requests get prefilled. Finished sequences are evicted; new ones join seamlessly. This is what vLLM introduced in 2023 and what every serious engine does today.</li>
          </ol>

          <h2>Why continuous batching wins</h2>
          <p>Decode per token is memory-bound at the model-weight level. Reading 140 GB of weights (Llama-3 70B at BF16) takes ~20 ms on H100 regardless of whether you produce 1 token or 32. Continuous batching lets a decode step produce up to $B$ tokens for the same weight read, so <em>tokens/joule</em>, <em>tokens/second</em>, and <em>tokens/$</em> all scale nearly linearly with $B$ until the kernel saturates compute.</p>
          <p>vLLM's own benchmarks on v0.6.0 reported <strong>2.7× higher throughput and 5× lower TPOT</strong> for Llama-3 8B on one H100 vs. the prior generation of serving engines, driven almost entirely by scheduling + continuous batching improvements.</p>

          <h2>Chunked prefill</h2>
          <p>A single 16k-token prompt would otherwise block every running decode for hundreds of milliseconds (prefill is quadratic in sequence length). <strong>Chunked prefill</strong> breaks the prompt into pieces (e.g., 512 tokens at a time) and interleaves them with decode steps for other sequences. TTFT for the long request grows slightly; TPOT for every other active request stays flat. The scheduler trades one user's TTFT for the whole fleet's TPOT — usually the right call under load.</p>

          <h2>The scheduling loop in pseudocode</h2>
          <pre><span class="lang-label">python</span><code>def schedule_step(running, waiting, kv_budget_tokens):
    # Prepare a mixed batch of decode + prefill for one forward pass
    batch = []
    # 1. Keep as many running (decode) sequences as fit
    for seq in running:
        if remaining_kv(seq) &gt; 0:
            batch.append(("decode", seq, 1))              # 1 new token
    # 2. Admit new sequences, chunking long prompts
    budget = kv_budget_tokens - sum(prompt_len(s) for _, s, _ in batch)
    for req in waiting:
        chunk = min(req.remaining_prefill, 512, budget)
        if chunk &lt;= 0: break
        batch.append(("prefill", req, chunk))
        budget -= chunk
    return batch</code></pre>
          <p>Every real engine is more sophisticated (priority, fairness, preemption, admission control), but the shape is this.</p>

          <h2>Pitfalls</h2>
          <ul>
            <li><strong>Over-batching when you're prefill-bound.</strong> Big batches during decode are free; big batches during prefill compete for the same compute. If your workload is prompt-heavy, chunked prefill matters more than batch size.</li>
            <li><strong>Head-of-line blocking from one huge prompt.</strong> Without chunked prefill, a 32k-token prompt stops the world for ~1 s on H100 — every active decode pauses.</li>
            <li><strong>Assuming linear batch scaling forever.</strong> Once you clear the ridge point (Module 2 Ch. 7), you're compute-bound and more batch doesn't help.</li>
          </ul>

          <section class="faq">
            <h2>FAQ</h2>
            <h3>What's the difference between dynamic and continuous batching?</h3>
            <p>Dynamic batching groups requests and waits for the slowest in the group to finish. Continuous batching admits and evicts sequences at every iteration, so a finished sequence immediately frees capacity.</p>
            <h3>Can I set the batch size manually?</h3>
            <p>Most engines size the batch automatically against KV cache capacity. You set an upper bound (<code>max_num_seqs</code> in vLLM) rather than a target.</p>
            <h3>Is chunked prefill always on?</h3>
            <p>vLLM enables it by default for mixed workloads. SGLang and TensorRT-LLM expose it as a configuration flag.</p>
            <h3>Does batching help prefill?</h3>
            <p>Less than it helps decode. Prefill is compute-bound, so batching mostly adds contention. Chunked prefill is the right lever for prefill-heavy workloads.</p>
          </section>

          <section>
            <h2>References</h2>
            <ul class="refs">
              <li><a href="https://www.usenix.org/conference/osdi22/presentation/yu" target="_blank" rel="noreferrer">Yu et al. — Orca (OSDI 2022): introduced iteration-level scheduling</a></li>
              <li><a href="https://vllm.ai/blog/2024-09-05-perf-update" target="_blank" rel="noreferrer">vLLM Blog — v0.6.0: 2.7× throughput improvement</a></li>
              <li><a href="https://docs.vllm.ai/en/stable/features/chunked_prefill.html" target="_blank" rel="noreferrer">vLLM — Chunked prefill documentation</a></li>
            </ul>
          </section>

          <h2>What's next</h2>
          <p>The batch size you can actually run is bounded by the <a href="?module=m6&chapter=kv-cache-paged">KV cache memory</a>. Continuous batching exists for the same memory-bandwidth reason that <a href="?module=m6&chapter=decoding-optimization">speculative decoding</a> pays off.</p>
