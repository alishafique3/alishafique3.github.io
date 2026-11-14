<div class="chapter-meta">Module 6 · Chapter 3</div>
          <h1>KV Cache Management and PagedAttention</h1>
          <p class="lede">The KV cache is the memory where every token's Keys and Values from previous attention layers live, waiting to be read on the next decode step. On any serious serving GPU, the KV cache is the <em>largest</em> allocation after model weights — and in high-concurrency scenarios, it's the thing that silently kills throughput through fragmentation. <strong>PagedAttention</strong>, the technique that launched vLLM, borrows virtual memory from operating systems to end the fragmentation problem.</p>

          <h2>Why the KV cache exists</h2>
          <p>In autoregressive generation, each new token attends to every previous token. If you re-computed $K$ and $V$ for every previous position on every decode step, decode would be $O(n^2)$ in compute per step, not $O(n)$ — catastrophic. The KV cache stores every $K$ and $V$ computed in prefill (and in subsequent decode steps), so only the <em>new</em> token's $K$ and $V$ need to be computed per step.</p>

          <h2>How big is it, really?</h2>
          <p>Per sequence, the KV cache grows linearly with length:</p>
          <p>$$ \text{KV}(s) = 2 \cdot L \cdot s \cdot n_{kv} \cdot d_h \cdot b_{\text{elem}} $$</p>
          <p>Llama-3 70B GQA ($L=80, n_{kv}=8, d_h=128$, BF16): <strong>0.31 MB per token</strong>. At 128k context that is 40 GB per sequence — more than half an H100.</p>

          <h2>The fragmentation problem (pre-PagedAttention)</h2>
          <p>Classic serving engines allocated one contiguous block of KV memory per request, sized to the maximum possible output length. Problems:</p>
          <ul>
            <li>You rarely use the max — typical answer is a fraction of it. The unused tail is dead memory.</li>
            <li>When a sequence finishes, its block is freed but adjacent blocks still occupied. The next request needs a contiguous block and must look elsewhere — classic external fragmentation.</li>
            <li>Effective GPU memory utilization sat at 20–40%.</li>
          </ul>

          <h2>PagedAttention</h2>
          <p>vLLM's insight: treat KV memory like an OS treats RAM. Split KV into small fixed-size <strong>blocks</strong> (e.g., 16 tokens of KV per block), and let each sequence hold a <em>block table</em> — a list of pointers to arbitrary physical blocks. Logically contiguous; physically scattered.</p>
          <p>Attention reads jump through the block table. The cost is a slightly more complex kernel; the win is that fragmentation goes to near zero and <strong>effective KV capacity grows 2–4×</strong>. More capacity → larger batches → higher throughput → lower TPOT at the same hardware.</p>

          <h2>Prefix caching sits on top</h2>
          <p>Because blocks are content-addressable (hash of their tokens), identical prefixes across requests can <em>share</em> the same physical blocks. A system prompt reused across 10k requests is computed once and reused everywhere. For chat workloads with heavy system prompts or multi-turn history, prefix caching can cut prefill cost by 50–80% in practice.</p>

          <h2>A minimal picture of the block table</h2>
          <pre><span class="lang-label">text</span><code>Sequence A (logical):  [blk0][blk1][blk2][blk3]
Sequence A (physical): [p17 ][p 4 ][p82 ][p 9 ]      # scattered in HBM
Sequence B (logical):  [blk0][blk1][blk2]
Sequence B (physical): [p17 ][p 4 ][p33 ]            # p17,p4 SHARED with A
                                                      # (common system prompt)</code></pre>
          <p>The attention kernel receives the block table and gathers from the right physical blocks — a pointer-chasing pattern GPUs now handle efficiently.</p>

          <h2>Worked numbers</h2>
          <p>On a 2×H100 node serving Llama-3 70B with 48 concurrent sequences averaging 2k context: classic contiguous allocation ends up fitting ~16 sequences at steady state after fragmentation; PagedAttention fits all 48 and leaves headroom. That's a ~3× throughput improvement from one scheduling decision.</p>

          <h2>Pitfalls</h2>
          <ul>
            <li><strong>Block size trade-off.</strong> Small blocks reduce internal fragmentation but add block-table overhead. 16-token blocks are the de facto default.</li>
            <li><strong>Assuming prefix caching is free.</strong> It stores hashes and keeps blocks alive — a long-tail of unique system prompts can reduce effectiveness.</li>
            <li><strong>Not accounting for KV in capacity planning.</strong> Your max concurrent sequence count is bounded by KV capacity, not just TP shards.</li>
          </ul>

          <section class="faq">
            <h2>FAQ</h2>
            <h3>What's the default PagedAttention block size?</h3>
            <p>Typically 16 tokens per block. Smaller reduces internal fragmentation but adds block-table overhead.</p>
            <h3>Does prefix caching happen automatically?</h3>
            <p>vLLM enables it by default on recent versions. TensorRT-LLM via config. Effectiveness depends on your prompts actually having shared prefixes.</p>
            <h3>How much capacity does PagedAttention actually add?</h3>
            <p>2–4× effective KV capacity vs. contiguous pre-allocation at the same hardware. The paper reports 2–4× throughput improvement on identical GPUs.</p>
            <h3>Can I use PagedAttention without vLLM?</h3>
            <p>Yes — SGLang, TensorRT-LLM, Mooncake, and several others implement the same idea. The term "PagedAttention" originated with vLLM; the concept is now universal.</p>
          </section>

          <section>
            <h2>References</h2>
            <ul class="refs">
              <li><a href="https://dl.acm.org/doi/10.1145/3600006.3613165" target="_blank" rel="noreferrer">Kwon et al. — Efficient Memory Management for LLM Serving with PagedAttention (SOSP 2023)</a></li>
              <li><a href="https://alishafique3.github.io/kv-cache-calculator/" target="_blank" rel="noreferrer">KV Cache Calculator — interactive KV sizing</a></li>
              <li><a href="https://arxiv.org/abs/2312.07104" target="_blank" rel="noreferrer">Zheng et al. — SGLang RadixAttention (extended prefix sharing)</a></li>
            </ul>
          </section>

          <h2>What's next</h2>
          <p>KV cache capacity is the dominant memory budget — see <a href="?module=m6&chapter=kv-cache-optimization">Chapter 4</a> for cache quantization and offloading, and the <a href="https://alishafique3.github.io/kv-cache-calculator/" target="_blank" rel="noreferrer">KV Cache Calculator</a> for interactive sizing.</p>
