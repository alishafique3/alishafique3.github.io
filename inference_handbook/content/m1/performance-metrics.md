<div class="chapter-meta">Module 1 · Chapter 7</div>

# Key Performance Metrics for LLM Inference

<p class="lede">You cannot tune what you cannot measure, and the metrics that matter for LLM inference are not the metrics most web services report. This chapter defines the standard vocabulary — TTFT, TPOT, ITL, E2E, RPS, TPS, and goodput — and the formulas production teams actually use.</p>

## Per-request latency metrics

- **TTFT (Time To First Token)** — Wall time from request submit to first output token delivered to the client. Dominated by queue time + prefill.
- **TPOT (Time Per Output Token)** — Average time between consecutive output tokens, *excluding* the first (which is TTFT). This is the metric users feel as "streaming speed".
- **ITL (Inter-Token Latency)** — The sequence of gaps between adjacent output tokens. TPOT is the mean (or median) of ITL.
- **E2E Latency** — Total wall time of the request.

The relationship:

$$ \text{E2E} = \text{TTFT} + \text{TPOT} \times (n_{\text{out}} - 1) $$

Measure them separately. If you report only E2E you cannot distinguish a slow prefill from a slow decode, and most optimizations target one or the other.

## Fleet throughput metrics

- **RPS (Requests Per Second)** — Completed requests per second across the fleet.
- **TPS (Tokens Per Second)** — Output tokens per second across the fleet. For a server returning long answers, TPS is a far better capacity signal than RPS.
- **Concurrency** — Average number of in-flight requests. By Little's Law: $\text{Concurrency} = \text{RPS} \times \text{E2E}$.

## Goodput — the metric that matters in production

Raw throughput is meaningless if half the requests violated the SLO. **Goodput** is the rate of requests that *completed within the SLO*:

$$ \text{Goodput} = \frac{\#\{\text{requests completed with TTFT} \le \text{SLO}_{\text{TTFT}} \text{ and TPOT} \le \text{SLO}_{\text{TPOT}}\}}{\text{wall time}} $$

Capacity planning in a serious LLM service is sized to the RPS at which goodput still meets the SLO, with headroom — typically deploying at 60–70 % of the measured ceiling ([M8 Ch. 5](?module=m8&chapter=load-testing-slo)).

## Percentiles, not means

Means hide tail behavior. Always report P50, P95, and P99 for each latency metric. A fleet with P50 TTFT = 200 ms and P99 TTFT = 4 s is a fleet with a serious tail problem that the average will not reveal.

## A minimal measurement sketch

```python
import time, asyncio, statistics
from openai import AsyncOpenAI
client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="x")

async def one_request(prompt, max_tokens=256):
    t0 = time.perf_counter()
    stream = await client.chat.completions.create(
        model="llama-3-8b", messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, stream=True)

    ttft = None; last = t0; itl = []; n = 0
    async for chunk in stream:
        if not chunk.choices or not chunk.choices[0].delta.content: continue
        now = time.perf_counter()
        if ttft is None: ttft = now - t0
        else:            itl.append(now - last)
        last = now; n += 1
    return {"ttft_ms": ttft * 1000, "tpot_ms": statistics.mean(itl)*1000 if itl else 0,
            "e2e_ms": (last - t0) * 1000, "n_out": n}
```

<section class="faq">
<h2>FAQ</h2>
<h3>Why not just measure end-to-end latency?</h3>
<p>You can't tell whether a slow request was slow at prefill or decode. The two have different fixes, so collapsing them hides the actionable signal.</p>
<h3>Is TPOT the same as ITL?</h3>
<p>TPOT is the mean (or median) of ITL per request. ITL is the list of inter-token gaps; TPOT is a scalar summary.</p>
<h3>What's goodput?</h3>
<p>Request rate completed within your SLO latency bounds. A server posting high raw throughput while missing SLO on half of them has low goodput.</p>
<h3>Should I report means or percentiles?</h3>
<p>Percentiles — specifically P50, P95, P99. Means hide tail behavior; the P99 is what wakes up on-call.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin" target="_blank" rel="noreferrer">DistServe (OSDI 2024) — formalizes goodput for LLM serving</a></li>
<li><a href="https://huggingface.co/docs/text-generation-inference/en/reference/metrics" target="_blank" rel="noreferrer">Hugging Face TGI — Metrics reference</a></li>
<li><a href="https://mlcommons.org/en/inference-datacenter-v31/" target="_blank" rel="noreferrer">MLPerf Inference — Datacenter benchmark rules</a></li>
</ul>
</section>

## What's next

With the metrics defined, [Chapter 5 — Prefill and Decode](?module=m1&chapter=prefill-and-decode) explains *why* TTFT and TPOT split the way they do. [Module 8 Chapter 5](?module=m8&chapter=load-testing-slo) builds the load generator and capacity search on top of these primitives.
