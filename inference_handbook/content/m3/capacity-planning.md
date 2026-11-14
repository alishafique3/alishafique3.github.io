<div class="chapter-meta">Module 3 · Chapter 4</div>

# Capacity Planning Fundamentals

<p class="lede">Capacity planning answers one question: <em>how many GPUs do we need?</em> The honest answer requires the workload shape (prompt / output token distribution), the latency budget (SLO), and the throughput a single replica delivers under that SLO. This chapter walks the full calculation end-to-end for a realistic workload.</p>

## The capacity formula

$$ \text{replicas} = \left\lceil \frac{\text{peak RPS} \times \text{safety}}{\text{sustainable goodput per replica}} \right\rceil $$

The ingredients:

- **Peak RPS** — the top of your traffic curve, not the mean. Usually 2–4 × daily average for a chat workload.
- **Safety factor** — multiplier for deploys, spikes, noisy neighbors. Typical: 1.3–1.5.
- **Sustainable goodput per replica** — RPS a single serving replica can hold while meeting the SLO. Measured, not guessed.

## What inputs you need first

1. **Workload shape.** Mean and P99 prompt length. Mean and P99 output length. Mix of request types (short chat, long summary, agent tool call).
2. **SLO.** P99 TTFT target and P99 TPOT target. Typical production SLO: TTFT P99 ≤ 500 ms, TPOT P99 ≤ 50 ms.
3. **Peak RPS.** From traffic data, product forecast, or market sizing.
4. **Growth horizon.** Plan for 6–12 months of traffic, not today's.

## Little's Law — the one formula everybody forgets

$$ L = \lambda \cdot W $$

Concurrent requests in flight = request rate × average time each takes. So if TTFT+TPOT×out = 2 s and RPS = 20, you have 40 requests in flight on average. This tells you the *batch size* you must sustain, which tells you the KV cache you need, which tells you the hardware.

## A worked example — a chat product

Workload:

- Mean prompt 300 tokens, P99 2,000.
- Mean output 200 tokens, P99 800.
- SLO: TTFT P99 ≤ 800 ms, TPOT P99 ≤ 60 ms.
- Peak RPS: 40.

Model + hardware:

- Llama 3.1 70B FP8 on 2 × H100 with vLLM + continuous batching.

Measured goodput (via load test, Module 8 Ch. 5): 12 RPS per replica under this SLO.

Capacity:

- $\lceil 40 \times 1.4 / 12 \rceil = \lceil 4.67 \rceil = 5$ replicas.
- Each replica = 2 × H100 → 10 H100s total.

Little's Law sanity check: average request E2E = TTFT + TPOT × output = 0.6 + 0.05 × 200 = 10.6 s. At 40 RPS, in-flight = 40 × 10.6 = 424 concurrent sequences across 5 replicas → ~85 per replica. KV cache for 85 × 2k context on 2×H100 TP=2 at FP8 fits comfortably (~22 GB per GPU). Math closes.

## Headroom and why to leave some

A system running at 100 % of its measured ceiling will fail the SLO at the first 20 % spike. Deploy at 60–70 % of measured goodput:

$$ \text{deploy at} \approx 0.65 \times \text{max sustainable RPS under SLO} $$

The gap is for traffic spikes, GPU node failures, cold starts, and unknown-unknowns. For regulated workloads or life-critical uptime, drop to 50 %.

## Minimum code to start the measurement

```python
import asyncio, time, random
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="x")

async def one_request(prompt_len, out_len):
    prompt = "Explain: " + "stuff. " * (prompt_len // 2)
    t0 = time.perf_counter()
    r = await client.chat.completions.create(
        model="llama-3-70b", messages=[{"role": "user", "content": prompt}],
        max_tokens=out_len, temperature=0.7)
    return time.perf_counter() - t0

async def workload(rps, duration_s):
    tasks = []
    end = time.time() + duration_s
    interval = 1.0 / rps
    while time.time() < end:
        p = random.choice([300, 300, 300, 2000])    # 25% long prompts
        o = random.choice([200, 200, 200, 800])
        tasks.append(asyncio.create_task(one_request(p, o)))
        await asyncio.sleep(interval)
    return await asyncio.gather(*tasks, return_exceptions=True)
```

Run this across a range of RPS values; find where P99 TTFT or TPOT breaches the SLO. That's your ceiling.

## Pitfalls

- **Using mean prompt/output length instead of P99.** The tail dominates GPU saturation.
- **Sizing to the average day, not peak.** Average RPS is a lie; peak is what matters.
- **Forgetting the growth horizon.** If you ship at exactly today's RPS, you repeat this exercise in three months.
- **Reporting benchmark throughput as "capacity".** Benchmarks are peak. Capacity is sustained under SLO.

<section class="faq">
<h2>FAQ</h2>
<h3>What's goodput again?</h3>
<p>Request rate completed within SLO latency bounds — defined in <a href="?module=m1&chapter=performance-metrics">M1 Ch. 7</a>. Raw throughput ignores whether each request was fast enough.</p>
<h3>How do I predict goodput without a server to measure?</h3>
<p>You can't, accurately. For an estimate: take published vLLM benchmarks (Llama-3 70B on 2×H100 ≈ 12–20 RPS at mild SLO), then halve for your workload shape until you measure.</p>
<h3>Do I really need 60% headroom?</h3>
<p>At steady scale yes. For dev/staging you can run at 90%. Beware that incidents compound — once goodput drops, your retry traffic multiplies, and 70 % utilization becomes 100 % in under a minute.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://en.wikipedia.org/wiki/Little%27s_law" target="_blank" rel="noreferrer">Little's Law (reference)</a></li>
<li><a href="https://docs.vllm.ai/en/stable/performance/benchmarks.html" target="_blank" rel="noreferrer">vLLM — Benchmarks documentation</a></li>
<li><a href="https://github.com/huggingface/inference-benchmarker" target="_blank" rel="noreferrer">Hugging Face — inference-benchmarker</a></li>
</ul>
</section>

## What's next

With capacity sized, pick the framework in [Chapter 5](?module=m3&chapter=inference-framework) and then actually spin one up in [Chapter 6 — Serving Your First LLM](?module=m3&chapter=serving-first-llm). The full load-test methodology is in [M8 Ch. 5](?module=m8&chapter=load-testing-slo).
