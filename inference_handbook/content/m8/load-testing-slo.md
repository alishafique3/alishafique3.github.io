<div class="chapter-meta">Module 8 · Chapter 5</div>

# Benchmarking, Load Testing, and SLO Engineering

<p class="lede">Benchmarks measure peak throughput; load tests measure <strong>sustained goodput under a realistic workload</strong>; SLO engineering decides what tradeoffs you will promise. These three activities are different. Confusing them is why "the vendor said 10,000 TPS" turns into "we barely do 2,000 in production." This chapter is the rigorous version.</p>

## The three activities

- **Benchmarking** — synthetic workload, maximum numbers, no SLO constraint. Answers "how fast can this go?"
- **Load testing** — realistic workload mix, realistic concurrency, measured against your SLO. Answers "what RPS can we promise?"
- **SLO engineering** — tuning the promises themselves. Answers "where is the acceptable operating point?"

Each builds on the previous.

## A production-grade load generator in one file

```python
import asyncio, time, random, statistics
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="x")

async def measure_request(prompt, max_tokens, workload_class):
    t0 = time.perf_counter()
    stream = await client.chat.completions.create(
        model="llama-3.1-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=0.7, stream=True,
        stream_options={"include_usage": True},
    )
    ttft = None; last = t0; itl = []; n = 0
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            now = time.perf_counter()
            if ttft is None: ttft = now - t0
            else:            itl.append(now - last)
            last = now; n += 1
    return {
        "ttft_ms": (ttft or 0) * 1000,
        "tpot_ms": statistics.mean(itl) * 1000 if itl else 0,
        "e2e_ms": (last - t0) * 1000,
        "n_out": n,
        "workload": workload_class,
    }

async def workload(rps, duration_s, sampler):
    tasks = []
    end = time.time() + duration_s
    interval = 1.0 / rps
    while time.time() < end:
        prompt, max_tok, cls = sampler()
        tasks.append(asyncio.create_task(measure_request(prompt, max_tok, cls)))
        await asyncio.sleep(interval)
    return await asyncio.gather(*tasks, return_exceptions=True)
```

Add percentile reporting and a mixed workload sampler (short chat / medium QA / long summary) to match your production distribution.

## Mixed workload — realistic, not uniform

Real traffic is heterogeneous. For a chat product, one reasonable mix:

- 60 % short (chat reply, ~25-token prompt, ~48-token output).
- 30 % medium (QA, ~120-token prompt, ~96-token output).
- 10 % long (summarization, ~700-token prompt, ~192-token output).

Load-testing with 100 % one workload class gives misleadingly clean numbers. The mix is what breaks your service.

## Binary-searching for max sustainable RPS

Find the RPS ceiling under an SLO via binary search:

```python
async def find_max_rps(slo, sampler, low=1, high=50, duration_s=30):
    while high - low > 1:
        mid = (low + high) // 2
        results = await workload(mid, duration_s, sampler)
        ok = [r for r in results if isinstance(r, dict)]
        ttft_p99 = sorted(r["ttft_ms"] for r in ok)[int(0.99 * len(ok))]
        tpot_p99 = sorted(r["tpot_ms"] for r in ok)[int(0.99 * len(ok))]
        if ttft_p99 <= slo["ttft_ms"] and tpot_p99 <= slo["tpot_ms"]:
            low = mid
        else:
            high = mid
    return low
```

Returns the max RPS at which both SLOs still hold.

## Goodput at multiple SLO levels

One capacity number isn't enough — you want to know how capacity changes as you tighten the SLO:

| SLO tightness | TTFT P99 | TPOT P99 | Max RPS |
|---|---|---|---|
| Relaxed | 1500 ms | 100 ms | 18 |
| Standard | 800 ms | 60 ms | 12 |
| Tight | 400 ms | 40 ms | 7 |

The gap between relaxed and tight tells you the "overhead of promises." If tight is half of relaxed, you have significant tail traffic.

## Headroom math

Never deploy to the ceiling. Rule of thumb from M3.4:

$$ \text{deploy target} = 0.65 \times \text{max RPS under SLO} $$

For the standard SLO above (12 RPS ceiling), deploy at ~7-8 RPS. The gap absorbs spikes, cold starts, node failures.

## Cross-check with vendor benchmarks

Every engine ships `bench serve`:

```bash
vllm bench serve \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --host 127.0.0.1 --port 8000 \
    --dataset-name sharegpt \
    --num-prompts 500 \
    --request-rate 10
```

Use this for two things:

1. Sanity-check your load generator (vendor numbers should be within 10 % of yours at peak RPS).
2. Compare engines (vLLM vs SGLang vs TGI on identical hardware).

Do not use vendor benchmarks as your capacity plan. They're peak, not sustained; synthetic, not real.

## What the numbers you'll see look like

Rough late-2026 numbers for Llama 3.1 70B at BF16/FP8, standard SLO (TTFT P99 ≤ 800 ms, TPOT P99 ≤ 60 ms):

| Hardware | Engine | Max RPS |
|---|---|---|
| 1 × H100 | vLLM, Q4 | ~2–3 |
| 2 × H100 | vLLM, FP8 | ~10–15 |
| 2 × H200 | vLLM, FP8 | ~15–22 |
| 8 × H100 | SGLang, FP8 + RadixAttention | ~35–50 |
| 1 × B200 | vLLM, FP4 | ~20–30 |

Your mileage will differ. These are order-of-magnitude anchors.

## Pitfalls

- **Load testing against a cold cache.** Warm up with 30+ seconds of traffic before measuring.
- **Co-located load generator.** The generator steals CPU from the server. Run from a different machine.
- **Testing without the gateway.** Gateway adds latency, queueing, rate limits. Test the real path.
- **Reporting average TTFT.** Averages hide tail. Always P99.
- **Ignoring errors.** A 500 that returns in 100 ms is not a successful request. Count errors.

<section class="faq">
<h2>FAQ</h2>
<h3>How long should a load test run?</h3>
<p>5–10 minutes at each RPS level. Longer catches steady-state effects (KV churn, cache saturation); shorter burns through steps faster.</p>
<h3>Should I use K6, Locust, or vLLM's bench?</h3>
<p>K6/Locust are language-generic; vLLM's bench is purpose-built. All work. The custom Python loop above is simpler to modify when you need a specific workload shape.</p>
<h3>Where should the load generator run?</h3>
<p>From a separate VM in the same region as your servers. Same region = realistic network latency. Different machine = no CPU contention.</p>
<h3>How do I set an SLO initially?</h3>
<p>Measure current performance at low load → add 20 % margin → that's your initial target. Tighten later as you optimize.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://docs.vllm.ai/en/stable/performance/benchmarks.html" target="_blank" rel="noreferrer">vLLM — Benchmark documentation</a></li>
<li><a href="https://github.com/huggingface/inference-benchmarker" target="_blank" rel="noreferrer">Hugging Face inference-benchmarker</a></li>
<li><a href="https://k6.io/docs/testing-guides/load-testing/" target="_blank" rel="noreferrer">k6 — Load testing guide</a></li>
</ul>
</section>

## What's next

[Chapter 6](?module=m8&chapter=multi-model-multi-cloud) covers routing and placement across models and clouds. [Chapter 7](?module=m8&chapter=inference-ops) wraps the module with InferenceOps — the lifecycle management of all this.
