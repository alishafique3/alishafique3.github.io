<div class="chapter-meta">Module 9 · Chapter 6</div>

# Cost-Aware and Energy-Aware Inference Scheduling

<p class="lede">Latency is one objective. Dollars and joules are two more. A scheduler that can see all three decides not just <em>what</em> runs but <em>where</em> and <em>when</em>: on which model tier, at which hour, in which region. This chapter covers the practical schedulers that trade quality, cost, and energy together.</p>

## The three-axis scheduling problem

Classical scheduling: minimize latency subject to throughput constraints. The LLM version adds two more objectives:

$$ \min\;\;(\text{latency},\; \$,\; J) $$

Not all three can be optimized at once. You pick a dominant axis and treat the others as constraints.

## Model cascades — cost-aware routing

Introduced in M8.6. Route to a small cheap model first; escalate to a large expensive model only when needed.

```python
def cascade(prompt):
    cheap = call("qwen-2.5-7b", prompt, logprobs=True)
    if avg_logprob(cheap) > THRESHOLD:
        return cheap.text
    return call("llama-3.1-70b", prompt).text
```

Reported wins: 3–5× cost reduction while meeting the same quality bar. Energy savings follow cost — small model is ~10 × cheaper in J/token.

Confidence signals (M8.6): logprob threshold, uncertainty, output length, LLM judge.

## Token-budget-aware routing

Variation on cascades: route based on expected output length, which correlates with cost.

- Short predicted output (under ~50 tokens) → small model.
- Medium output → medium tier (if you have one).
- Long output → large model where quality matters.

Predict output length from a tiny classifier on the prompt. ~70 % accurate; good enough to save cost on obvious cases.

## Carbon-aware deferrals

For non-urgent batch jobs (nightly summarization, offline enrichment):

- **Grid carbon intensity varies** — sometimes by 3× within one day (solar hours vs. gas-heavy evenings).
- **Delay jobs to green hours** — the same compute, fewer gCO₂.

APIs: [Electricity Maps](https://www.electricitymaps.com/), [WattTime](https://www.watttime.org/) provide real-time regional carbon intensity.

Pattern:

```python
def scheduled_job():
    if current_gCO2_per_kWh() > threshold:
        reschedule_hours(1)
    else:
        run_batch()
```

Works beautifully for truly deferrable jobs. Can't help real-time serving.

## Cross-region routing

If your service runs in multiple regions:

- **Latency-aware**: user in Lahore → APAC region.
- **Cost-aware**: specialist cloud in one region cheaper than hyperscaler in another.
- **Carbon-aware**: route non-critical traffic to the greenest region.

A simple scheduler:

$$ \text{score}(region) = \alpha \cdot \text{latency} + \beta \cdot \$/\text{token} + \gamma \cdot gCO_2/\text{token} $$

Pick the lowest score. Weights $\alpha, \beta, \gamma$ encode your preferences.

## Workload class awareness

Not all requests deserve the same tier:

- Internal analytics queries: tolerate 10 s latency, want cheap. Route to small model + reserved instances.
- Customer chat: 1 s SLO, want frontier quality. Route to large model + hot capacity.
- Batch enrichment: 1-hour window, want cheapest. Spot instances, offline mode.

Multi-tier infrastructure encoded in the scheduler gives order-of-magnitude cost spread across workload types.

## Scale-to-zero for intermittent workloads

For workloads that run minutes-per-hour, KEDA (M8.3) can scale replicas to zero. When traffic arrives, a cold start (2–10 min) reinitializes.

Economic analysis:

- Warm pool: &#36;4,380/mo per 2 × H100 running 100 % of the time.
- Scale-to-zero + 10 % utilization: ~&#36;440/mo per 2 × H100.

Ten-fold cost reduction for genuinely bursty workloads. Only acceptable when cold-start latency is tolerable.

## Spot + preemption-tolerant scheduling

Spot GPUs are 60–90 % off, reclaimable at short notice. For a batch pipeline:

```python
def spot_job(task):
    try:
        run_on_spot(task)
    except SpotPreempted:
        checkpoint_state()
        reschedule(task, backend="spot_or_on_demand")
```

Requires:

- Checkpointable state (easier for batch than for streaming).
- Fast restart.
- Fallback to on-demand for jobs with tight deadlines.

## A combined scheduler sketch

For a mid-size serving fleet:

```python
def dispatch(request, slo):
    # 1. Classify
    workload = classify(request)
    # 2. Select tier
    tier = {
        "short_chat":  "cheap_small_model",
        "reasoning":   "expensive_large_model",
        "batch":       "spot_bulk_tier",
    }[workload]
    # 3. Carbon / cost / region decision
    region = pick_region(tier, slo.max_latency,
                        budget=slo.max_dollars,
                        carbon_cap=slo.max_gCO2)
    # 4. Route
    return route_to(tier, region, request)
```

Encoding all three objectives explicitly is the point. Even cheap approximations (static weights) beat latency-only routing on long-run cost.

## Pitfalls

- **Over-tuning scheduler weights.** The 20 % win from smart routing beats the 2 % from fine-tuning weights. Ship a simple version first.
- **Cascades without evaluation.** Small-model routing can degrade quality quietly. Monitor downstream quality metrics.
- **Spot for real-time.** Spot preemption during a streaming response is user-visible.
- **Carbon-aware for user-facing.** Users don't wait for the grid. Only defer batch.

<section class="faq">
<h2>FAQ</h2>
<h3>How much does cascade routing really save?</h3>
<p>3–5× on cost in published studies; depends heavily on your workload distribution.</p>
<h3>Can I rely on gCO₂/kWh APIs?</h3>
<p>Yes — Electricity Maps and WattTime are production-grade. Rate limits apply; cache.</p>
<h3>What about user-visible carbon claims?</h3>
<p>Marketing-grade carbon accounting requires auditable sources. Code-level estimates (codecarbon + grid intensity) are engineering-grade.</p>
<h3>Is this worth it for small fleets?</h3>
<p>Cascades: yes, even for 1–2 GPUs. Carbon-aware deferrals: only once batch workloads exist. Spot: only once you have fallbacks.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://www.electricitymaps.com/" target="_blank" rel="noreferrer">Electricity Maps — real-time grid carbon intensity</a></li>
<li><a href="https://www.watttime.org/" target="_blank" rel="noreferrer">WattTime — carbon-aware scheduling API</a></li>
<li><a href="https://codecarbon.io/" target="_blank" rel="noreferrer">codecarbon — energy and CO₂ measurement</a></li>
</ul>
</section>

## What's next

The capstone — [Chapter 7 — Designing a Cost- and Energy-Efficient Inference Stack](?module=m9&chapter=designing-cost-energy-stack) — ties everything from modules 1–9 into one coherent production design.
