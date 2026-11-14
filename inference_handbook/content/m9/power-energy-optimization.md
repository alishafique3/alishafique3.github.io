<div class="chapter-meta">Module 9 · Chapter 5</div>

# GPU Power and Energy Optimization

<p class="lede">An H100 draws 700 W at load. A B200 draws 1,000 W. Scaled to a 1,000-GPU fleet, you're pulling 1 MW — before cooling. For on-prem or utility-conscious operators, power is a first-class constraint. This chapter covers the knobs that reduce energy per token without proportional throughput loss.</p>

## The energy identity

$$ \text{energy per token} = \frac{\text{avg power (W)} \times \text{wall time (s)}}{\text{tokens produced}} $$

Three ways to lower this:

1. Reduce average power (per-GPU power management).
2. Reduce wall time (better engine, bigger batch).
3. Produce more tokens in the same time (speculative, batching).

(2) and (3) are covered in modules 6 and 7. This chapter is (1): direct power management.

## Power capping

Every GPU has a configurable power limit — the maximum it will draw. Below the TDP (H100 default 700 W), the GPU clocks down to stay within budget. Lower power → slower execution, but typically not proportionally slower.

Set a power cap:

```bash
# H100 default is 700W. Cap at 500W:
sudo nvidia-smi -i 0 -pl 500
```

Published results: dropping H100 from 700 W → 500 W (28 % power reduction) typically costs 10–15 % throughput — a net energy-per-token improvement of 10–15 %. The exact number depends on workload (compute-bound workloads scale more linearly with power than memory-bound).

## When power capping wins

- **On-prem with real power/cooling constraints.** You can't just add more GPUs; power is the bottleneck.
- **Carbon-conscious serving.** &#36;/token goes up slightly; CO₂/token goes down meaningfully.
- **Fleet density.** More GPUs per rack when each draws less.

When it loses:

- **Cloud pricing is time-based.** Longer wall time → more &#36;/token. Power capping *increases* your bill in the cloud.
- **Latency-sensitive workloads.** The TPOT penalty hits user experience.

Rule: power cap on-prem; don't power cap in the cloud (unless you're the cloud provider).

## Clock management

GPUs expose fine-grained clock control:

```bash
# Lock SM clock to 1200 MHz (below default ~1410):
sudo nvidia-smi -lgc 1200

# Lock memory clock:
sudo nvidia-smi -lmc 2619
```

Lower clock → lower power. Similar trade-off to power capping, but more granular. Rarely worth the operational complexity in modern fleets — power capping is usually sufficient.

## MIG (Multi-Instance GPU) for small models

A single H100 can be partitioned into up to 7 isolated instances, each with its own SM allocation and HBM slice. For serving several small models simultaneously on one GPU, MIG lets each get guaranteed resources.

Energy benefit: a MIG-partitioned H100 draws 500–600 W at full utilization vs. 700 W, and you serve multiple models. Net energy per model-token is lower than dedicating full H100s to each model.

Cost: setup complexity, less flexible scheduling.

## Utilization-aware scheduling

The biggest waste is idle GPUs drawing 60–100 W at 0 % utilization (H100 idle). Mitigations:

- **Power down on scale-down.** Kubernetes can cordon and power off idle GPU nodes. Takes minutes; not for bursty workloads.
- **Consolidate traffic to fewer GPUs.** Better load balancing keeps each GPU busy.
- **Batch jobs when interactive dies down.** Offline inference fills the valleys in daily traffic curves.

A fleet that runs at 70 % average utilization (traffic-shaped) uses dramatically less energy per token than one at 40 % with the rest idle.

## Measuring with pynvml

Covered in M9.4. A direct measurement sketch:

```python
import pynvml, time, threading

class PowerMeter:
    def __init__(self, interval_ms=10):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.interval = interval_ms / 1000
        self.samples = []
        self._stop = threading.Event()

    def __enter__(self):
        self.samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop)
        self._thread.start()
        return self

    def _loop(self):
        t0 = time.perf_counter()
        while not self._stop.is_set():
            t = time.perf_counter() - t0
            w = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000
            self.samples.append((t, w))
            time.sleep(self.interval)

    def __exit__(self, *args):
        self._stop.set(); self._thread.join()
        self.energy_J = sum(
            0.5 * (self.samples[i][1] + self.samples[i-1][1]) *
            (self.samples[i][0] - self.samples[i-1][0])
            for i in range(1, len(self.samples)))
```

Wrap your inference call: `with PowerMeter() as m: inference_call()`. Measure baseline (idle) separately; subtract.

## A realistic optimization example

Llama 3.1 70B on 2 × H100, chat workload, baseline:

- Throughput: ~10 RPS at standard SLO.
- Avg power per GPU: 620 W.
- Energy per token: ~1.8 J.

After optimizations:

| Change | Throughput | Avg power | J/token |
|---|---|---|---|
| Baseline (BF16, static batch) | 10 RPS | 620 W | 1.8 |
| + Continuous batching | 14 RPS | 640 W | 1.3 |
| + FP8 weights + KV | 18 RPS | 600 W | 0.95 |
| + Power cap 500 W | 16 RPS | 490 W | 0.87 |
| + Prefix caching (40 % hit) | 22 RPS | 510 W | 0.66 |

Compounding ~2.7× energy-per-token improvement from engineering, ~10 % more from power capping. Power capping is the tail, not the head.

## Pitfalls

- **Power capping in the cloud.** You pay for time; longer time = more money, even if less power.
- **Ignoring idle power.** 10 idle GPUs at 60 W each = 600 W burned 24×7 = 5.2 MWh/year.
- **Benchmarking at peak but deploying at low batch.** Energy per token is batch-dependent. Measure at your production batch.
- **Confusing power and energy.** Power is instantaneous (W); energy is integrated (J). Only energy scales with work done.

<section class="faq">
<h2>FAQ</h2>
<h3>Does FP8 really save energy?</h3>
<p>Yes — half the HBM reads per forward pass. Memory-bound decode especially benefits. Typical: 20–30 % J/token reduction.</p>
<h3>Can I reduce HBM power separately?</h3>
<p>Yes with memory clock control, but gains are small. Compute clocks matter more.</p>
<h3>What about carbon-aware deferrals?</h3>
<p>Covered in Chapter 6. Scheduling batch work to greener grid hours.</p>
<h3>Do newer GPUs have better perf/watt?</h3>
<p>Yes — Blackwell delivers roughly 25 % better perf/W than Hopper for comparable workloads, with FP4 tipping it further.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://docs.nvidia.com/deploy/nvml-api/" target="_blank" rel="noreferrer">NVIDIA NVML API — power/clock controls</a></li>
<li><a href="https://docs.nvidia.com/datacenter/tesla/mig-user-guide/" target="_blank" rel="noreferrer">NVIDIA MIG User Guide</a></li>
<li><a href="https://codecarbon.io/" target="_blank" rel="noreferrer">codecarbon — energy and CO₂ measurement</a></li>
</ul>
</section>

## What's next

[Chapter 6](?module=m9&chapter=cost-energy-scheduling) extends the ideas to scheduling: routing requests and jobs to minimize $ and J under SLO.
