<div class="chapter-meta">Module 9 · Chapter 4</div>

# Energy Consumption and Tokens-per-Joule Metrics

<p class="lede">A GPU converts electricity into tokens. Dollars are one way to count that; joules are the other. For datacenter operators, the two are nearly the same — power + cooling is the single biggest OPEX line after the GPUs themselves. This chapter defines energy-per-token and tokens-per-joule, shows the asymmetry between prefill and decode, and connects the measurement to the dashboards you already have.</p>

## The two metrics

Energy is power integrated over time: $E = \int P(t) \, dt$. For LLM inference, you integrate the GPU's measured power over the time the request runs.

Per-token:

$$ \text{J/token} \;=\; \frac{E_{\text{work}}}{n_{\text{tokens produced}}} $$

Reciprocal:

$$ \text{tokens/J} \;=\; \frac{n_{\text{tokens produced}}}{E_{\text{work}}} $$

$E_{\text{work}}$ is energy *above idle* — you subtract the baseline power the GPU would have burned doing nothing to isolate the useful work.

## Prefill and decode have very different energy profiles

- **Prefill** is compute-bound — the GPU pegs near TDP (e.g., 700 W on H100). Short in duration, high in power.
- **Decode** is memory-bound — the GPU runs at a sustained lower power (~400–500 W on H100 at bs = 1). Long in duration, lower in power.

Measured J/output-token on H100 serving Llama-3 8B at bs = 1 is typically 1–2 J/token. At batch 16 under continuous batching, the same GPU produces ~5–8× more tokens per second while its power only rises ~30 %, so J/token drops to ~0.2–0.3. **Batching is the single biggest energy lever.**

## Measurement — pynvml in 15 lines

NVIDIA's NVML library exposes per-millisecond power via `nvmlDeviceGetPowerUsage`. A background thread sampling every 10 ms and integrating with the trapezoid rule gives you joules with reasonable accuracy.

```python
import threading, time, pynvml

class PowerMeter:
    def __init__(self):
        pynvml.nvmlInit()
        self.h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.samples = []  # (t_rel, watts)

    def __enter__(self):
        self.samples.clear()
        self._stop = threading.Event()
        self._t0 = time.perf_counter()
        self._th = threading.Thread(
            target=lambda: self._loop(), daemon=True)
        self._th.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            t = time.perf_counter() - self._t0
            w = pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000  # mW -> W
            self.samples.append((t, w))
            time.sleep(0.01)

    def __exit__(self, *_):
        self._stop.set(); self._th.join()
        # Trapezoid integration
        self.energy_J = sum(
            0.5*(self.samples[i][1]+self.samples[i-1][1]) *
            (self.samples[i][0]-self.samples[i-1][0])
            for i in range(1, len(self.samples)))
```

Wrap your inference call in `with PowerMeter() as m:` and you've measured work energy.

## From J/token to &#36;/1M tokens

At grid price $p$ (&#36;/kWh), a token costing $J$ joules costs:

$$ \$/\text{token} = \frac{J \times p}{3.6 \times 10^6} $$

At J = 0.3 and $p = \$0.10$/kWh: $8.3 \times 10^{-9}$ &#36;/token ≈ &#36;0.008 per 1M tokens of *electricity*. Note: this is *just* electricity — not GPU rental, cooling, or amortized hardware cost. Grid power is usually 5–20 % of total token cost on commercial infrastructure, more for on-prem with low cooling PUE.

## What the metric does and does not include

- GPU core + HBM power (what NVML reports) — **included**.
- CPU + DRAM power — **not included** (use Intel RAPL or [codecarbon](https://codecarbon.io)).
- Datacenter cooling overhead — **not included** (multiply by PUE, typically 1.1–1.5, to compare to grid electricity).
- Inter-GPU NVLink power — **not included** (small on single-node, non-trivial multi-node).

## Pitfalls

- **Not subtracting idle.** A T4 at idle draws ~15 W; failing to subtract it inflates per-token energy at low batch.
- **Reporting instantaneous power as "energy".** Power is watts; energy is joules. $E = P \times t$.
- **Measuring at the wrong batch size.** J/token at bs = 1 vs. bs = 32 can be 10× apart. Report at your production batch.

<section class="faq">
<h2>FAQ</h2>
<h3>Why subtract idle power?</h3>
<p>Idle power is paid whether you run inference or not; it isolates the "work" energy that scales with your workload.</p>
<h3>Does FP8 save energy?</h3>
<p>Usually yes — half the bytes moved per forward, smaller kernel durations, lower average power. FP4 extends the trend on Blackwell.</p>
<h3>What's a good tokens-per-joule target?</h3>
<p>H100 serving Llama-3 8B at batch 16+ should deliver 5–10 tokens/J after idle subtraction. Your mileage depends on batch, prompt/output length, and quantization.</p>
<h3>Is pynvml accurate?</h3>
<p>Within ~1 W on H100 at 10 ms sampling. Absolute accuracy is better than ±5 %; relative comparisons (batch sweep, quant-on vs. quant-off) are robust.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://docs.nvidia.com/datacenter/dcgm/" target="_blank" rel="noreferrer">NVIDIA DCGM — GPU power telemetry</a></li>
<li><a href="https://codecarbon.io/" target="_blank" rel="noreferrer">codecarbon — CPU + GPU energy &amp; CO₂</a></li>
<li><a href="https://github.com/gpuopenanalytics/pynvml" target="_blank" rel="noreferrer">pynvml — Python bindings for NVML</a></li>
<li><a href="https://github.com/neuralwatt/ai_energy_benchmarks" target="_blank" rel="noreferrer">neuralwatt/ai_energy_benchmarks — framework for energy measurements</a></li>
</ul>
</section>

## What's next

Energy-aware scheduling ([Chapter 6](?module=m9&chapter=cost-energy-scheduling)) uses these numbers as a routing signal. For a hands-on walkthrough with code and plots, see Part 8 of the LLM Infra from Scratch Colab series.
