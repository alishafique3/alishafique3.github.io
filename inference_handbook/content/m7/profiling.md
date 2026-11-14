<div class="chapter-meta">Module 7 · Chapter 6</div>

# GPU Profiling and Kernel Performance Analysis

<p class="lede">Every optimization debate — "FlashAttention-3 really helps", "torch.compile is worth it", "this kernel is memory-bound" — is settled by a profiler. Three tools matter for LLM inference work: <strong>Nsight Systems</strong> (timeline view), <strong>Nsight Compute</strong> (kernel deep dive), and <strong>DCGM</strong> (fleet-level telemetry). This chapter teaches the workflow, not the UI.</p>

## What each tool is for

| Tool | Scope | When to use |
|---|---|---|
| nvidia-smi | High-level | "Is the GPU alive? How much memory used?" |
| DCGM / dcgmi dmon | Rich telemetry over time | SM %, HBM %, power — monitor serving |
| Nsight Systems | Timeline of kernels + CPU + NCCL | "What ran when? Where are the gaps?" |
| Nsight Compute | Deep per-kernel analysis | "Why is this specific kernel slow?" |
| PyTorch Profiler | Python → kernel mapping | "Which torch op launched this kernel?" |

Most LLM work lives in Nsight Systems and PyTorch Profiler.

## Nsight Systems — the timeline view

Captures all kernel launches, CUDA API calls, NCCL collectives, and (optionally) CPU activity on a timeline. You look for:

- **Gaps between kernels.** Launch overhead — fix with CUDA Graphs (M7.5).
- **Tall NCCL regions.** Communication-bound — look at parallelism layout (M2.6).
- **One kernel dominating.** That's your optimization target.

Basic recording:

```bash
nsys profile -o trace -t cuda,nvtx \
    python my_benchmark.py

# Open trace.nsys-rep in the Nsight Systems GUI
```

Add `nvtx` ranges in your Python code (`torch.cuda.nvtx.range_push("prefill")`) to label regions.

## PyTorch Profiler — Python-friendly

For when you want the trace in terms of torch ops rather than raw kernels:

```python
import torch

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
) as prof:
    model.generate(prompt, max_new_tokens=100)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

Shows which torch op consumed the most GPU time. Good first look before diving into Nsight.

## Nsight Compute — kernel deep dive

When you know *which* kernel is slow and need to know *why*:

```bash
ncu -k fa3_forward_kernel \
    --set full \
    -o kernel_profile \
    python my_benchmark.py
```

Reports per-kernel: SM throughput, HBM throughput, roofline position, stall reasons. Overkill for most infra work; essential if you're writing CUDA.

## DCGM — fleet telemetry

For production monitoring, not debugging. `dcgmi dmon` streams real-time GPU metrics:

```bash
dcgmi dmon -e 203,252,253,155,156   # SM util, tensor core active, HBM util, power, temp
```

Export to Prometheus via DCGM Exporter for Grafana dashboards. Covered more fully in M8.4 (Observability).

## A realistic diagnostic workflow

Scenario: "Our Llama 3 70B serving has P99 TPOT = 45 ms; expected 25 ms. Why?"

1. **First look — nvidia-smi dmon.** SM utilization is 12 %, HBM is 30 %. That's low — the GPU is underfed.
2. **Hypothesize.** Low SM + low HBM points to launch overhead or scheduling, not kernel inefficiency.
3. **Nsight Systems profile a single request.** Timeline shows many tiny kernels with 5 μs gaps between each. Pattern = launch overhead.
4. **Fix.** Enable CUDA Graphs. In vLLM this is `--enforce-eager=False` (default is already False in recent versions; verify).
5. **Re-measure.** TPOT drops to 27 ms. Problem solved.

Contrast with:

Scenario: "TPOT = 45 ms, but SM utilization is 92 %."

Diagnosis: not overhead-bound. Compute-bound. Fix: lower precision, bigger tiles, better kernel.

The utilization pattern is diagnostic.

## Classifying kernels with the roofline

Nsight Compute can map each kernel onto the roofline. Visual bucket:

- **Top of roof, right side** — compute-bound. Full saturation.
- **Diagonal line** — memory-bound at the attained arithmetic intensity.
- **Below the line** — overhead-bound. GPU not fully busy.

Memory-bound kernels want precision reduction, bigger batch, or better fusion. Compute-bound kernels want better math — different kernel, different precision. Overhead-bound wants fewer launches.

## Pitfalls

- **Profiling in debug mode.** Debug builds have 2–10× overhead. Always profile optimized code.
- **Profiling one iteration.** First iteration includes compile and warm-up. Capture after ~10 warm iterations.
- **Trusting nvidia-smi GPU %.** Rough. Use `dcgmi dmon` or Nsight for tensor-core utilization vs. SM utilization distinction.
- **Chasing per-kernel gains at the expense of system throughput.** 10 % speedup on a kernel that takes 2 % of time is 0.2 % system win. Priorities.

<section class="faq">
<h2>FAQ</h2>
<h3>Do I need to install Nsight manually?</h3>
<p>It ships with recent CUDA toolkits. Enable via <code>nsys</code> and <code>ncu</code> commands.</p>
<h3>Can I profile a production server?</h3>
<p>Yes with caveats — Nsight overhead is typically &lt; 5 %. Don't record long traces on prod; capture a 10-second window.</p>
<h3>What about Apple Silicon / AMD profilers?</h3>
<p>Apple: Instruments and Metal System Trace. AMD: ROCm Profiler (rocprof). Comparable but LLM tooling is more mature on NVIDIA.</p>
<h3>How do I know if NCCL is my bottleneck?</h3>
<p>Nsight Systems shows NCCL kernels on their own row. Tall and overlapping with compute gaps = you're communication-bound.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://developer.nvidia.com/nsight-systems" target="_blank" rel="noreferrer">NVIDIA Nsight Systems</a></li>
<li><a href="https://developer.nvidia.com/nsight-compute" target="_blank" rel="noreferrer">NVIDIA Nsight Compute</a></li>
<li><a href="https://pytorch.org/docs/stable/profiler.html" target="_blank" rel="noreferrer">PyTorch Profiler documentation</a></li>
<li><a href="https://docs.nvidia.com/datacenter/dcgm/" target="_blank" rel="noreferrer">NVIDIA DCGM — Data Center GPU Manager</a></li>
</ul>
</section>

## What's next

Module 7 is done. [Module 8 — Production Operations and Load Testing](?module=m8&chapter=production-infra) covers the layer above kernels: how you run the whole serving stack in production.
