<div class="chapter-meta">Module 2 · Chapter 7</div>

# Understanding GPU Performance Bottlenecks

<p class="lede">"Why is this slow?" has exactly three answers for a GPU workload: it is compute-bound, memory-bound, or communication-bound. Occasionally a fourth — overhead-bound — when launches and host-device copies dominate. The roofline model is the diagnostic tool. This chapter teaches it as a workflow, not a graph.</p>

## The roofline in one line

A kernel's achievable throughput is capped by the smaller of:

- Peak compute (flops/sec)
- Peak bandwidth × arithmetic intensity (bytes/sec × flops/byte)

Plot "achieved flops/sec" vs. "arithmetic intensity" with the hardware peak on each axis, and you get a roof. Every kernel is a point under the roof.

## The four bottleneck classes

| Class | Symptom | Example in LLM serving |
|---|---|---|
| Compute-bound | Tensor cores pegged, HBM has slack | Long-prompt prefill |
| Memory-bound | HBM pegged, tensor cores idle | Decode at small batch |
| Communication-bound | Interconnect pegged, compute waits | TP across slow fabric |
| Overhead-bound | GPU shows gaps between kernels | Many tiny ops, Python loop overhead |

## A diagnostic workflow

1. **Separate the phases.** Measure TTFT and TPOT independently (M1.7). They have opposite bottlenecks.
2. **Watch GPU utilization during each.** `nvidia-smi dmon` and `dcgmi dmon` show SM utilization and HBM bandwidth. If SM % is high and HBM % is low → compute-bound. If HBM % is high and SM % is middling → memory-bound.
3. **Profile with Nsight Systems.** Look for gaps between kernels (overhead-bound) or long NCCL regions (communication-bound).
4. **Match the symptom to the fix.** See the table below.

## Fixes by class

| Bottleneck | Fix |
|---|---|
| Compute-bound prefill | Lower precision (fp8), FlashAttention, prefill chunking to overlap with decode |
| Memory-bound decode | Bigger batch (continuous batching), quantize weights, quantize KV, speculative decoding, H200 instead of H100 |
| Communication-bound TP | Fewer TP shards, keep TP within NVLink, overlap comm with compute |
| Overhead-bound | CUDA graphs, kernel fusion, torch.compile, bigger kernels (less of them) |

## A worked example — "my decode is slow"

Scenario: Llama 3 70B on 2×H100, TP=2, bs=1, TPOT = 50 ms. Expected floor ~21 ms. Diagnose:

1. Separate: TPOT, so decode. Not prefill.
2. GPU utilization during decode: SM % = 10, HBM % = 85 → memory-bound.
3. Matches the theory — bs=1 decode is textbook memory-bound.
4. Fix options: enable continuous batching with target batch 16 (brings HBM % to 95, SM % up, TPOT at batch amortizes down), or quantize weights to fp8 (halves HBM reads, ~2× TPOT improvement), or enable EAGLE-3 speculative (2–3× effective token rate).

The point: you did not guess. You measured, mapped to a class, and the fix was the one the class prescribes.

## What peak numbers you need in your head

- H100: 1,979 TF BF16 / 3,958 TF FP8 / 3.35 TB/s HBM → ridge ~590 (bf16) or ~1180 (fp8) flops/byte
- H200: same compute, 4.8 TB/s HBM → ridge ~412 (bf16) or ~824 (fp8) — decode-friendly
- B200: 4,500 TF BF16 / 9,000 TF FP8 / 8.0 TB/s → ridge ~563 (bf16) or ~1125 (fp8)

Lower ridge = lenient on memory-bound kernels. H200's lower ridge is exactly why it beats H100 at decode despite identical TFLOPs.

## Pitfalls

- **Blaming the engine for a physics problem.** If you are memory-bound at bs=1 on bf16, no engine will make you compute-bound.
- **Reading GPU utilization as "work being done".** The `nvidia-smi` GPU % is any kernel running, not tensor-core utilization. Use Nsight or DCGM.
- **Optimizing the wrong phase.** If decode is 90% of wall time, prefill speedups are cosmetic.

<section class="faq">
<h2>FAQ</h2>
<h3>My GPU shows 100% utilization but tokens/s is low — why?</h3>
<p><code>nvidia-smi</code>'s GPU % counts any kernel running, not tensor-core utilization. Use <code>dcgmi dmon</code> or Nsight to see tensor-core utilization and HBM % separately.</p>
<h3>How do I know I'm communication-bound?</h3>
<p>Nsight Systems shows long NCCL kernels in the timeline, and SM % drops during the cross-GPU phases.</p>
<h3>Is overhead-bound common?</h3>
<p>It is when you have many small kernels (classic PyTorch eager with many ops). CUDA Graphs and <code>torch.compile</code> collapse the launch overhead by batching kernel launches.</p>
<h3>Does the ridge point move with precision?</h3>
<p>Yes — FP8 doubles peak compute at the same bandwidth, so the ridge moves right. A kernel that was compute-bound at bf16 may become memory-bound at fp8 on the same hardware.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf" target="_blank" rel="noreferrer">Williams et al. — The Roofline Model</a></li>
<li><a href="https://developer.nvidia.com/nsight-systems" target="_blank" rel="noreferrer">NVIDIA Nsight Systems</a></li>
<li><a href="https://horace.io/brrr_intro.html" target="_blank" rel="noreferrer">Horace He — Making Deep Learning Go Brrrr</a></li>
</ul>
</section>

## What's next

Module 2 is done. The next module — [Planning Your Deployment](?module=m3&chapter=deployment-architectures) — takes these fundamentals and turns them into GPU selection, framework choice, and capacity math.
