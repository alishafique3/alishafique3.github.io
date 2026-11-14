<div class="chapter-meta">Module 2 · Chapter 5</div>

# GPU Interconnects and Communication

<p class="lede">The moment you need more than one GPU, you need to care about how GPUs talk to each other. For tensor parallelism, every forward pass triggers all-reduces that consume real wall time. For multi-node serving, network bandwidth becomes a primary constraint on which parallelism strategies make sense. This chapter maps the interconnect fabric.</p>

## Three tiers of interconnect

1. **Intra-node** — GPUs inside the same server, connected by PCIe or NVLink.
2. **Intra-rack** — Nodes within a chassis sharing a backplane (NVLink Switch System, or a top-of-rack InfiniBand switch).
3. **Inter-node** — Between servers, over InfiniBand or Ethernet fabric.

## Intra-node fabric

| Fabric | Bandwidth (per direction) | Notes |
|---|---|---|
| PCIe 5.0 x16 | 63 GB/s | Baseline on consumer boards & some cheap servers |
| PCIe 6.0 x16 | 128 GB/s | Emerging on 2026 platforms |
| NVLink 3 (A100) | 300 GB/s aggregate | 12 links × 25 GB/s |
| NVLink 4 (H100) | 450 GB/s aggregate | 18 links × 25 GB/s |
| NVLink 5 (B200) | 900 GB/s aggregate | 18 links × 50 GB/s |

The practical gap: NVLink is ~5–15 × PCIe. For any TP-sharded model, you want NVLink. PCIe-only multi-GPU is viable only for pipeline or data parallelism, where cross-GPU traffic is sparse.

## NVSwitch and the NVLink Switch System

Inside an HGX H100 / H200 / B200 chassis, an **NVSwitch** fabric gives every GPU full-bandwidth access to every other GPU — no topology surprises, no bottleneck links. This is what makes an 8-GPU server behave like one big GPU for TP purposes.

NVIDIA's **NVLink Switch System** extends this to 32 GPUs in a 4-chassis rack at full NVLink bandwidth — enabling TP × EP configurations that were previously multi-node. B200-based NVL72 systems extend further to 72 GPUs in one coherent fabric.

## Inter-node fabric

| Fabric | Bandwidth (per link, per direction) | Notes |
|---|---|---|
| InfiniBand NDR (400 Gbps) | 50 GB/s | Standard for HPC clusters |
| InfiniBand XDR (800 Gbps) | 100 GB/s | Next generation, 2025–2026 deployments |
| Ethernet 400 Gbps (RoCE) | 50 GB/s | Hyperscaler default |
| Ethernet 800 Gbps (RoCE) | 100 GB/s | Emerging |

Note: inter-node is *10 × slower* than intra-NVLink and ~2 × slower than PCIe. Any parallelism that crosses nodes pays this tax.

## Which operations hit the fabric

- **Tensor parallelism (TP)** — AllReduce on activations after every attention and FFN block. Latency-sensitive. NVLink is non-negotiable.
- **Pipeline parallelism (PP)** — Point-to-point sends between adjacent stages. Modest bandwidth, higher latency. Tolerates inter-node fabric.
- **Expert parallelism (EP)** — All-to-all for MoE routing. Bandwidth-hungry; benefits enormously from NVLink Switch.
- **Prefill-decode disaggregation** — KV cache transfer between prefill and decode workers. Bandwidth matters, but it's a one-time move per request. InfiniBand RDMA is standard.

## Rough allreduce math

For a TP-sharded Llama 3 70B layer with hidden dim 8192 at batch 16, seq 2048, bf16:

$$ \text{allreduce bytes per block} \approx B \cdot s \cdot d \cdot 2 = 16 \cdot 2048 \cdot 8192 \cdot 2 \approx 536 \text{ MB} $$

At NVLink 4 bandwidth (450 GB/s), one allreduce takes ~1.2 ms. Over 80 layers × 2 blocks (attn + FFN) = 160 allreduces = ~190 ms of pure interconnect time per forward pass. If that forward is prefill of 2k tokens taking ~70 ms of compute, communication is 2–3 × the compute — the fabric is on the critical path.

Overlap-with-compute techniques (double buffering, flash-decoding style async) hide much of this, which is why modern engines matter so much.

## Pitfalls

- **Buying "8 × H100" boxes with PCIe only.** Looks the same in a datasheet; roughly half the serving throughput for TP workloads.
- **Assuming NIC = application bandwidth.** 400 Gbps on the datasheet becomes 30–40 GB/s sustained after protocol overhead and GPU-to-NIC path.
- **Multi-node TP without a plan.** TP across nodes over InfiniBand usually costs more in communication than it saves in memory. Use PP across nodes and TP within nodes.

<section class="faq">
<h2>FAQ</h2>
<h3>Can I do TP over PCIe?</h3>
<p>Yes, but NVLink is 5–15× faster. PCIe-only TP is bearable for 2 GPUs on a small model; above that, latency balloons.</p>
<h3>What is RoCE?</h3>
<p>RDMA over Converged Ethernet — RDMA-like performance over standard 100/400/800 Gbps Ethernet. Hyperscalers' preferred fabric.</p>
<h3>Do I need InfiniBand for 2-node serving?</h3>
<p>If you cross nodes with TP, generally yes. PP across nodes is much more forgiving of Ethernet.</p>
<h3>What does NVLink 5 give me over NVLink 4?</h3>
<p>Double the per-link bandwidth (50 vs 25 GB/s per link per direction), and the NVLink Switch System scales up to 72 GPUs in a single coherent fabric on B200 NVL72.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://www.nvidia.com/en-us/data-center/nvlink/" target="_blank" rel="noreferrer">NVIDIA NVLink and NVSwitch overview</a></li>
<li><a href="https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html" target="_blank" rel="noreferrer">NCCL — Collectives library documentation</a></li>
<li><a href="https://network.nvidia.com/pdf/whitepapers/IB-NDR-800G.pdf" target="_blank" rel="noreferrer">InfiniBand NDR / XDR specifications (Mellanox)</a></li>
</ul>
</section>

## What's next

With the fabric in mind, [Chapter 6 — Multi-GPU Inference Fundamentals](?module=m2&chapter=multi-gpu-fundamentals) walks through how models are actually sharded across GPUs in practice.
