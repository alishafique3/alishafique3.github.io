/* ==========================================================================
   LLM Inference Engineering Handbook — navigation + on-page TOC
   Zero deps. 9 modules × 61 chapters. Works on GitHub Pages as plain static.
   ========================================================================== */

(function () {
  "use strict";

  // Flat list of every page. The sidebar below is built from MODULES.
  // Each page id follows "<module-slug>/<chapter-slug>" so data-page attributes
  // can be looked up directly from URL params.
  const PAGES = [
    // ----- Module 1 — LLM and Inference Fundamentals -----
    { id: "m1/intro",              module: "m1", chapter: "intro",              title: "Introduction to LLM Inference Engineering" },
    { id: "m1/how-llms-work",      module: "m1", chapter: "how-llms-work",      title: "How Large Language Models Work" },
    { id: "m1/tokens",             module: "m1", chapter: "tokens",             title: "Understanding Tokens and Tokenization" },
    { id: "m1/training-vs-inference", module: "m1", chapter: "training-vs-inference", title: "Training vs. Inference" },
    { id: "m1/prefill-and-decode", module: "m1", chapter: "prefill-and-decode", title: "Anatomy of LLM Inference: Prefill and Decode" },
    { id: "m1/inference-parameters", module: "m1", chapter: "inference-parameters", title: "Understanding LLM Inference Parameters" },
    { id: "m1/performance-metrics", module: "m1", chapter: "performance-metrics", title: "Key Performance Metrics for LLM Inference" },

    // ----- Module 2 — GPU Fundamentals -----
    { id: "m2/where-inference-runs", module: "m2", chapter: "where-inference-runs", title: "Where Does LLM Inference Run?" },
    { id: "m2/gpu-architecture",   module: "m2", chapter: "gpu-architecture",   title: "GPU Architecture Fundamentals" },
    { id: "m2/compute-memory-hierarchy", module: "m2", chapter: "compute-memory-hierarchy", title: "GPU Compute and Memory Hierarchy" },
    { id: "m2/gpu-memory-for-inference", module: "m2", chapter: "gpu-memory-for-inference", title: "Understanding GPU Memory for LLM Inference" },
    { id: "m2/gpu-interconnects",  module: "m2", chapter: "gpu-interconnects",  title: "GPU Interconnects and Communication" },
    { id: "m2/multi-gpu-fundamentals", module: "m2", chapter: "multi-gpu-fundamentals", title: "Multi-GPU Inference Fundamentals" },
    { id: "m2/performance-bottlenecks", module: "m2", chapter: "performance-bottlenecks", title: "Understanding GPU Performance Bottlenecks" },

    // ----- Module 3 — Planning Your Deployment -----
    { id: "m3/deployment-architectures", module: "m3", chapter: "deployment-architectures", title: "Understanding LLM Deployment Architectures" },
    { id: "m3/choosing-the-llm",   module: "m3", chapter: "choosing-the-llm",   title: "Choosing the Right LLM — Selection and Configuration" },
    { id: "m3/gpu-selection-memory", module: "m3", chapter: "gpu-selection-memory", title: "GPU Selection and Memory Sizing" },
    { id: "m3/capacity-planning",  module: "m3", chapter: "capacity-planning",  title: "Capacity Planning Fundamentals" },
    { id: "m3/inference-framework", module: "m3", chapter: "inference-framework", title: "Choosing the Right Inference Framework" },
    { id: "m3/serving-first-llm",  module: "m3", chapter: "serving-first-llm",  title: "Serving Your First LLM" },
    { id: "m3/cloud-hybrid-onprem", module: "m3", chapter: "cloud-hybrid-onprem", title: "Cloud, Hybrid, and On-Premises Deployments" },

    // ----- Module 4 — Model Optimization -----
    { id: "m4/intro-model-optimization", module: "m4", chapter: "intro-model-optimization", title: "Introduction to Model Optimization" },
    { id: "m4/quantization-fundamentals", module: "m4", chapter: "quantization-fundamentals", title: "LLM Quantization Fundamentals" },
    { id: "m4/quantization-methods", module: "m4", chapter: "quantization-methods", title: "Quantization Methods and Frameworks" },
    { id: "m4/distillation",       module: "m4", chapter: "distillation",       title: "Model Distillation" },
    { id: "m4/fine-tuning",        module: "m4", chapter: "fine-tuning",        title: "Fine-Tuning Models for Inference" },
    { id: "m4/evaluating-optimized", module: "m4", chapter: "evaluating-optimized", title: "Evaluating Optimized Models" },

    // ----- Module 5 — Model Interaction Choices -----
    { id: "m5/inference-apis",     module: "m5", chapter: "inference-apis",     title: "Understanding LLM Inference APIs" },
    { id: "m5/openai-anthropic-apis", module: "m5", chapter: "openai-anthropic-apis", title: "OpenAI- and Anthropic-Compatible APIs" },
    { id: "m5/streaming-vs-buffered", module: "m5", chapter: "streaming-vs-buffered", title: "Streaming vs. Buffered Inference" },
    { id: "m5/prompt-engineering", module: "m5", chapter: "prompt-engineering", title: "Prompt Engineering and Context Management" },
    { id: "m5/structured-outputs", module: "m5", chapter: "structured-outputs", title: "Structured Outputs and Constrained Generation" },
    { id: "m5/function-calling-mcp", module: "m5", chapter: "function-calling-mcp", title: "Function Calling, Tool Use, and MCP" },
    { id: "m5/agent-harnesses",    module: "m5", chapter: "agent-harnesses",    title: "Agent Harnesses and Multi-Step Workflows" },

    // ----- Module 6 — Inference Infrastructure Optimization -----
    { id: "m6/engine-anatomy",     module: "m6", chapter: "engine-anatomy",     title: "Anatomy of an LLM Inference Engine" },
    { id: "m6/batching-scheduling", module: "m6", chapter: "batching-scheduling", title: "Batching and Scheduling" },
    { id: "m6/kv-cache-paged",     module: "m6", chapter: "kv-cache-paged",     title: "KV Cache Management and PagedAttention" },
    { id: "m6/kv-cache-optimization", module: "m6", chapter: "kv-cache-optimization", title: "KV Cache Optimization — Quantization, Compression, Offloading" },
    { id: "m6/decoding-optimization", module: "m6", chapter: "decoding-optimization", title: "Decoding Optimization — Speculative and Long-Context" },
    { id: "m6/distributed-parallelism", module: "m6", chapter: "distributed-parallelism", title: "Distributed Parallelism" },
    { id: "m6/advanced-serving",   module: "m6", chapter: "advanced-serving",   title: "Advanced Serving — Disaggregation, Routing, Offline Batch" },

    // ----- Module 7 — GPU Kernel Optimization -----
    { id: "m7/intro-kernel",       module: "m7", chapter: "intro-kernel",       title: "Introduction to GPU Kernel Optimization" },
    { id: "m7/compute-kernels",    module: "m7", chapter: "compute-kernels",    title: "Understanding LLM Compute Kernels" },
    { id: "m7/cuda-triton",        module: "m7", chapter: "cuda-triton",        title: "CUDA and Triton for LLM Inference" },
    { id: "m7/flashattention",     module: "m7", chapter: "flashattention",     title: "FlashAttention and Efficient Attention Kernels" },
    { id: "m7/kernel-fusion",      module: "m7", chapter: "kernel-fusion",      title: "Kernel Fusion and Low-Level Optimization" },
    { id: "m7/profiling",          module: "m7", chapter: "profiling",          title: "GPU Profiling and Kernel Performance Analysis" },

    // ----- Module 8 — Production Operations and Load Testing -----
    { id: "m8/production-infra",   module: "m8", chapter: "production-infra",   title: "Understanding Production LLM Inference Infrastructure" },
    { id: "m8/containerization-k8s", module: "m8", chapter: "containerization-k8s", title: "Containerization and Kubernetes Deployment" },
    { id: "m8/load-balancing-autoscaling", module: "m8", chapter: "load-balancing-autoscaling", title: "Load Balancing, Request Routing, and Autoscaling" },
    { id: "m8/observability",      module: "m8", chapter: "observability",      title: "LLM Observability and Monitoring" },
    { id: "m8/load-testing-slo",   module: "m8", chapter: "load-testing-slo",   title: "Benchmarking, Load Testing, and SLO Engineering" },
    { id: "m8/multi-model-multi-cloud", module: "m8", chapter: "multi-model-multi-cloud", title: "Multi-Model, Multi-Cloud, and Cross-Region Inference" },
    { id: "m8/inference-ops",      module: "m8", chapter: "inference-ops",      title: "InferenceOps and Production Lifecycle Management" },

    // ----- Module 9 — Cost & Energy-Efficient Inference -----
    { id: "m9/inference-economics", module: "m9", chapter: "inference-economics", title: "Understanding LLM Inference Economics" },
    { id: "m9/calculating-cost",   module: "m9", chapter: "calculating-cost",   title: "Calculating Inference Cost" },
    { id: "m9/cloud-vs-self-hosted", module: "m9", chapter: "cloud-vs-self-hosted", title: "Cloud vs. Self-Hosted Inference Economics" },
    { id: "m9/energy-tokens-per-joule", module: "m9", chapter: "energy-tokens-per-joule", title: "Energy Consumption and Tokens-per-Joule Metrics" },
    { id: "m9/power-energy-optimization", module: "m9", chapter: "power-energy-optimization", title: "GPU Power and Energy Optimization" },
    { id: "m9/cost-energy-scheduling", module: "m9", chapter: "cost-energy-scheduling", title: "Cost-Aware and Energy-Aware Inference Scheduling" },
    { id: "m9/designing-cost-energy-stack", module: "m9", chapter: "designing-cost-energy-stack", title: "Designing a Cost- and Energy-Efficient Inference Stack" },
  ];

  const MODULES = [
    { id: "m1", title: "Module 1 — LLM and Inference Fundamentals" },
    { id: "m2", title: "Module 2 — GPU Fundamentals" },
    { id: "m3", title: "Module 3 — Planning Your Deployment" },
    { id: "m4", title: "Module 4 — Model Optimization" },
    { id: "m5", title: "Module 5 — Model Interaction Choices" },
    { id: "m6", title: "Module 6 — Inference Infrastructure Optimization" },
    { id: "m7", title: "Module 7 — GPU Kernel Optimization" },
    { id: "m8", title: "Module 8 — Production Operations and Load Testing" },
    { id: "m9", title: "Module 9 — Cost & Energy-Efficient Inference" },
  ];

  // Chapters whose content lives in a .md file under content/<module>/<chapter>.md
  // and is fetched + rendered on first visit. Everything else uses inline HTML.
  const MARKDOWN_CHAPTERS = new Set([
    "m1/intro",
    "m1/how-llms-work",
    "m1/tokens",
    "m1/training-vs-inference",
    "m1/prefill-and-decode",
    "m1/inference-parameters",
    "m1/performance-metrics",
    "m2/where-inference-runs",
    "m2/gpu-architecture",
    "m2/compute-memory-hierarchy",
    "m2/gpu-memory-for-inference",
    "m2/gpu-interconnects",
    "m2/multi-gpu-fundamentals",
    "m2/performance-bottlenecks",
    "m3/deployment-architectures",
    "m3/choosing-the-llm",
    "m3/gpu-selection-memory",
    "m3/capacity-planning",
    "m3/inference-framework",
    "m3/serving-first-llm",
    "m3/cloud-hybrid-onprem",
    "m4/intro-model-optimization",
    "m4/quantization-fundamentals",
    "m4/quantization-methods",
    "m4/distillation",
    "m4/fine-tuning",
    "m4/evaluating-optimized",
    "m5/inference-apis",
    "m5/openai-anthropic-apis",
    "m5/streaming-vs-buffered",
    "m5/prompt-engineering",
    "m5/structured-outputs",
    "m5/function-calling-mcp",
    "m5/agent-harnesses",
    "m6/engine-anatomy",
    "m6/batching-scheduling",
    "m6/kv-cache-paged",
    "m6/kv-cache-optimization",
    "m6/decoding-optimization",
    "m6/distributed-parallelism",
    "m6/advanced-serving",
    "m7/intro-kernel",
    "m7/compute-kernels",
    "m7/cuda-triton",
    "m7/flashattention",
    "m7/kernel-fusion",
    "m7/profiling",
    "m8/production-infra",
    "m8/containerization-k8s",
    "m8/load-balancing-autoscaling",
    "m8/observability",
    "m8/load-testing-slo",
    "m8/multi-model-multi-cloud",
    "m8/inference-ops",
    "m9/inference-economics",
    "m9/calculating-cost",
    "m9/cloud-vs-self-hosted",
    "m9/energy-tokens-per-joule",
    "m9/power-energy-optimization",
    "m9/cost-energy-scheduling",
    "m9/designing-cost-energy-stack",
  ]);

  const mdCache = {};   // page.id -> true once rendered

  // ------------------------------------------------------------------ URL

  function readParams() {
    const q = new URLSearchParams(window.location.search);
    return {
      module: q.get("module") || "m1",
      chapter: q.get("chapter"),
    };
  }

  function buildUrl(module, chapter) {
    const q = new URLSearchParams();
    q.set("module", module);
    if (chapter) q.set("chapter", chapter);
    return window.location.pathname + "?" + q.toString();
  }

  // ------------------------------------------------------------------ RESOLVE

  function resolvePage({ module, chapter }) {
    if (chapter) {
      const exact = PAGES.find((p) => p.module === module && p.chapter === chapter);
      if (exact) return exact;
    }
    const firstInModule = PAGES.find((p) => p.module === module);
    if (firstInModule) return firstInModule;
    return PAGES[0];
  }

  function pageIndex(page) {
    return PAGES.findIndex((p) => p.id === page.id);
  }

  // ------------------------------------------------------------------ RENDER

  async function ensureMarkdownLoaded(page) {
    if (!MARKDOWN_CHAPTERS.has(page.id)) return;
    if (mdCache[page.id]) return;
    const article = document.querySelector(`.content article[data-page="${page.id}"]`);
    if (!article) return;
    try {
      const res = await fetch(`content/${page.module}/${page.chapter}.md`, { cache: "no-store" });
      if (!res.ok) {
        article.innerHTML = `<p>Failed to load chapter (HTTP ${res.status}).</p>`;
        return;
      }
      const md = await res.text();
      article.innerHTML = window.marked
        ? window.marked.parse(md, { mangle: false, headerIds: true })
        : `<pre>${md.replace(/</g, "&lt;")}</pre>`;
      mdCache[page.id] = true;

      // Post-render: lang labels, copy buttons, Prism hints
      article.querySelectorAll("pre > code[class*='language-']").forEach((code) => {
        const pre = code.parentElement;
        if (pre.querySelector(".lang-label")) return;
        const m = code.className.match(/language-([\w+-]+)/);
        if (!m) return;
        const label = document.createElement("span");
        label.className = "lang-label";
        label.textContent = m[1];
        pre.appendChild(label);
      });
      injectCopyButtons();      // idempotent
    } catch (e) {
      article.innerHTML = `<p>Error loading chapter: ${escapeHTML(e.message)}</p>`;
    }
  }

  async function activatePage(page) {
    await ensureMarkdownLoaded(page);

    document.querySelectorAll(".content article").forEach((el) => {
      el.classList.toggle("is-active", el.dataset.page === page.id);
    });

    document.querySelectorAll("[data-module-head]").forEach((el) => {
      el.classList.toggle("is-active", el.dataset.moduleHead === page.module);
    });

    document.querySelectorAll("[data-chapter-link]").forEach((el) => {
      el.classList.toggle(
        "is-active",
        el.dataset.module === page.module && el.dataset.chapter === page.chapter
      );
    });

    document.title = page.title + " — LLM Inference Engineering Handbook";
    renderPageTOC(page);
    renderChapterNav(page);

    const active = document.querySelector(`.content article[data-page="${page.id}"]`);

    // Re-typeset math if MathJax is loaded
    if (active && window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise([active]).catch(() => {});
    }

    // Highlight code if Prism is loaded (autoloader pulls the language on demand)
    if (active && window.Prism && window.Prism.highlightAllUnder) {
      window.Prism.highlightAllUnder(active);
    }

    window.scrollTo({ top: 0, behavior: "auto" });
  }

  // One-time at boot: translate our <span class="lang-label">python</span> hint
  // into the language-* class Prism expects on the <code> element.
  function labelCodeBlocksForPrism() {
    document.querySelectorAll(".content pre").forEach((pre) => {
      const label = pre.querySelector(".lang-label");
      const code = pre.querySelector("code");
      if (!label || !code) return;
      const lang = label.textContent.trim().toLowerCase();
      if (!lang) return;
      code.classList.add("language-" + lang);
    });
  }

  // One-time at boot: add a Copy button to every code block.
  function injectCopyButtons() {
    const copyIcon = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';
    const checkIcon = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="20 6 9 17 4 12"/></svg>';

    document.querySelectorAll(".content pre").forEach((pre) => {
      if (pre.querySelector(".copy-btn")) return; // idempotent
      const btn = document.createElement("button");
      btn.className = "copy-btn";
      btn.type = "button";
      btn.setAttribute("aria-label", "Copy code");
      btn.innerHTML = copyIcon + '<span class="copy-btn-label">Copy</span>';

      btn.addEventListener("click", async () => {
        const code = pre.querySelector("code");
        if (!code) return;
        const text = code.innerText;
        try {
          await navigator.clipboard.writeText(text);
        } catch (_) {
          // Fallback for pages served from file:// or without permission
          const ta = document.createElement("textarea");
          ta.value = text;
          document.body.appendChild(ta);
          ta.select();
          try { document.execCommand("copy"); } catch (__) {}
          document.body.removeChild(ta);
        }
        btn.classList.add("is-copied");
        btn.innerHTML = checkIcon + '<span class="copy-btn-label">Copied</span>';
        setTimeout(() => {
          btn.classList.remove("is-copied");
          btn.innerHTML = copyIcon + '<span class="copy-btn-label">Copy</span>';
        }, 1500);
      });

      pre.appendChild(btn);
    });
  }

  // Theme toggle — swaps data-theme on <html> and persists to localStorage.
  function wireThemeToggle() {
    const btn = document.querySelector(".theme-toggle");
    if (!btn) return;
    btn.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme") || "light";
      const next = current === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("theme", next); } catch (_) {}
    });
  }

  function renderPageTOC(page) {
    const article = document.querySelector(`.content article[data-page="${page.id}"]`);
    const tocHost = document.getElementById("page-toc");
    tocHost.innerHTML = "";
    if (!article) return;

    const headings = article.querySelectorAll("h2, h3");
    headings.forEach((h, i) => {
      if (!h.id) h.id = slugify(h.textContent) + "-" + i;
      const a = document.createElement("a");
      a.href = "#" + h.id;
      a.textContent = h.textContent;
      a.className = "level-" + (h.tagName === "H3" ? "3" : "2");
      a.addEventListener("click", (e) => {
        e.preventDefault();
        document.getElementById(h.id).scrollIntoView({ behavior: "smooth", block: "start" });
        history.replaceState(null, "", "#" + h.id);
      });
      tocHost.appendChild(a);
    });

    document.getElementById("page-toc-title").style.display = headings.length ? "block" : "none";
  }

  function renderChapterNav(page) {
    const host = document.getElementById("chapter-nav");
    host.innerHTML = "";
    const idx = pageIndex(page);
    const prev = idx > 0 ? PAGES[idx - 1] : null;
    const next = idx < PAGES.length - 1 ? PAGES[idx + 1] : null;
    host.appendChild(buildNavButton("prev", "Previous", prev));
    host.appendChild(buildNavButton("next", "Next", next));
  }

  function buildNavButton(dir, label, target) {
    const btn = document.createElement("button");
    btn.className = "nav-link " + dir;
    btn.disabled = !target;
    btn.innerHTML =
      `<span class="direction">${label}</span>` +
      `<span class="title">${target ? escapeHTML(target.title) : "—"}</span>`;
    if (target) {
      btn.addEventListener("click", () => navigateTo(target));
    }
    return btn;
  }

  // ------------------------------------------------------------------ NAV

  function navigateTo(page) {
    const url = buildUrl(page.module, page.chapter);
    history.pushState({ module: page.module, chapter: page.chapter }, "", url);
    activate(page);
  }

  function wireSidebar() {
    document.querySelectorAll("[data-module-head]").forEach((el) => {
      el.addEventListener("click", (e) => {
        e.preventDefault();
        const mod = el.dataset.moduleHead;
        const page = resolvePage({ module: mod, chapter: null });
        navigateTo(page);
      });
    });
    document.querySelectorAll("[data-chapter-link]").forEach((el) => {
      el.addEventListener("click", (e) => {
        e.preventDefault();
        const page = PAGES.find(
          (p) => p.module === el.dataset.module && p.chapter === el.dataset.chapter
        );
        if (page) navigateTo(page);
      });
    });
  }

  function wireScrollSpy() {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          const id = entry.target.id;
          if (!id) return;
          const link = document.querySelector(`#page-toc a[href="#${CSS.escape(id)}"]`);
          if (!link) return;
          if (entry.isIntersecting) {
            document.querySelectorAll("#page-toc a.is-active").forEach((a) => a.classList.remove("is-active"));
            link.classList.add("is-active");
          }
        });
      },
      { rootMargin: "-80px 0px -60% 0px", threshold: 0 }
    );
    document.addEventListener("page:activated", () => {
      const active = document.querySelector(".content article.is-active");
      if (!active) return;
      observer.disconnect();
      active.querySelectorAll("h2, h3").forEach((h) => observer.observe(h));
    });
  }

  function slugify(s) {
    return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
  }

  function escapeHTML(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
  }

  async function activate(page) {
    await activatePage(page);
    document.dispatchEvent(new CustomEvent("page:activated", { detail: { page } }));
  }

  function boot() {
    labelCodeBlocksForPrism();
    injectCopyButtons();
    wireThemeToggle();
    wireSidebar();
    wireScrollSpy();
    const params = readParams();
    const page = resolvePage(params);
    activate(page);

    window.addEventListener("popstate", () => {
      const params = readParams();
      const page = resolvePage(params);
      activate(page);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
