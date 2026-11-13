/* ==========================================================================
   LLM Inference Handbook — navigation + on-page TOC
   Zero deps. Works on GitHub Pages as plain static files.
   ========================================================================== */

(function () {
  "use strict";

  const PAGES = [
    { id: "introduction",                                         module: "introduction",             chapter: null,                      title: "Introduction" },
    { id: "foundations/what-is-inference",                        module: "foundations",              chapter: "what-is-inference",       title: "What is inference?" },
    { id: "foundations/understanding-tokens",                     module: "foundations",              chapter: "understanding-tokens",    title: "Understanding tokens" },
    { id: "planning-your-deployment/selecting-right-model",       module: "planning-your-deployment", chapter: "selecting-right-model",   title: "Selecting the right model" },
    { id: "planning-your-deployment/capacity-planning-basics",    module: "planning-your-deployment", chapter: "capacity-planning-basics", title: "Capacity planning basics" },
    { id: "model-preparation/quantization-fundamentals",          module: "model-preparation",        chapter: "quantization-fundamentals", title: "Quantization fundamentals" },
    { id: "model-preparation/kv-cache-optimization",              module: "model-preparation",        chapter: "kv-cache-optimization",   title: "KV cache optimization" },
    { id: "model-interaction/openai-compatible-api",              module: "model-interaction",        chapter: "openai-compatible-api",   title: "OpenAI-compatible API" },
    { id: "model-interaction/streaming-vs-buffered",              module: "model-interaction",        chapter: "streaming-vs-buffered",   title: "Streaming vs. buffered" },
  ];

  // ------------------------------------------------------------------ URL

  function readParams() {
    const q = new URLSearchParams(window.location.search);
    return {
      module: q.get("module") || "introduction",
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

  // Given (module, chapter), find the matching page (fall back to first
  // chapter in the module if the chapter is missing).
  function resolvePage({ module, chapter }) {
    if (chapter) {
      const exact = PAGES.find((p) => p.module === module && p.chapter === chapter);
      if (exact) return exact;
    }
    // chapter-less — prefer the page with `chapter: null`, otherwise first in the module
    const landing = PAGES.find((p) => p.module === module && p.chapter === null);
    if (landing) return landing;
    const firstInModule = PAGES.find((p) => p.module === module);
    if (firstInModule) return firstInModule;
    return PAGES[0];
  }

  function pageIndex(page) {
    return PAGES.findIndex((p) => p.id === page.id);
  }

  // ------------------------------------------------------------------ RENDER

  function activatePage(page) {
    // content articles
    document.querySelectorAll(".content article").forEach((el) => {
      el.classList.toggle("is-active", el.dataset.page === page.id);
    });

    // left nav — module headers
    document.querySelectorAll("[data-module]").forEach((el) => {
      el.classList.toggle("is-active", el.dataset.module === page.module);
    });

    // left nav — chapter links
    document.querySelectorAll("[data-chapter]").forEach((el) => {
      el.classList.toggle(
        "is-active",
        el.dataset.module === page.module && el.dataset.chapter === page.chapter
      );
    });

    // update page title
    document.title = page.title + " — LLM Inference Handbook";

    // build right-column TOC from the active article's h2/h3s
    renderPageTOC(page);

    // build prev/next
    renderChapterNav(page);

    // scroll the main content back to top
    window.scrollTo({ top: 0, behavior: "auto" });
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

    // show/hide the title row based on whether we have any entries
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
    activatePage(page);
  }

  // ------------------------------------------------------------------ WIRE

  function wireSidebar() {
    document.querySelectorAll("[data-module]").forEach((el) => {
      el.addEventListener("click", (e) => {
        e.preventDefault();
        const mod = el.dataset.module;
        const page = resolvePage({ module: mod, chapter: null });
        navigateTo(page);
      });
    });

    document.querySelectorAll("[data-chapter]").forEach((el) => {
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
    // Highlight TOC items as the user scrolls through h2/h3 landmarks.
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
    // Re-scan on every page activation
    document.addEventListener("page:activated", () => {
      const active = document.querySelector(".content article.is-active");
      if (!active) return;
      observer.disconnect();
      active.querySelectorAll("h2, h3").forEach((h) => observer.observe(h));
    });
  }

  // ------------------------------------------------------------------ UTIL

  function slugify(s) {
    return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "");
  }

  function escapeHTML(s) {
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // ------------------------------------------------------------------ WIDGET: Token Decode

  const TD_TOKENS = ["Modular", "is", "a", "unified", "inference", "?"];
  const TD_DISTRIBUTION = [
    { label: "platform", percentage: 64 },
    { label: "stack",    percentage: 18 },
    { label: "engine",   percentage: 11 },
    { label: "service",  percentage: 7  },
  ];
  const TD_MAX_STEPS = 8;
  const TD_KV_SLOTS = 6;

  function initTokenDecodeWidget(root) {
    const slots = {
      stepCount:    root.querySelector('[data-slot="step-count"]'),
      progress:     root.querySelector('[data-slot="progress"]'),
      sequence:     root.querySelector('[data-slot="sequence"]'),
      distribution: root.querySelector('[data-slot="distribution"]'),
      kv:           root.querySelector('[data-slot="kv"]'),
      kvNote:       root.querySelector('[data-slot="kv-note"]'),
    };
    const btnStep     = root.querySelector('[data-action="step"]');
    const btnAutoplay = root.querySelector('[data-action="autoplay"]');
    const btnReset    = root.querySelector('[data-action="reset"]');

    let step = 0;
    let autoplayTimer = null;

    // one-time build: distribution rows + KV cells
    slots.distribution.innerHTML = TD_DISTRIBUTION.map((d, i) => `
      <div class="wtd-dist-item${i === 0 ? ' is-top' : ''}" data-dist-item>
        <div class="wtd-dist-row"><span class="label">${escapeHTML(d.label)}</span><span class="pct">${d.percentage}%</span></div>
        <div class="wtd-dist-bar"><div class="wtd-dist-fill" data-dist-fill style="width:0%"></div></div>
      </div>
    `).join("");

    slots.kv.innerHTML = Array.from({ length: TD_KV_SLOTS })
      .map(() => '<div class="wtd-kv-cell"></div>').join("");

    function render() {
      // step counter
      slots.stepCount.textContent = `Step ${step + 1} / ${TD_MAX_STEPS}`;

      // progress bar
      slots.progress.style.width = (((step + 1) / TD_MAX_STEPS) * 100) + "%";

      // sequence
      const tokenHTML = TD_TOKENS.map((tok, idx) => {
        let cls = "wtd-token";
        if (idx < step) cls += " is-done";
        else if (idx === step) cls += " is-current";
        return `<div class="${cls}">T${idx} ${escapeHTML(tok)}</div>`;
      }).join("");
      const ghost = step < TD_MAX_STEPS - 1
        ? `<div class="wtd-token is-ghost">T${step + 1} ?</div>`
        : "";
      slots.sequence.innerHTML = tokenHTML + ghost;

      // distribution — highlight top row only on step 0 (matches React source)
      const distItems = slots.distribution.querySelectorAll('[data-dist-item]');
      const distFills = slots.distribution.querySelectorAll('[data-dist-fill]');
      distItems.forEach((item, i) => {
        item.classList.toggle("is-top", step === 0 && i === 0);
      });
      distFills.forEach((fill, i) => {
        fill.style.width = TD_DISTRIBUTION[i].percentage + "%";
      });

      // KV cache cells
      const cells = slots.kv.querySelectorAll('.wtd-kv-cell');
      cells.forEach((cell, idx) => {
        cell.classList.toggle("is-current", idx === step);
      });

      // KV note
      const reused = Math.min(step, TD_KV_SLOTS - 1);
      slots.kvNote.textContent =
        `Reuses ${reused} cached K/V pair${reused === 1 ? '' : 's'}; ` +
        `computes ${step > 0 ? '1 new' : 'fresh'} one this step.`;

      // Disable step at the end
      btnStep.disabled = step >= TD_MAX_STEPS - 1;

      // If autoplay reached the end, stop it
      if (autoplayTimer && step >= TD_MAX_STEPS - 1) stopAutoplay();
    }

    function startAutoplay() {
      if (autoplayTimer) return;
      btnAutoplay.textContent = "Pause";
      autoplayTimer = setInterval(() => {
        if (step < TD_MAX_STEPS - 1) {
          step += 1;
          render();
        } else {
          stopAutoplay();
        }
      }, 650);
    }

    function stopAutoplay() {
      if (!autoplayTimer) return;
      clearInterval(autoplayTimer);
      autoplayTimer = null;
      btnAutoplay.textContent = "Autoplay";
    }

    btnStep.addEventListener("click", () => {
      if (step < TD_MAX_STEPS - 1) {
        step += 1;
        render();
      }
    });

    btnAutoplay.addEventListener("click", () => {
      if (autoplayTimer) stopAutoplay();
      else startAutoplay();
    });

    btnReset.addEventListener("click", () => {
      stopAutoplay();
      step = 0;
      render();
    });

    render();
  }

  function initAllWidgets() {
    document.querySelectorAll('[data-widget="token-decode"]').forEach((el) => {
      if (el.dataset.inited === "true") return;
      el.dataset.inited = "true";
      initTokenDecodeWidget(el);
    });
  }

  // ------------------------------------------------------------------ BOOT

  function activate(page) {
    activatePage(page);
    document.dispatchEvent(new CustomEvent("page:activated", { detail: { page } }));
  }

  function boot() {
    wireSidebar();
    wireScrollSpy();
    initAllWidgets();
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
