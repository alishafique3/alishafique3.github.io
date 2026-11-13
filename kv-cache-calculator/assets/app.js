/*
 * KV Cache Calculator v2 — UI bridge.
 * Reuses models.js + calculator-core.js verbatim (adapted from
 * kvcache-ai/kvcache-blog @ d455485bd1c5265951849f011a60564da1b23cfe, MIT).
 * See ../THIRD_PARTY_NOTICES.md.
 */
(function () {
  "use strict";

  const data = window.KV_MODEL_DATA;
  const calc = window.KVCacheCalculator;
  if (!data || !calc) {
    console.error("KV_MODEL_DATA or KVCacheCalculator missing");
    return;
  }

  const $ = (id) => document.getElementById(id);

  const CACHE_COLOR = {
    kv:        "#0ea5e9",
    attention: "#06b6d4",
    cache:     "#38bdf8",
    indexer:   "#a78bfa",
    draft:     "#ec4899",
    linear:    "#f59e0b",
    conv:      "#f59e0b",
    recurrent: "#fb923c",
  };


  /* ---------- Model helpers (ported from upstream script.js) ---------- */
  function numericField(model, name, fallback) {
    const v = model && model.fields ? Number(model.fields[name]) : NaN;
    return Number.isFinite(v) ? v : fallback;
  }
  function visibleFamily(model) {
    const f = model.family || "Other";
    return f.indexOf("Qwen") === 0 ? "Qwen" : f;
  }
  function allFamilies() {
    return Array.from(new Set(data.models.map(visibleFamily))).sort();
  }
  function modelsOf(family) {
    return data.models.filter((m) => visibleFamily(m) === family);
  }
  function validTpSizes(model) {
    const n = Math.max(1, Math.floor(numericField(model, "num_key_value_heads", 1)));
    const sizes = [];
    for (let i = 1; i <= n; i++) if (n % i === 0) sizes.push(i);
    return sizes;
  }
  function tpHelp(model) {
    const n = Math.max(1, Math.floor(numericField(model, "num_key_value_heads", 1)));
    const sizes = validTpSizes(model);
    const choices = sizes.length === 1
      ? String(sizes[0])
      : sizes.length === 2
        ? sizes.join(" or ")
        : sizes.slice(0, -1).join(", ") + ", or " + sizes[sizes.length - 1];
    const label = n === 1 ? "KV head" : "KV heads";
    return `Model has ${n} ${label}. Valid TP sizes: ${choices}.`;
  }
  function hasIndexer(model) {
    return Number.isFinite(numericField(model, "index_head_dim", NaN));
  }
  function hasDraftCache(model) {
    if (model.formula === "deepseek_v4_hybrid") {
      const main = numericField(model, "num_hidden_layers", 0);
      return Array.isArray(model.fields.compress_ratios)
        && model.fields.compress_ratios.length > main;
    }
    if (!model.fields || model.fields.disable_draft_kv_cache === true) return false;
    if (numericField(model, "num_nextn_predict_layers", 0) > 0) return true;
    if (model.fields.use_mtp === true) {
      return numericField(model, "num_mtp_modules", 0)
           * numericField(model, "mtp_transformer_layers", 0) > 0;
    }
    return false;
  }
  function hasLinearState(model) {
    return model.formula === "qwen_linear_full_hybrid";
  }
  function speculativeModelsFor(model) {
    const ids = (model && model.speculative_model_ids) || [];
    const pool = (data && data.speculative_models) || [];
    return ids.map((id) => pool.find((x) => x.id === id)).filter(Boolean);
  }
  function defaultPrecision(model) {
    return model.formula === "deepseek_v4_hybrid" ? "fp8_int8" : "bf16_fp16";
  }
  function defaultIndexerPrecision(model) {
    return model.formula === "deepseek_v4_hybrid" ? "fp4_int4" : "bf16_fp16";
  }
  function modelForCalculation(model) {
    if (model.id !== "minimax-m3") return model;
    const fields = Object.assign({}, model.fields);
    delete fields.indexer_fixed_precision_id;
    return Object.assign({}, model, { fields });
  }

  /* ---------- Formula row synthesis (ported from upstream) ----------
   * Converts elementPlan.formulaRows into a UI-friendly list, applying
   *  - withoutSequences(): strips redundant "sequences x" prefix
   *  - localHeads():       rewrites head counts as max(h / TP, 1) when TP > 1
   *  - appends per_device_bytes and all_device_bytes derived rows
   */
  function withoutSequences(expr) {
    return String(expr || "")
      .replace(/tokens x sequences x /g, "tokens x ")
      .replace(/min\(tokens, sliding_window\) x sequences x /g, "min(tokens, sliding_window) x ")
      .replace(/^sequences x /, "");
  }
  function localHeads(expr, field) {
    return String(expr).replace(new RegExp("\\b" + field + "\\b", "g"),
      "max(" + field + " / TP size, 1)");
  }
  function prettyExpr(expr) {
    return String(expr || "").replace(/ x /g, " × ");
  }

  function buildFormulaRows(result, model) {
    const formula = model.formula;
    const raw = (result.elementPlan && result.elementPlan.formulaRows) || [];
    const tp = result.tensorParallel || 1;
    const tpActive = tp > 1;
    const perDeviceRows = [];
    const draftTotal = raw.find((r) => r.name === "draft_total_bytes");

    raw.forEach((row) => {
      if (row.name === "total_bytes" || row.name === "draft_total_bytes") return;
      let expression = withoutSequences(row.expression);
      if (tpActive) {
        if (formula === "standard_gqa" && row.name !== "active_layers") {
          expression = localHeads(expression, "num_key_value_heads");
        }
        if (formula === "dsa_mla" && row.name === "indexer_bytes") {
          expression = localHeads(expression, "index_num_key_heads");
        }
        if (formula === "qwen_linear_full_hybrid") {
          expression = localHeads(expression, "num_key_value_heads");
          expression = localHeads(expression, "linear_num_key_heads");
          expression = localHeads(expression, "linear_num_value_heads");
          expression = localHeads(expression, "draft_num_key_value_heads");
        }
        if (formula === "mixed_full_sliding_gqa") {
          expression = localHeads(expression, "full_kv_heads");
          expression = localHeads(expression, "sliding_kv_heads");
        }
        if (formula === "minimax_msa" && row.name === "kv_bytes") {
          expression = localHeads(expression, "num_key_value_heads");
        }
        if ((formula === "minimax_msa" || formula === "deepseek_v4_hybrid") && row.name === "indexer_bytes") {
          expression = localHeads(expression, "index_num_key_heads");
        }
      }
      perDeviceRows.push({ name: row.name, expression });
    });

    // Terms for per_device_bytes derivation
    let terms;
    if (formula === "standard_gqa") {
      const total = raw.find((r) => r.name === "total_bytes");
      if (total) {
        let expression = withoutSequences(total.expression);
        if (tpActive) expression = localHeads(expression, "num_key_value_heads");
        perDeviceRows.push({ name: "kv_bytes", expression });
      }
      terms = "kv_bytes";
    } else if (formula === "mla") {
      const total = raw.find((r) => r.name === "total_bytes");
      if (total) perDeviceRows.push({ name: "kv_bytes", expression: withoutSequences(total.expression) });
      terms = "kv_bytes";
    } else if (formula === "dsa_mla") {
      terms = "kv_bytes + indexer_bytes";
    } else if (formula === "qwen_linear_full_hybrid") {
      terms = perDeviceRows.some((r) => r.name === "linear_conv_state_bytes")
        ? "full_kv_bytes + linear_conv_state_bytes + linear_recurrent_state_bytes"
        : "full_kv_bytes";
    } else if (formula === "mixed_full_sliding_gqa") {
      terms = "full_kv_bytes + sliding_kv_bytes";
    } else {
      terms = "kv_bytes + indexer_bytes";
    }
    if (draftTotal) terms += " + " + withoutSequences(draftTotal.expression);

    const payload = terms.includes(" + ") ? "(" + terms + ")" : terms;
    perDeviceRows.push({ name: "per_device_bytes", expression: "sequences × " + payload });
    perDeviceRows.push({ name: "all_device_bytes", expression: "TP size × per_device_bytes" });
    return perDeviceRows;
  }

  /* ---------- Formatters ---------- */
  const GIB = 1024 ** 3;
  const MIB = 1024 ** 2;
  const KIB = 1024;
  function fmtBytes(bytes) {
    if (bytes >= GIB) return (bytes / GIB).toFixed(3) + " GiB";
    if (bytes >= MIB) return (bytes / MIB).toFixed(2) + " MiB";
    if (bytes >= KIB) return (bytes / KIB).toFixed(2) + " KiB";
    return Math.round(bytes).toLocaleString() + " B";
  }
  function fmtCount(n) {
    if (n == null) return "—";
    return Number(n).toLocaleString(undefined, { maximumFractionDigits: 3 });
  }

  /* ---------- State ---------- */
  const state = {
    family: null,
    modelId: null,
    tokens: 1024,
    sequences: 1,
    tp: 1,
    precision: "bf16_fp16",
    indexerPrecision: "bf16_fp16",
    specModelId: "",
    specTokens: 0,
    draft: false,
    linear: true,
  };

  /* ---------- Populators ---------- */
  function populateFamilies() {
    const sel = $("family");
    sel.innerHTML = "";
    allFamilies().forEach((f) => {
      const opt = document.createElement("option");
      opt.value = f; opt.textContent = f;
      sel.appendChild(opt);
    });
  }
  function populateModels(family) {
    const sel = $("model");
    sel.innerHTML = "";
    modelsOf(family).forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m.id; opt.textContent = m.label || m.id;
      sel.appendChild(opt);
    });
  }
  function populateTp(model) {
    const sel = $("tp");
    const sizes = validTpSizes(model);
    sel.innerHTML = "";
    sizes.forEach((s) => {
      const opt = document.createElement("option");
      opt.value = String(s); opt.textContent = String(s);
      sel.appendChild(opt);
    });
    $("tp-help").textContent = tpHelp(model);
    if (!sizes.includes(state.tp)) state.tp = sizes[0] || 1;
    sel.value = String(state.tp);
  }
  function populateSpecModels(model) {
    const sel = $("spec-model");
    const list = speculativeModelsFor(model);
    sel.innerHTML = "";
    const none = document.createElement("option");
    none.value = ""; none.textContent = "— none —";
    sel.appendChild(none);
    list.forEach((m) => {
      const opt = document.createElement("option");
      opt.value = m.id; opt.textContent = m.label || m.id;
      sel.appendChild(opt);
    });
  }

  /* ---------- Apply model defaults ---------- */
  function currentModel() { return data.models.find((m) => m.id === state.modelId); }
  function currentSpecModel() {
    if (!state.specModelId) return null;
    return (data.speculative_models || []).find((m) => m.id === state.specModelId) || null;
  }

  function onModelChanged() {
    const m = currentModel();
    if (!m) return;

    state.tokens = m.default_tokens || 1024;
    $("tokens").value = state.tokens;
    state.sequences = 1; $("sequences").value = state.sequences;

    populateTp(m);
    state.tp = Number($("tp").value);

    state.precision = defaultPrecision(m);
    $("precision").value = state.precision;
    state.indexerPrecision = defaultIndexerPrecision(m);
    $("indexer").value = state.indexerPrecision;

    // Toggle conditional sections
    $("indexer-row").style.display = hasIndexer(m) ? "flex" : "none";
    $("draft-wrap").style.display  = hasDraftCache(m) ? "flex" : "none";
    $("linear-wrap").style.display = hasLinearState(m) ? "flex" : "none";

    const specs = speculativeModelsFor(m);
    $("spec-row").style.display = specs.length > 0 ? "flex" : "none";
    if (specs.length > 0) populateSpecModels(m);
    state.specModelId = ""; $("spec-model") && ($("spec-model").value = "");
    state.specTokens = 0; $("spec-tokens").value = 0;

    state.draft = false; $("draft").checked = false;
    state.linear = true; $("linear").checked = true;
  }

  /* ---------- Compute & render ---------- */
  function compute() {
    const m = currentModel();
    if (!m) return null;

    const input = {
      tokens: state.tokens,
      sequences: state.sequences,
      tensorParallel: state.tp,
      precision: state.precision,
      indexerPrecision: hasIndexer(m) ? state.indexerPrecision : undefined,
      includeDraftKvCache: hasDraftCache(m) ? state.draft : false,
      includeLinearAttentionState: hasLinearState(m) ? state.linear : false,
      speculativeModel: currentSpecModel(),
      specTokens: state.specTokens,
    };

    try {
      return calc.calculate(modelForCalculation(m), input, data);
    } catch (e) {
      console.error("calc failed", e);
      return null;
    }
  }

  function render() {
    const m = currentModel();
    if (!m) return;

    const r = compute();
    if (!r) {
      $("per-device-val").textContent = "—";
      $("summary-note").textContent = "—";
      $("cache-list").innerHTML = "";
      $("component-list").innerHTML = "";
      return;
    }

    // Big number
    $("per-device-val").textContent = Number(r.perDeviceGiB).toFixed(3);
    const formulaLabel = r.elementPlan && r.elementPlan.formulaLabel ? r.elementPlan.formulaLabel : m.formula;
    $("summary-note").textContent =
      `${formulaLabel} · ${fmtCount(r.totalCachedTokens)} tokens · TP ${r.tensorParallel}`;

    // Metric cards
    $("per-seq-val").textContent   = fmtBytes(r.bytesPerSequence);
    $("per-token-val").textContent = fmtBytes(r.bytesPerToken);
    $("bpp-val").textContent       = r.precisionLabel + ` · ${r.bytesPerElement}`;
    $("all-devices-val").textContent = fmtBytes(r.allDeviceBytes);

    // Cache groups
    const items = r.cacheGroups.map((g) => {
      const color = CACHE_COLOR[g.role] || "#64748b";
      const sharded = g.replicated ? "replicated on every device" :
        (g.localHeads < g.heads ? `${g.localHeads}/${g.heads} heads/device` : `${g.heads} heads`);
      return `
        <li>
          <div class="primary">
            <span class="dot" style="background:${color}"></span>
            <span>${g.label || g.role}</span>
            <span class="num">${fmtBytes(g.perDeviceBytes)}<br><span style="color:var(--muted);font-weight:500;font-size:11px;">per device</span></span>
          </div>
          <div class="secondary">${sharded} · total ${fmtBytes(g.bytes)}</div>
        </li>
      `;
    }).join("");
    $("cache-list").innerHTML = items;

    // Per-model formula rows (symbolic equations) + numeric model-field values
    $("formula-label").textContent = formulaLabel || "—";
    const rows = buildFormulaRows(r, m);
    $("formula-rows").innerHTML = rows.map((row) => `
      <div class="formula-row">
        <div class="fr-name">${row.name}</div>
        <div class="fr-expr">${prettyExpr(row.expression)}</div>
      </div>
    `).join("");

    const fieldItems = (r.components || [])
      .filter((c) => c && c[0] && c[1] != null && c[1] !== "")
      .map((c) => {
        const [name, value] = c;
        const isNum = typeof value === "number";
        const val = isNum ? fmtCount(value) : String(value);
        // Long multi-item strings (e.g. "a=1, b=2, c=3") break out onto their own line
        const wrap = !isNum && (val.length > 38 || val.includes(", "));
        return `
          <div class="mf-item ${wrap ? 'wrap' : ''}">
            <span class="mf-label">${name}</span>
            <span class="mf-value">${val}</span>
          </div>
        `;
      }).join("");
    if (fieldItems) {
      $("model-fields").style.display = "";
      $("model-fields-list").innerHTML = fieldItems;
    } else {
      $("model-fields").style.display = "none";
    }

    $("cache-note").textContent = (r.elementPlan && r.elementPlan.note) || "";

    // Source link
    const src = m.source_url || "";
    const srcLink = $("source-link");
    if (src) {
      srcLink.href = src;
      srcLink.textContent = src.replace(/^https?:\/\//, "");
      srcLink.style.display = "";
    } else {
      srcLink.href = "#"; srcLink.textContent = "—"; srcLink.style.display = "";
    }

  }

  /* ---------- Wiring ---------- */
  function bindNumber(id, setter, { minZero = false } = {}) {
    const el = $(id);
    el.addEventListener("input", () => {
      const v = Number(el.value);
      if (Number.isFinite(v) && v >= (minZero ? 0 : 1)) setter(v);
      render();
    });
  }

  function init() {
    populateFamilies();
    state.family = $("family").value || allFamilies()[0];
    populateModels(state.family);
    state.modelId = $("model").value || (data.models[0] && data.models[0].id);
    onModelChanged();
    render();

    $("family").addEventListener("change", (e) => {
      state.family = e.target.value;
      populateModels(state.family);
      state.modelId = $("model").value;
      onModelChanged(); render();
    });
    $("model").addEventListener("change", (e) => {
      state.modelId = e.target.value;
      onModelChanged(); render();
    });

    bindNumber("tokens",    (v) => { state.tokens = Math.max(1, Math.floor(v)); });
    bindNumber("sequences", (v) => { state.sequences = Math.max(1, Math.floor(v)); });

    $("tp").addEventListener("change", (e) => { state.tp = Number(e.target.value); render(); });
    $("precision").addEventListener("change", (e) => { state.precision = e.target.value; render(); });
    $("indexer").addEventListener("change",   (e) => { state.indexerPrecision = e.target.value; render(); });

    $("spec-model").addEventListener("change", (e) => { state.specModelId = e.target.value; render(); });
    bindNumber("spec-tokens", (v) => { state.specTokens = Math.max(0, Math.floor(v)); }, { minZero: true });

    $("draft").addEventListener("change", (e) => { state.draft = e.target.checked; render(); });
    $("linear").addEventListener("change", (e) => { state.linear = e.target.checked; render(); });
  }

  document.addEventListener("DOMContentLoaded", init);
})();
