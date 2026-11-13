/*
 * Model Explorer — multi-model edition.
 * - Renders field-explanation cards for each supported model.
 * - Wires click-to-highlight sync between config.json rows and cards,
 *   scoped per model view.
 * - Model picker toggles between GPT-2 Small (124M) and Llama 3.1 8B.
 */
(function () {
  "use strict";

  /* -------- Field catalog per model -------- */

  const FIELDS_GPT2 = [
    {
      key: "architectures",
      desc: "Python class HF Transformers instantiates when loading the model. GPT-2's head is <code>GPT2LMHeadModel</code> — the backbone + a tied LM head for next-token prediction.",
    },
    {
      key: "model_type",
      desc: "Architecture family string used by HF's <code>AutoModel</code> to pick the right code path. <code>\"gpt2\"</code> → the GPT-2 implementation in <code>modeling_gpt2.py</code>.",
    },
    {
      key: "vocab_size",
      affects: true,
      desc: "Number of tokens in the BPE vocabulary (<code>V</code>). Drives the size of the token-embedding matrix and (when tied) the LM-head projection.",
      calc: [
        "Token embedding = V × h",
        "            = 50,257 × 768",
      ],
      result: "38,597,376 params",
    },
    {
      key: "n_positions",
      affects: true,
      desc: "Maximum sequence length the model was trained on (<code>P</code>). GPT-2 uses a learned position-embedding table of shape P × h.",
      calc: [
        "Position embedding = P × h",
        "               = 1,024 × 768",
      ],
      result: "786,432 params",
    },
    {
      key: "n_ctx",
      desc: "Legacy alias for <code>n_positions</code>; kept for backward compatibility. Does not add new parameters.",
    },
    {
      key: "n_embd",
      affects: true,
      desc: "Hidden size (<code>h</code>) — the width of every residual stream and the dimension of every token representation. Drives nearly every weight matrix in the model.",
      calc: [
        "Appears everywhere: embeddings (V × h, P × h),",
        "attention projections (h × 3h and h × h),",
        "MLP projections (h × 4h and 4h × h),",
        "layernorm vectors (2 × h).",
      ],
      result: "h = 768",
    },
    {
      key: "n_layer",
      affects: true,
      desc: "Number of stacked transformer blocks (<code>L</code>). Total transformer weight count scales linearly with this.",
      calc: [
        "Transformer-block params = L × (per-block params)",
        "              = 12 × 7,087,872",
      ],
      result: "85,054,464 params",
    },
    {
      key: "n_head",
      desc: "Number of attention heads. The hidden dim is split so each head has <code>head_dim = h / n_head = 768 / 12 = 64</code>. The <strong>total</strong> QKV weight count is unchanged by the split — <code>n_head</code> is purely a reshape.",
      calc: [
        "head_dim = h / n_head",
        "        = 768 / 12 = 64",
      ],
      result: "0 extra params (reshape only)",
    },
    {
      key: "n_inner",
      affects: true,
      desc: "MLP intermediate size. When <code>null</code>, GPT-2 defaults to <code>4 × h</code>. This is the width of the MLP's hidden activation.",
      calc: [
        "i = 4 × h = 4 × 768 = 3,072",
        "",
        "Up-projection:    h × i  + bias i  → 2,362,368",
        "Down-projection:  i × h  + bias h  → 2,360,064",
      ],
      result: "4,722,432 params per block",
    },
    {
      key: "activation_function",
      desc: "Non-linearity used inside the MLP. <code>\"gelu_new\"</code> is the smooth approximation to GELU used in the original GPT-2 code. Element-wise — no parameters.",
    },
    {
      key: "layer_norm_epsilon",
      desc: "Small constant added inside LayerNorm's denominator for numerical stability. Hyperparameter; not a trainable parameter.",
    },
    {
      key: "initializer_range",
      desc: "Standard deviation of the normal distribution used to initialize weights at model-creation time. Doesn't affect the parameter count — only their initial values.",
    },
    {
      key: "resid_pdrop",
      desc: "Dropout probability applied to the residual connection outputs during training. Runtime-only — no parameters.",
    },
    {
      key: "embd_pdrop",
      desc: "Dropout probability applied to the sum of token + position embeddings during training. Runtime-only — no parameters.",
    },
    {
      key: "attn_pdrop",
      desc: "Dropout applied to attention probabilities during training. Runtime-only — no parameters.",
    },
    {
      key: "bos_token_id",
      desc: "Token ID the tokenizer uses to mark the beginning of a sequence. GPT-2 reuses the end-of-text token (<code>50256</code>) for both BOS and EOS.",
    },
    {
      key: "eos_token_id",
      desc: "Token ID for end-of-sequence. Reused as a padding token when no dedicated pad token is set.",
    },
    {
      key: "use_cache",
      desc: "Whether to return key/value states from the attention layers for reuse during autoregressive generation (the KV cache). Inference-time knob; affects memory, not parameter count.",
    },
    {
      key: "scale_attn_weights",
      desc: "Divide attention scores by <code>√head_dim</code> before the softmax (the standard scaled-dot-product variant). Computation-only.",
    },
    {
      key: "tie_word_embeddings",
      affects: true,
      desc: "If <code>true</code>, the LM-head weight matrix <strong>shares</strong> parameters with the input token-embedding matrix. GPT-2 does this, which avoids an extra <code>V × h = 38,597,376</code> parameters.",
      calc: [
        "Without tying: + V × h   = + 38,597,376",
        "With tying:    + 0                    (shared)",
      ],
      result: "Saves 38,597,376 params",
    },
  ];

  const FIELDS_LLAMA = [
    {
      key: "architectures",
      desc: "Python class HF Transformers instantiates when loading the model: <code>LlamaForCausalLM</code>. Same backbone as Llama 2/3, differences live in the config values.",
    },
    {
      key: "model_type",
      desc: "Architecture family string used by HF's <code>AutoModel</code>. <code>\"llama\"</code> → <code>modeling_llama.py</code> code path.",
    },
    {
      key: "vocab_size",
      affects: true,
      desc: "Vocabulary size (<code>V</code> = 128,256) — a ~2.5× larger vocabulary than GPT-2's. Larger vocab means shorter tokenizations and better multilingual coverage. Since <code>tie_word_embeddings</code> is <code>false</code>, this matrix is counted <strong>twice</strong>: once for <code>embed_tokens</code> and once for <code>lm_head</code>.",
      calc: [
        "embed_tokens = V × h = 128,256 × 4,096 = 525,336,576",
        "lm_head       = V × h                     = 525,336,576",
      ],
      result: "1,050,673,152 params total",
    },
    {
      key: "hidden_size",
      affects: true,
      desc: "Residual-stream width (<code>h</code> = 4,096). Every attention projection, MLP projection, and RMSNorm vector is sized against this.",
      calc: [
        "Appears everywhere: embed (V × h),",
        "Q/O projections (h × h), K/V (h × kv_heads × head_dim),",
        "MLP (h × i and i × h × 3 matrices),",
        "RMSNorm scales (1 × h each).",
      ],
      result: "h = 4,096",
    },
    {
      key: "intermediate_size",
      affects: true,
      desc: "MLP intermediate width (<code>i</code> = 14,336). Llama uses SwiGLU which needs <strong>three</strong> matrices of width <code>i</code>, so MLPs dominate the per-layer cost. The ratio <code>i / h ≈ 3.5</code> is tuned so the SwiGLU block has a similar FLOPs cost to a 4·h GELU-MLP.",
      calc: [
        "Per block, MLP contributes 3 × h × i:",
        "  gate_proj + up_proj + down_proj",
        "  = 3 × (4,096 × 14,336) = 176,160,768",
      ],
      result: "176,160,768 params per block",
    },
    {
      key: "num_hidden_layers",
      affects: true,
      desc: "Number of transformer blocks (<code>L</code> = 32).",
      calc: [
        "Transformer-block params = L × (per-block params)",
        "              = 32 × 218,112,000",
      ],
      result: "6,979,584,000 params",
    },
    {
      key: "num_attention_heads",
      desc: "Number of query (Q) heads. <code>head_dim = h / n_head = 4,096 / 32 = 128</code>. Q, K, V, O projections produce outputs of width <code>n_head × head_dim = h</code> for Q and O.",
    },
    {
      key: "num_key_value_heads",
      affects: true,
      desc: "<strong>GQA</strong> setting — Llama shares each K/V head across <code>n_head / n_kv_heads = 32 / 8 = 4</code> Q heads. This shrinks the K and V projection matrices and the KV cache by a factor of 4 relative to full MHA.",
      calc: [
        "k_proj = h × (n_kv_heads × head_dim) = 4,096 × 1,024 = 4,194,304",
        "v_proj = h × (n_kv_heads × head_dim) = 4,096 × 1,024 = 4,194,304",
        "",
        "With MHA both would be 16,777,216 — 4× larger.",
      ],
      result: "8,388,608 K+V params per block",
    },
    {
      key: "head_dim",
      desc: "Explicit per-head dimension (128). In older HF configs it was derived from <code>h / n_head</code>; Llama 3.1 stores it directly so decoupled configurations are allowed.",
    },
    {
      key: "max_position_embeddings",
      desc: "Maximum context length (131,072 tokens). This is a <strong>runtime</strong> limit — not a parameter. Llama uses RoPE, so there is no learned table of this size.",
    },
    {
      key: "rope_theta",
      desc: "Base frequency for the Rotary Positional Embedding rotations. Higher <code>θ</code> → longer wavelengths → better long-context behavior. Pure hyperparameter; 0 learned weights.",
    },
    {
      key: "rope_scaling",
      desc: "Llama 3 extends the RoPE positions to reach 131K context. <code>\"rope_type\": \"llama3\"</code> uses a piecewise-interpolated frequency stretch. Runtime-only; no learned weights.",
    },
    {
      key: "rms_norm_eps",
      desc: "Epsilon added inside RMSNorm for numerical stability. Hyperparameter, not a trainable parameter.",
    },
    {
      key: "hidden_act",
      desc: "Activation used inside the MLP's SwiGLU gate. <code>\"silu\"</code> is the Sigmoid-Linear Unit (also called Swish). Element-wise — 0 parameters.",
    },
    {
      key: "tie_word_embeddings",
      affects: true,
      desc: "<strong>False</strong> for Llama 3.1 8B — the LM head is a <strong>separate</strong> matrix of shape <code>[h, V]</code>, not shared with the input embedding. That's another full <code>V × h</code> on top of <code>embed_tokens</code>.",
      calc: [
        "lm_head = h × V = 4,096 × 128,256 = 525,336,576",
      ],
      result: "+525,336,576 extra params",
    },
    {
      key: "attention_bias",
      affects: true,
      desc: "<code>false</code> — Q, K, V, and O linears have <strong>no bias vectors</strong>. Saves <code>4h</code> params per layer vs GPT-2.",
      calc: [
        "If true: 4 × h × L = 4 × 4,096 × 32 = 524,288",
        "If false (Llama): + 0",
      ],
      result: "Saves 524,288 params",
    },
    {
      key: "mlp_bias",
      affects: true,
      desc: "<code>false</code> — MLP linears (gate, up, down) have no bias either.",
      calc: [
        "If true: (2 × i + h) × L = 32,768 × 32 ≈ 1 M",
        "If false (Llama): + 0",
      ],
      result: "Saves ~1 M params",
    },
    {
      key: "bos_token_id",
      desc: "Token ID the tokenizer uses for begin-of-sequence (<code>128000</code>). Llama 3 uses dedicated BOS and EOS IDs.",
    },
    {
      key: "eos_token_id",
      desc: "Token ID for end-of-sequence (<code>128001</code>). Chat-tuned variants add more end-of-turn tokens.",
    },
    {
      key: "use_cache",
      desc: "Whether the model returns KV cache tensors for reuse across generation steps. Inference-time knob; affects memory, not parameter count.",
    },
  ];

  const FIELDS_DEEPSEEK = [
    {
      key: "architectures",
      desc: "Python class HF Transformers instantiates when loading the model: <code>DeepseekV2ForCausalLM</code>. Ships with custom modeling code (<code>trust_remote_code=True</code>).",
    },
    {
      key: "model_type",
      desc: "<code>\"deepseek_v2\"</code> — DeepSeek's custom MLA + MoE architecture. Not covered by HF's built-in model types.",
    },
    {
      key: "vocab_size",
      affects: true,
      desc: "Vocabulary size (<code>V</code> = 102,400). Used for <code>embed_tokens</code> and (separately, since <code>tie_word_embeddings</code> is false) <code>lm_head</code>.",
      calc: [
        "embed_tokens = V × h = 102,400 × 2,048 = 209,715,200",
        "lm_head       = V × h                     = 209,715,200",
      ],
      result: "419,430,400 params total",
    },
    {
      key: "hidden_size",
      affects: true,
      desc: "Residual-stream width (<code>h</code> = 2,048). Every projection, RMSNorm vector, and the embedding are sized against this. Smaller than Llama's 4,096.",
      result: "h = 2,048",
    },
    {
      key: "num_hidden_layers",
      affects: true,
      desc: "Total transformer blocks (<code>L</code> = 27). The first <code>first_k_dense_replace</code>=1 is dense; the remaining 26 are MoE.",
    },
    {
      key: "num_attention_heads",
      desc: "Number of attention heads (<code>n_head</code> = 16). Each head carries a 192-dim slice of Q (128 non-RoPE + 64 RoPE).",
    },
    {
      key: "q_lora_rank",
      affects: true,
      desc: "Low-rank bottleneck for Q (null in DeepSeek-V2 Lite — Q is projected directly). The full DeepSeek-V2 sets this to 1,536 to shrink Q as well. Setting it saves memory and compute but Lite skips the extra latent path to stay small.",
      result: "null → Q uses direct h × (n_head × qk_head_dim) projection",
    },
    {
      key: "kv_lora_rank",
      affects: true,
      desc: "<strong>MLA's key knob.</strong> K and V are projected through a shared low-rank latent of this size (512). Only this latent (+ a small RoPE sidecar) is kept in the KV cache, independently of <code>n_head</code>.",
      calc: [
        "kv_a_proj : h × (kv_lora_rank + qk_rope_head_dim)",
        "           = 2,048 × (512 + 64) = 1,179,648",
        "kv_b_proj : kv_lora_rank × n_head × (qk_nope + v_head_dim)",
        "           = 512 × 16 × (128 + 128) = 2,097,152",
      ],
      result: "3,276,800 K/V params per block (vs ~16 M in Llama's GQA)",
    },
    {
      key: "qk_nope_head_dim",
      desc: "Size of the non-RoPE part of Q and K per head (128). RoPE is only applied to the remaining <code>qk_rope_head_dim</code> slice. DeepSeek splits attention channels so position information rides on just part of the vector.",
    },
    {
      key: "qk_rope_head_dim",
      desc: "Size of the RoPE-rotated part per head (64). This is the only K-slice that depends on position; it's shared across heads via the decoupled RoPE-K path, and it's cached.",
    },
    {
      key: "v_head_dim",
      desc: "Per-head V dimension (128). Only used during the forward pass — V is reconstructed from the compressed latent and discarded; it's not cached separately in MLA.",
    },
    {
      key: "first_k_dense_replace",
      affects: true,
      desc: "The first <code>k</code> layers use a dense SwiGLU MLP instead of MoE. Setting this to 1 means <strong>layer 0 is dense</strong>; layers 1..26 are MoE. A common pattern: early layers see more diverse tokens and benefit from undifferentiated capacity.",
      calc: [
        "Dense layer 0 MLP: 3 × h × intermediate_size",
        "                 = 3 × 2,048 × 10,944 = 67,239,936",
      ],
      result: "+67,239,936 params for layer 0's MLP",
    },
    {
      key: "intermediate_size",
      affects: true,
      desc: "Intermediate size for the <strong>dense</strong> MLP in layer 0 (10,944). Only used in the first <code>first_k_dense_replace</code> layers.",
    },
    {
      key: "moe_intermediate_size",
      affects: true,
      desc: "Intermediate size per <strong>MoE expert</strong> (1,408). Each routed or shared expert is a small SwiGLU with this width. Keeping it small is what makes DeepSeekMoE \"fine-grained\": many small experts instead of a few large ones.",
      calc: [
        "Per expert: 3 × h × moe_i = 3 × 2,048 × 1,408 = 8,650,752",
        "64 routed experts: 553,648,128   (64 × 8,650,752)",
        "2 shared experts:   17,301,504   (2 × 8,650,752)",
      ],
      result: "570,949,632 expert-MLP params per MoE layer",
    },
    {
      key: "n_routed_experts",
      affects: true,
      desc: "Total routed experts per MoE layer (64). Only <code>num_experts_per_tok</code>=6 of these are active for any given token. All 64 must still fit in VRAM.",
    },
    {
      key: "n_shared_experts",
      affects: true,
      desc: "Shared experts per MoE layer (2). These are <strong>always active</strong> — bypass the router. They act like a residual \"base\" MLP alongside the routed experts.",
    },
    {
      key: "num_experts_per_tok",
      affects: true,
      desc: "Top-<code>k</code> routing: for each token, the router picks this many routed experts (6 of 64). Combined with the 2 shared experts, that's <strong>8 active experts per token per MoE layer</strong>.",
      calc: [
        "Active expert MLP per token per MoE layer:",
        "  (top_k + n_shared) × 3 × h × moe_i",
        "= (6 + 2) × 3 × 2,048 × 1,408 = 69,206,016",
      ],
      result: "+69,206,016 active per MoE layer per token",
    },
    {
      key: "moe_layer_freq",
      desc: "How often MoE layers appear after <code>first_k_dense_replace</code> (1 = every layer). DeepSeek-V2 Lite makes every post-layer-0 block MoE.",
    },
    {
      key: "topk_method",
      desc: "<code>\"greedy\"</code> — the router simply picks the top-k experts by score. Alternative is <code>\"group_limited_greedy\"</code> (used by the full DeepSeek-V2) which constrains picks per expert-group to help cross-GPU load-balancing.",
    },
    {
      key: "scoring_func",
      desc: "<code>\"softmax\"</code> — the router applies softmax over 64 expert logits to get routing weights, then selects the top-k. Those weights are later used in the weighted sum over chosen experts.",
    },
    {
      key: "aux_loss_alpha",
      desc: "Weight of the auxiliary load-balancing loss added during training. Encourages roughly equal expert utilization. Pure training-time knob; not a parameter.",
    },
    {
      key: "max_position_embeddings",
      desc: "Deployed context window (163,840 tokens). The model was pretrained at 4K and extended to 160K via YaRN RoPE scaling. Not a parameter.",
    },
    {
      key: "rope_theta",
      desc: "Base RoPE frequency (10,000 — the classic value). Combined with YaRN, this stretches to the full 160K context.",
    },
    {
      key: "rope_scaling",
      desc: "YaRN RoPE extension: <code>factor</code>=40 means scale from the pretraining 4K to 160K. <code>mscale</code> adjusts softmax temperature so attention scores stay well-calibrated at long context. No learned weights.",
    },
    {
      key: "rms_norm_eps",
      desc: "Epsilon inside RMSNorm (<code>1e-06</code>). Smaller than Llama's <code>1e-05</code>. Hyperparameter.",
    },
    {
      key: "hidden_act",
      desc: "Activation in SwiGLU: <code>\"silu\"</code> (Swish). Same as Llama. 0 parameters.",
    },
    {
      key: "tie_word_embeddings",
      affects: true,
      desc: "<code>false</code> — <code>lm_head</code> is a separate matrix, adding another <code>V × h</code>=209,715,200 params on top of <code>embed_tokens</code>.",
    },
    {
      key: "attention_bias",
      desc: "<code>false</code> — no biases on the attention projections (q_proj, kv_a_proj, kv_b_proj, o_proj).",
    },
    {
      key: "bos_token_id",
      desc: "Begin-of-sequence token ID (100,000).",
    },
    {
      key: "eos_token_id",
      desc: "End-of-sequence token ID (100,001).",
    },
    {
      key: "use_cache",
      desc: "Whether to return KV cache tensors. For MLA, the cache stores the compressed latent + decoupled RoPE K — far smaller than MHA's per-head K + V.",
    },
  ];

  const MODELS = { gpt2: FIELDS_GPT2, llama: FIELDS_LLAMA, deepseek: FIELDS_DEEPSEEK };
  const CARDS_CONTAINER_ID = { gpt2: "cards-gpt2", llama: "cards-llama", deepseek: "cards-deepseek" };

  /* -------- Render field cards for each model -------- */
  function renderCards(modelKey) {
    const container = document.getElementById(CARDS_CONTAINER_ID[modelKey]);
    if (!container) return;
    const fields = MODELS[modelKey];
    container.innerHTML = fields.map((f) => {
      const calc = f.calc ? `
        <div class="calc">${f.calc.join("\n")}${f.result ? `<span class="result">= ${f.result}</span>` : ""}</div>
      ` : "";
      const tag = f.affects
        ? `<span class="tag affects">affects params</span>`
        : `<span class="tag">metadata</span>`;
      return `
        <div class="card" data-field="${f.key}" id="card-${modelKey}-${f.key}">
          <div class="card-head">
            <span class="k">${f.key}</span>
            ${tag}
          </div>
          <div class="desc">${f.desc}</div>
          ${calc}
        </div>
      `;
    }).join("");
  }

  /* -------- Click-to-highlight within a single view -------- */
  function wireView(modelKey) {
    const view = document.getElementById("view-" + modelKey);
    if (!view) return;
    const config = view.querySelector(".config");
    const cards = view.querySelector(".cards");
    if (!config || !cards) return;

    function clearActive() {
      view.querySelectorAll(".cfg-row.active, .card.active").forEach((el) => el.classList.remove("active"));
    }

    config.addEventListener("click", (e) => {
      const row = e.target.closest(".cfg-row");
      if (!row || !row.dataset.field) return;
      clearActive();
      row.classList.add("active");
      const card = document.getElementById(`card-${modelKey}-${row.dataset.field}`);
      if (card) {
        card.classList.add("active");
        card.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    });

    cards.addEventListener("click", (e) => {
      const card = e.target.closest(".card");
      if (!card || !card.dataset.field) return;
      clearActive();
      card.classList.add("active");
      const row = config.querySelector(`.cfg-row[data-field="${card.dataset.field}"]`);
      if (row) {
        row.classList.add("active");
        row.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    });
  }

  /* -------- Model picker -------- */
  function switchModel(modelKey) {
    document.querySelectorAll(".model-view").forEach((v) => {
      const show = v.id === "view-" + modelKey;
      if (show) v.removeAttribute("hidden");
      else v.setAttribute("hidden", "");
    });
    document.querySelectorAll(".model-picker button").forEach((b) => {
      b.classList.toggle("active", b.dataset.model === modelKey);
    });
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function initPicker() {
    document.querySelectorAll(".model-picker button").forEach((b) => {
      b.addEventListener("click", () => switchModel(b.dataset.model));
    });
  }

  /* -------- Init -------- */
  Object.keys(MODELS).forEach((k) => {
    renderCards(k);
    wireView(k);
  });
  initPicker();
})();
