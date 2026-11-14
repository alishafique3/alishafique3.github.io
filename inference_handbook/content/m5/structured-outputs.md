<div class="chapter-meta">Module 5 · Chapter 5</div>

# Structured Outputs and Constrained Generation

<p class="lede">Downstream systems want structured data — JSON, YAML, XML, SQL — not free-form prose. The naive "please output JSON" approach fails 2–10 % of the time. <strong>Constrained generation</strong> fixes that: the sampler is forced to produce only tokens that respect a schema or grammar. The quality tax is small; the reliability win is large.</p>

## Three levels of "structured"

1. **JSON mode** — output is guaranteed to be valid JSON (parseable). No schema enforcement; field names/types are up to the model.
2. **JSON schema** — output conforms to a specific JSON Schema. Field names and types are guaranteed.
3. **Full grammar** — output conforms to an arbitrary context-free grammar (BNF, regex, Lark). Covers JSON, YAML, SQL, Python ASTs, etc.

Each level adds more constraint and more serving overhead (though modern implementations are surprisingly fast).

## JSON mode

OpenAI-style:

```python
r = client.chat.completions.create(
    model="m",
    messages=[{"role": "user", "content": "Return a JSON with a 'city' field"}],
    response_format={"type": "json_object"},
)
```

The sampler is restricted to tokens that keep the output a valid JSON prefix. The result parses. The *contents* are still up to the model — you might get `{"city": "Paris"}` or `{"town": "Paris"}` depending on how the prompt frames it.

## JSON schema — structured outputs

Modern OpenAI, vLLM, and SGLang all support strict JSON schema:

```python
schema = {
  "type": "object",
  "properties": {
    "city":    {"type": "string"},
    "country": {"type": "string"},
    "population": {"type": "integer"}
  },
  "required": ["city", "country", "population"],
  "additionalProperties": False
}

r = client.chat.completions.create(
    model="m",
    messages=[...],
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "city_info", "schema": schema, "strict": True}
    }
)
```

The output is guaranteed to conform. Required fields will be present; extra fields will not appear; types match.

## Pydantic — the Python-friendly path

```python
from pydantic import BaseModel
from openai import OpenAI

class CityInfo(BaseModel):
    city: str
    country: str
    population: int

client = OpenAI(base_url="...")
r = client.chat.completions.parse(
    model="m",
    messages=[...],
    response_format=CityInfo,
)
city = r.choices[0].message.parsed    # typed CityInfo instance
```

The SDK converts the Pydantic model into a JSON schema and parses the response. No manual schema maintenance.

## Grammar-constrained — beyond JSON

For SQL, structured code, or domain-specific formats, use grammar constraint libraries:

- **XGrammar** — fast, integrated into vLLM.
- **Outlines** — Python library, works with many engines, supports regex + CFG.
- **Guidance** — Microsoft's library for templated structured generation.

Example — constrain to a simple numeric list:

```python
# Outlines-style
from outlines import generate, models

m = models.transformers("Qwen/Qwen2.5-7B-Instruct")
regex = r'\[\d+(,\d+)*\]'
generator = generate.regex(m, regex)
print(generator("List first five primes:"))   # → "[2,3,5,7,11]"
```

## How constraints work (one sentence)

At every decode step, the engine computes the set of tokens that would keep the output valid under the constraint, masks all other logits to $-\infty$, then samples. Lookups are precomputed into a trie or FSM, so the per-token cost is small — typically &lt; 5% TPOT overhead with modern implementations.

## Serving overhead

| Approach | TPOT overhead | Setup cost |
|---|---|---|
| JSON mode | &lt; 1% | Negligible |
| JSON schema | 2–5% | Schema compilation once per schema |
| XGrammar / Outlines CFG | 3–10% | Grammar compilation once |

In vLLM, enable with `--guided-decoding-backend=xgrammar`. In SGLang, grammar is built in.

## When structured output isn't enough

Constrained generation guarantees syntax, not semantics. The model can still output:

- Plausible-looking but wrong values (`"population": 999999999`).
- Correct JSON that fails your business logic (`"email": "notanemail"`).

Validation downstream is still mandatory. Constraints just remove the parse-error class of failure.

## Pitfalls

- **Over-constraining.** Very strict grammars can force the model into low-probability regions and hurt content quality. Validate end-to-end.
- **Relying on "just ask for JSON".** Without constraint, failure rate is 1–5% even on well-tuned models. Use real constraints in prod.
- **Mixing streaming with strict JSON.** Partial JSON mid-stream is not valid JSON. Buffer or render tokens as placeholders until the stream closes.
- **Forgetting to wire the schema through the SDK.** The Pydantic helper handles this; manual callers must pass the schema in the `response_format` field correctly.

<section class="faq">
<h2>FAQ</h2>
<h3>Does structured output slow down generation?</h3>
<p>Typically 3–10% TPOT overhead on modern implementations. Worth every millisecond for the reliability win.</p>
<h3>Can constrained generation produce invalid-but-plausible values?</h3>
<p>Yes. The constraint enforces syntax, not business rules. Validate downstream.</p>
<h3>What about XML?</h3>
<p>Possible with a grammar backend, but JSON is dramatically better supported across all engines. Use JSON unless the consumer is XML-only.</p>
<h3>Can I enforce a schema on a streaming response?</h3>
<p>Yes — the constraint works per-token, so the stream is valid at every point (a JSON prefix is always valid). But clients can't parse a partial JSON until it closes.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://platform.openai.com/docs/guides/structured-outputs" target="_blank" rel="noreferrer">OpenAI — Structured Outputs documentation</a></li>
<li><a href="https://github.com/mlc-ai/xgrammar" target="_blank" rel="noreferrer">XGrammar — GitHub</a></li>
<li><a href="https://github.com/dottxt-ai/outlines" target="_blank" rel="noreferrer">Outlines — GitHub</a></li>
<li><a href="https://docs.vllm.ai/en/stable/features/structured_outputs.html" target="_blank" rel="noreferrer">vLLM — Structured Outputs documentation</a></li>
</ul>
</section>

## What's next

Function calling ([Chapter 6](?module=m5&chapter=function-calling-mcp)) uses structured outputs under the hood — the model emits a tool call as a JSON object matching a schema.
