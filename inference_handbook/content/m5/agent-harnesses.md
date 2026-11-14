<div class="chapter-meta">Module 5 · Chapter 7</div>

# Agent Harnesses and Multi-Step Workflows

<p class="lede">An agent is an LLM in a loop with tools, memory, and a stopping condition. The harness is the code that runs the loop. For an inference engineer, agents are the most expensive, least predictable workload you will see — one user request becomes 10–100 LLM calls. This chapter is the honest picture of the engineering and the inference load.</p>

## The agent loop in pseudocode

```python
def agent(user_msg, tools, max_steps=20):
    messages = [{"role": "user", "content": user_msg}]
    for step in range(max_steps):
        r = llm(messages, tools=tools)
        messages.append(r.message)
        if not r.tool_calls:
            return r.message.content         # done
        for tc in r.tool_calls:
            result = run_tool(tc.name, tc.args)
            messages.append({"role": "tool",
                             "tool_call_id": tc.id,
                             "content": result})
    return "exceeded max_steps"
```

This is it. Every production agent — AutoGPT, CrewAI, LangGraph, your own — is a variation on this loop.

## Why the loop is expensive

- **Each step is a full LLM call.** Prefill of the whole conversation history + decode of the next reasoning + tool call.
- **History grows.** After 10 steps the context is 10 × longer than step 1. Prefill cost grows super-linearly.
- **Tools are slow.** A web search or DB query adds hundreds of milliseconds per step.

A simple "find the answer" query can easily hit 20–40 LLM calls and 30–60 seconds of total wall time. The LLM inference bill scales with steps, not user requests.

## ReAct — the most common pattern

[ReAct](https://arxiv.org/abs/2210.03629) (Reason + Act) interleaves reasoning ("Thought:") and actions ("Action:"):

```
Thought: I need to find the current weather in Lahore.
Action: get_weather(city="Lahore")
Observation: {"temp": 28, "unit": "c"}
Thought: User asked if it's warm. 28C is warm.
Final answer: Yes, 28°C counts as warm.
```

Modern tool-tuned LLMs internalize ReAct — you don't have to template it yourself; the model emits reasoning + tool_calls natively.

## Memory beyond the conversation

Agents need to remember across sessions, not just within one. Patterns:

- **Vector memory.** Store past interactions as embeddings; retrieve top-k relevant ones at each step. Adds a retrieval step per LLM call.
- **Scratchpad / long-term state.** Persist key facts in a KV store the agent can read and write. Keeps context small.
- **Graph memory.** For entity-heavy domains, store relationships in a graph DB; query via a tool.

The right pattern depends on domain. A customer-support agent needs vector memory of past tickets. A code agent needs file-system awareness. A trading agent needs real-time market data.

## Inference-side implications

For an inference platform serving agents, three things dominate capacity planning:

1. **Prefill dominates.** The conversation history fills the context on every step. Prefix caching (M6.3) is critical — if your agent harness places variable content at the top of the prompt, caching is defeated and prefill cost multiplies.
2. **Latency budget per step is tight.** Users accept 10 s for an agent answer if it's clearly thinking. They don't accept 30 s. Each step should be &lt; 1 s if possible.
3. **Concurrency is weird.** One user request produces bursts of 20+ sequential LLM calls. Your concurrency metric is not "RPS" but "active agent sessions".

## A realistic agent-workload sizing

Scenario: a customer-support agent, 5-step average, Llama 3.1 70B FP8 on 2 × H100, prefix caching on, mean prompt 300 tokens, mean growth 200 tokens per step.

- Step 1: prefill 300, decode ~150. Cost ~0.5s.
- Step 5: prefill 1,100 (mostly cached), decode ~150. Cost ~0.4s.
- Total per user: 5 × ~0.5s = 2.5 s + tool time.

At 100 concurrent agent sessions, you need ~100 effective in-flight requests at any moment. For a fleet, that's ~5-8 replicas (M3.4) — more than you'd expect for 100 "users" because each one is really 5 LLM calls in flight sequentially.

## Pitfalls

- **Unbounded loops.** `max_steps` is not optional. Every agent harness must cap it, otherwise a confused model loops forever.
- **No tool retry logic.** Tools fail. Transient failures should be retryable automatically. Permanent failures should surface to the LLM as error observations so it can adapt.
- **Shared context across users.** Agents with persistent memory must scope memory per user/session. Mixing memories is both a privacy bug and a correctness bug.
- **Logging every step as a separate request.** You'll have 100× the log volume of a simple chat. Plan for it.
- **Testing on happy paths only.** Agent regressions appear in unusual multi-step trajectories. Have an eval set that tracks end-to-end success, not just single-step accuracy.

<section class="faq">
<h2>FAQ</h2>
<h3>Which agent framework should I use?</h3>
<p>For production: LangGraph, CrewAI, or roll your own (the pattern is 50 lines of Python). For research: lots of options. Avoid hype-driven frameworks that abstract away the LLM call.</p>
<h3>What's "agentic" vs. "chatbot"?</h3>
<p>Agents take actions (call tools) in a loop. Chatbots answer questions in a single turn. The line is fuzzy — modern chat often includes agentic behavior.</p>
<h3>Does streaming work for agent loops?</h3>
<p>Stream the final answer; intermediate steps are usually buffered. Users want to see the final response token-by-token; they don't need to see intermediate reasoning streamed.</p>
<h3>How do I handle failures?</h3>
<p>Catch exceptions around each tool call. Feed the error back to the LLM as an observation. The model is usually smart enough to try a different approach.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://arxiv.org/abs/2210.03629" target="_blank" rel="noreferrer">Yao et al. — ReAct: Synergizing Reasoning and Acting in Language Models</a></li>
<li><a href="https://langchain-ai.github.io/langgraph/" target="_blank" rel="noreferrer">LangGraph — Agent orchestration framework</a></li>
<li><a href="https://docs.crewai.com/" target="_blank" rel="noreferrer">CrewAI — Multi-agent framework</a></li>
<li><a href="https://modelcontextprotocol.io/" target="_blank" rel="noreferrer">MCP — Model Context Protocol (tool server standard)</a></li>
</ul>
</section>

## What's next

Module 5 is done. [Module 6 — Inference Infrastructure Optimization](?module=m6&chapter=engine-anatomy) covers how the serving engine itself handles the concurrency and prefix patterns that agents generate.
