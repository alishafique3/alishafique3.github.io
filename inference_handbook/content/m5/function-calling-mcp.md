<div class="chapter-meta">Module 5 · Chapter 6</div>

# Function Calling, Tool Use, and the Model Context Protocol (MCP)

<p class="lede">Function calling lets an LLM invoke external code — database queries, web searches, calculators — mid-generation. It is the foundation of every agent, RAG pipeline, and copilot. This chapter covers the OpenAI/Anthropic tool-call schema, parallel tool calls, and the Model Context Protocol (MCP) that is becoming the universal tool-server standard.</p>

## The function-calling loop

```
1. Client sends messages + tool schemas to the LLM.
2. LLM returns either a text answer OR a structured tool_call.
3. Client executes the tool with the provided arguments.
4. Client appends the tool result as a message with role "tool".
5. Client re-calls the LLM with the extended conversation.
6. Loop until the LLM returns a final text answer.
```

The model doesn't run your code — it asks you to. The client is always in control.

## Defining a tool

OpenAI schema:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["c", "f"]}
                },
                "required": ["city"]
            }
        }
    }
]
```

The `parameters` field is a standard JSON Schema. The model uses it to emit calls with the right shape.

## Invoking tools

```python
msgs = [{"role": "user", "content": "What's the weather in Lahore?"}]
r = client.chat.completions.create(model="m", messages=msgs, tools=tools)

tool_calls = r.choices[0].message.tool_calls
if tool_calls:
    for tc in tool_calls:
        name = tc.function.name
        args = json.loads(tc.function.arguments)
        result = run_tool(name, args)        # your code
        msgs.append(r.choices[0].message)    # record the tool_call
        msgs.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": json.dumps(result)
        })
    # Second call with tool result
    r2 = client.chat.completions.create(model="m", messages=msgs, tools=tools)
    print(r2.choices[0].message.content)
```

## Parallel tool calls

Modern models can emit multiple tool_calls in one response. Example — "Weather in Lahore and Karachi?" produces two tool_calls at once; the client runs both (ideally concurrently) and sends both results back.

OpenAI, Anthropic, and most open-weight tuned models (Llama 3.1+, Qwen 2.5, etc.) support parallel tool calls. Flag per request to enable/disable.

## The Model Context Protocol (MCP)

Every app wires its own tools differently — a maintenance nightmare. **MCP** is an open protocol (Anthropic-published, supported by OpenAI, Google, and the wider ecosystem) that standardizes how LLM clients discover and call tools exposed by external servers.

The shape:

- An **MCP server** advertises tools (and resources, prompts) over a simple JSON-RPC stream.
- An **MCP client** (Claude Desktop, VS Code Agent, Cursor, Claude Code) connects to any MCP server and auto-registers its tools.
- The LLM calls tools via the standard function-calling API; the client translates to MCP calls.

Result: one well-designed MCP server (Slack, Gmail, your internal DB) lights up every MCP-capable LLM client.

## Minimal MCP server (Python)

```python
# pip install mcp-server
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather-server")

@mcp.tool()
def get_weather(city: str, unit: str = "c") -> dict:
    """Current weather for a city."""
    return {"city": city, "temp": 24, "unit": unit}

if __name__ == "__main__":
    mcp.run()
```

Any MCP client can now connect and expose `get_weather` as a tool to its LLM.

## Serving-side support

Function calling is purely a prompting convention — any instruction-tuned model that was fine-tuned with tool-use data can do it. In practice, model-specific chat templates include special tokens for `tool_call` and `tool_result`. Serving engines (vLLM, SGLang, TGI) handle the formatting when you use the chat API with `tools=`.

Not every model is equal at tool use. Benchmarks on BFCL (Berkeley Function Calling Leaderboard) in 2026:

- GPT-4o: ~90% accuracy
- Claude Sonnet: ~90%
- Qwen 2.5 72B: ~85%
- Llama 3.3 70B: ~85%
- Llama 3.1 8B: ~65%
- Qwen 2.5 7B: ~70%

Pick a model that's been tool-use-tuned. Not all open-weight small models are.

## Pitfalls

- **Expecting single-call semantics with parallel calls on.** Code that handles only `tool_calls[0]` silently drops the rest.
- **Not giving tools good descriptions.** The model picks tools by reading descriptions. "Get weather data" beats "Query the API".
- **Exposing dangerous tools.** `run_shell(cmd)` is tempting. Don't. Give the model narrow, auditable operations.
- **Infinite tool loops.** A buggy prompt can send the model into "call tool → call tool → ..." forever. Cap the iteration count.

<section class="faq">
<h2>FAQ</h2>
<h3>What's the difference between function calling and tool use?</h3>
<p>Mostly naming. OpenAI changed "functions" to "tools" in 2023 to generalize. Semantics are identical.</p>
<h3>Can the LLM execute code directly?</h3>
<p>No. The LLM only emits a tool_call. Your client decides whether and how to run it.</p>
<h3>Is MCP the only standard?</h3>
<p>The dominant one in 2026. Older proprietary protocols (OpenAI Assistants API, Google ADK) exist but MCP has cross-vendor adoption.</p>
<h3>Do I need separate models for tool use and chat?</h3>
<p>Modern instruction-tuned models handle both. Use a tool-use-tuned variant if BFCL accuracy matters to you.</p>
</section>

<section>
<h2>References</h2>
<ul class="refs">
<li><a href="https://platform.openai.com/docs/guides/function-calling" target="_blank" rel="noreferrer">OpenAI — Function calling documentation</a></li>
<li><a href="https://docs.anthropic.com/en/docs/build-with-claude/tool-use" target="_blank" rel="noreferrer">Anthropic — Tool use documentation</a></li>
<li><a href="https://modelcontextprotocol.io/" target="_blank" rel="noreferrer">Model Context Protocol — Official site</a></li>
<li><a href="https://gorilla.cs.berkeley.edu/leaderboard.html" target="_blank" rel="noreferrer">Berkeley Function Calling Leaderboard (BFCL)</a></li>
</ul>
</section>

## What's next

[Chapter 7 — Agent Harnesses](?module=m5&chapter=agent-harnesses) covers the orchestration loop that strings many tool calls together into a multi-step workflow.
