# Provider Capability Matrix

What each provider/model accepts at the API layer.

| Provider | Representative model | Tools | tool_choice | JSON mode | JSON schema | Streaming | Async |
|----------|----------------------|:-----:|:-----------:|:---------:|:-----------:|:---------:|:-----:|
| Groq | llama-3.3-70b-versatile | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| Groq | llama-3.1-8b-instant | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| Groq | meta-llama/llama-4-* | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| OpenAI | gpt-5.5 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Anthropic | claude-sonnet-4-6 | ✅ | ✅ | n/a | n/a | ✅ | ✅ |
| Google | gemini-2.5-pro | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| DeepSeek | deepseek-v4-pro | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ |
| MiniMax | MiniMax-M2.7 | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Ollama | (any) | ✅ | varies | varies | ❌ | ✅ | ✅ |

Legend: ✅ supported · ❌ not supported · n/a not applicable.

The authoritative declaration lives in
[`tradingagents/llm_clients/capabilities.py`](../../tradingagents/llm_clients/capabilities.py).
This table is for human reference and may lag the code.

## What each column means

- **Tools** — model accepts a `tools=[...]` array of function definitions.
- **tool_choice** — model accepts the `tool_choice` parameter (`"auto"`,
  `"required"`, or a specific function spec). When false, langchain still
  binds the schema as a tool but must omit `tool_choice`.
- **JSON mode** — `response_format={"type": "json_object"}` is honored.
- **JSON schema** — `response_format={"type": "json_schema", "schema": {...}}`
  is honored (stricter; usually OpenAI-only).
- **Streaming / Async** — `stream()`, `astream()`, `ainvoke()` work.
