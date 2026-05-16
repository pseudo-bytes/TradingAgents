# Troubleshooting

## "API key for provider 'groq' is not set"

`OpenAIClient.get_llm()` raises this when `GROQ_API_KEY` is missing.

Fix: set the env var in your shell or `.env`.

```bash
export GROQ_API_KEY="gsk_..."
```

Verify:

```bash
python -c "import os; print('set' if os.environ.get('GROQ_API_KEY') else 'missing')"
```

## "Unsupported LLM provider: ..."

`create_llm_client()` raises `ValueError` when the provider name isn't
in the registry. Valid names live in
[`provider_registry.py`](../../tradingagents/llm_clients/provider_registry.py).

Common slips: `"Groq"` vs `"groq"` (the factory lowercases, so both
work, but configuration files sometimes don't).

## 429 rate limit errors from Groq

The free tier has tight per-minute limits. Options:

- Wait and retry.
- Switch to `llama-3.1-8b-instant` (looser limits).
- Upgrade the Groq account.

## Tool calls aren't returned

If `response.tool_calls` is empty after binding tools:

- Check the [capability matrix](capability-matrix.md) — does the model
  support tools?
- Confirm the tool schema is well-formed (the OpenAI tool-spec shape).
- For DeepSeek thinking / MiniMax M2.x: the client suppresses
  `tool_choice` automatically; the schema is still bound as a tool.

## JSON mode returns text that isn't valid JSON

Llama-family models sometimes hallucinate JSON syntax. Mitigations:

- Use a stricter prompt ("Respond ONLY with a JSON object…").
- Wrap parsing in `try/except json.JSONDecodeError` and retry once.
- If your provider supports `json_schema` (OpenAI, Google), prefer
  that over `json_object`.

## Custom Ollama model fails

`OLLAMA_BASE_URL` controls where the client connects. If you're
running ollama-serve on a remote host:

```bash
export OLLAMA_BASE_URL="http://10.0.0.5:11434/v1"
```

The CLI surfaces the resolved endpoint after provider selection
(via `cli/utils.confirm_ollama_endpoint`).

## "Model 'X' is not in the known model list..."

A `RuntimeWarning`, not an error. The call proceeds. To silence,
add the model ID to the relevant entry in `model_catalog.py`.
