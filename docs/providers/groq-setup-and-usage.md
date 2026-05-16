# Groq Setup & Usage

Groq provides ultra-fast inference for open-source Llama models via an
OpenAI-compatible API. A free tier with rate limits is available.

## Getting Started

### 1. Get an API key

Sign up at https://console.groq.com and create an API key.

### 2. Set the environment variable

```bash
export GROQ_API_KEY="gsk_..."
```

Or add it to your `.env` file:

```
GROQ_API_KEY=gsk_...
```

### 3. Use it in code

```python
from tradingagents.llm_clients.factory import create_llm_client

client = create_llm_client("groq", "llama-3.3-70b-versatile")
llm = client.get_llm()

response = llm.invoke([("human", "Summarize the merger arbitrage strategy.")])
print(response.content)
```

## CLI

The CLI offers Groq in the provider list ("Groq (Free tier — Llama 3.3 / Llama 4)"),
with model choices defined in `model_catalog.py`.

## Models

### Quick mode (lower latency)

| Model ID | Notes |
|----------|-------|
| `llama-3.1-8b-instant` | Fastest, cheapest |
| `llama-3.3-70b-specdec` | 70B with speculative decoding |
| `meta-llama/llama-4-scout-17b-16e-instruct` | Llama 4 Scout |

### Deep mode (higher accuracy)

| Model ID | Notes |
|----------|-------|
| `llama-3.3-70b-versatile` | Recommended default |
| `meta-llama/llama-4-maverick-17b-128e-instruct` | Llama 4 Maverick |
| `llama-3.1-70b-versatile` | Previous-gen flagship |

The full list lives in
[`tradingagents/llm_clients/model_catalog.py`](../../tradingagents/llm_clients/model_catalog.py).

## Feature Support

See [capability-matrix.md](capability-matrix.md) for a per-feature table.
At a glance: tools and JSON mode are supported; JSON schema is not.

## Free Tier Rate Limits

Groq publishes per-model rate limits at
https://console.groq.com/docs/rate-limits. As of writing, free-tier
accounts have ~30 RPM and a per-minute token budget that varies by
model. Production workloads should upgrade to a paid tier.

If you hit rate limits, the OpenAI SDK will surface a 429 error
through langchain. Wait 60 seconds, switch to a smaller model
(`llama-3.1-8b-instant`), or upgrade your Groq account.

## Custom endpoint

You can route Groq through a proxy by passing `base_url`:

```python
client = create_llm_client(
    "groq",
    "llama-3.3-70b-versatile",
    base_url="https://your-proxy.example.com/v1",
)
```

The default endpoint is `https://api.groq.com/openai/v1`.

## Troubleshooting

See [troubleshooting.md](troubleshooting.md).
