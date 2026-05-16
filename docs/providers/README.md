# LLM Provider Documentation

Guides for working with LLM providers in TradingAgents.

## Quick Links

- [Architecture](architecture.md) — How the provider system works
- [Adding a Provider](adding-a-provider.md) — Step-by-step guide
- [Groq Setup](groq-setup-and-usage.md) — Groq configuration and models
- [Capability Matrix](capability-matrix.md) — Feature support per provider
- [Troubleshooting](troubleshooting.md) — Common issues

## Supported Providers

| Provider | Native or OpenAI-compatible | Notes |
|----------|----------------------------|-------|
| OpenAI | Native (Responses API) | GPT-4, GPT-5 families |
| Anthropic | Native | Claude Opus/Sonnet/Haiku |
| Google | Native | Gemini 2.5 / 3 |
| Azure OpenAI | Native | Azure-deployed OpenAI models |
| Groq | OpenAI-compatible | Llama 3.1 / 3.3 / 4 — free tier |
| xAI | OpenAI-compatible | Grok models |
| DeepSeek | OpenAI-compatible | V3 / V4 families |
| Qwen / Qwen-CN | OpenAI-compatible | Alibaba DashScope |
| GLM / GLM-CN | OpenAI-compatible | Zhipu BigModel |
| MiniMax / MiniMax-CN | OpenAI-compatible | M2.x family |
| OpenRouter | OpenAI-compatible | Multi-model router |
| Ollama | OpenAI-compatible | Local runtime |

## Quick Start

```python
from tradingagents.llm_clients.factory import create_llm_client

client = create_llm_client("groq", "llama-3.3-70b-versatile")
llm = client.get_llm()
response = llm.invoke([("human", "What is 2+2?")])
print(response.content)
```

See [Groq Setup](groq-setup-and-usage.md) for API key configuration.
