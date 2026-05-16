# Provider System Architecture

## Overview

The provider system uses a **registry pattern** to centralize metadata
and a thin **factory** to dispatch construction across client classes.

## Components

### 1. `provider_registry.py`

Single source of truth for which providers exist and how they map to
client classes:

```python
@dataclass(frozen=True)
class ProviderConfig:
    name: str
    client_class: type
    base_url: Optional[str]
    api_key_env: Optional[str]

PROVIDER_REGISTRY: Dict[str, ProviderConfig] = { ... }
```

### 2. `factory.py`

Resolves a provider name to a `ProviderConfig`, then instantiates the
appropriate client class. `OpenAIClient` is special-cased because it
serves multiple OpenAI-compatible providers and needs the provider
name to resolve its internal base-URL / API-key maps.

### 3. Client classes (one per native API)

- `OpenAIClient` — OpenAI, Groq, xAI, DeepSeek, Qwen, GLM, MiniMax,
  OpenRouter, Ollama (all OpenAI-compatible).
- `AnthropicClient` — Claude models.
- `GoogleClient` — Gemini models.
- `AzureOpenAIClient` — Azure-hosted OpenAI deployments.

Each inherits from `BaseLLMClient` and implements `get_llm()` and
`validate_model()`.

### 4. `capabilities.py`

Declares per-model API quirks (does the model accept `tool_choice`?
JSON schema response_format?). The OpenAI-compatible client subclasses
consult this table instead of hardcoding model-name `if` ladders.

Lookup precedence: exact-ID match → regex pattern → `_DEFAULT`
(permissive).

### 5. `model_catalog.py`

The list of models shown in CLI dropdowns, organized by provider and
selection mode (quick vs deep). Used by the CLI for interactive
selection and by `validators.validate_model()` for warning on unknown
model names.

## Data Flow

```
create_llm_client("groq", "llama-3.3-70b-versatile")
        │
        ▼
PROVIDER_REGISTRY["groq"]  →  ProviderConfig(client_class=OpenAIClient,
                                              base_url="https://api.groq.com/openai/v1",
                                              api_key_env="GROQ_API_KEY")
        │
        ▼  (factory: special-cases OpenAIClient to pass provider= kwarg)
OpenAIClient(model=..., base_url=..., provider="groq")
        │
        ▼  client.get_llm()
NormalizedChatOpenAI(model=..., base_url=..., api_key=<from env>)
```

## Why a Registry?

Before: provider metadata was scattered across `factory.py` (conditionals),
`openai_client.py` (`_PROVIDER_BASE_URL`), `api_key_env.py` (env var map),
and `model_catalog.py` (model list).

After: registry consolidates **who** is registered. Per-implementation
details (base URLs for OpenAI-compatible providers, model lists) still
live in their natural homes, but the registry tells you at a glance
which providers exist and which client class serves each.

See [Adding a Provider](adding-a-provider.md) for the workflow.
