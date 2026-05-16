"""Provider registry: single source of truth for provider metadata.

The registry holds metadata (which client class to instantiate, default
base URL for display/docs, API key env var). Actual client construction
is done by the factory, which can apply per-client-class quirks (e.g.,
OpenAIClient receives the `provider` kwarg; other clients do not).

base_url here is informational/documentation — the live source of truth
for OpenAI-compatible providers' endpoints remains
`openai_client._PROVIDER_BASE_URL`, which OpenAIClient consults at
runtime. The registry stays in sync with that mapping by convention.

When adding a new provider:
  1. Add an entry here.
  2. If OpenAI-compatible: ensure openai_client._PROVIDER_BASE_URL and
     api_key_env.PROVIDER_API_KEY_ENV are also updated.
  3. If a new client class: import and reference it here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ProviderConfig:
    """Metadata for a single LLM provider.

    Attributes:
        name: Canonical lowercase provider key.
        client_class: BaseLLMClient subclass to instantiate.
        base_url: Default API endpoint, or None if dynamically resolved.
        api_key_env: Environment variable holding the API key, or None
            for providers like ollama that do not authenticate.
    """
    name: str
    client_class: type
    base_url: Optional[str]
    api_key_env: Optional[str]


def _build_registry() -> Dict[str, ProviderConfig]:
    """Construct the registry with lazy imports to avoid heavy SDK imports
    at module-load time (mirrors the lazy-import pattern in factory.py)."""
    from tradingagents.llm_clients.openai_client import OpenAIClient
    from tradingagents.llm_clients.anthropic_client import AnthropicClient
    from tradingagents.llm_clients.google_client import GoogleClient
    from tradingagents.llm_clients.azure_client import AzureOpenAIClient

    # All OpenAI-compatible providers use OpenAIClient. Base URLs mirror
    # openai_client._PROVIDER_BASE_URL — keep these in sync.
    openai_compatible: Dict[str, tuple] = {
        "openai":     ("https://api.openai.com/v1",                                "OPENAI_API_KEY"),
        "xai":        ("https://api.x.ai/v1",                                      "XAI_API_KEY"),
        "deepseek":   ("https://api.deepseek.com",                                 "DEEPSEEK_API_KEY"),
        "qwen":       ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1",   "DASHSCOPE_API_KEY"),
        "qwen-cn":    ("https://dashscope.aliyuncs.com/compatible-mode/v1",        "DASHSCOPE_CN_API_KEY"),
        "glm":        ("https://api.z.ai/api/paas/v4/",                            "ZHIPU_API_KEY"),
        "glm-cn":     ("https://open.bigmodel.cn/api/paas/v4/",                    "ZHIPU_CN_API_KEY"),
        "minimax":    ("https://api.minimax.io/v1",                                "MINIMAX_API_KEY"),
        "minimax-cn": ("https://api.minimaxi.com/v1",                              "MINIMAX_CN_API_KEY"),
        "openrouter": ("https://openrouter.ai/api/v1",                             "OPENROUTER_API_KEY"),
        "groq":       ("https://api.groq.com/openai/v1",                           "GROQ_API_KEY"),
        "ollama":     ("http://localhost:11434/v1",                                 None),
    }

    registry: Dict[str, ProviderConfig] = {
        name: ProviderConfig(name=name, client_class=OpenAIClient,
                             base_url=base_url, api_key_env=api_key_env)
        for name, (base_url, api_key_env) in openai_compatible.items()
    }

    # Native (non-OpenAI-compatible) clients
    registry["anthropic"] = ProviderConfig(
        name="anthropic", client_class=AnthropicClient,
        base_url=None, api_key_env="ANTHROPIC_API_KEY",
    )
    registry["google"] = ProviderConfig(
        name="google", client_class=GoogleClient,
        base_url=None, api_key_env="GOOGLE_API_KEY",
    )
    registry["azure"] = ProviderConfig(
        name="azure", client_class=AzureOpenAIClient,
        base_url=None, api_key_env="AZURE_OPENAI_API_KEY",
    )
    return registry


PROVIDER_REGISTRY: Dict[str, ProviderConfig] = _build_registry()
