"""Factory for LLM clients.

Resolves the provider via the registry and instantiates the right
client class. OpenAIClient takes a `provider` kwarg because one
class serves many OpenAI-compatible providers (it consults that
kwarg to pick base URL / API key env). Other client classes are
provider-specific and don't need it; we don't pass it to them.
"""

from typing import Optional

from .base_client import BaseLLMClient
from .provider_registry import PROVIDER_REGISTRY


def create_llm_client(
    provider: str,
    model: str,
    base_url: Optional[str] = None,
    **kwargs,
) -> BaseLLMClient:
    """Create an LLM client for the given provider.

    Args:
        provider: Provider name (case-insensitive).
        model: Model identifier.
        base_url: Optional override for the provider's default endpoint.
        **kwargs: Forwarded to the underlying client class.

    Returns:
        Configured BaseLLMClient instance.

    Raises:
        ValueError: If the provider is not registered.
    """
    provider_lower = provider.lower()

    if provider_lower not in PROVIDER_REGISTRY:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    config = PROVIDER_REGISTRY[provider_lower]
    effective_base_url = base_url or config.base_url

    # OpenAIClient handles many OpenAI-compatible providers, so it needs
    # the provider name to resolve its own internal base-URL/api-key
    # mappings. Other client classes are provider-specific and accept
    # only (model, base_url, **kwargs).
    from .openai_client import OpenAIClient
    if config.client_class is OpenAIClient:
        return config.client_class(
            model=model,
            base_url=effective_base_url,
            provider=provider_lower,
            **kwargs,
        )
    return config.client_class(
        model=model,
        base_url=effective_base_url,
        **kwargs,
    )
