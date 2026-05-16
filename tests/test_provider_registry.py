"""Tests for the provider registry.

Verifies all existing providers are registered and that registry
metadata matches the factory's behavior. The registry must cover
EVERY provider currently supported by create_llm_client() to
preserve backward compatibility.
"""

import pytest

from tradingagents.llm_clients.provider_registry import (
    PROVIDER_REGISTRY,
    ProviderConfig,
)


# Single source of truth for which providers must be registered.
# Mirrors the union of _OPENAI_COMPATIBLE and the explicit branches
# in the current factory.py.
_EXPECTED_PROVIDERS = (
    "openai",
    "anthropic",
    "google",
    "azure",
    "xai",
    "deepseek",
    "qwen",
    "qwen-cn",
    "glm",
    "glm-cn",
    "minimax",
    "minimax-cn",
    "openrouter",
    "groq",
    "ollama",
)


@pytest.mark.unit
class TestRegistryCoverage:
    """Every provider supported by the legacy factory must be in the registry."""

    @pytest.mark.parametrize("provider", _EXPECTED_PROVIDERS)
    def test_provider_is_registered(self, provider):
        assert provider in PROVIDER_REGISTRY, (
            f"Provider '{provider}' missing from registry — "
            f"this is a backward-compatibility regression."
        )

    def test_registry_entries_are_provider_config(self):
        for name, config in PROVIDER_REGISTRY.items():
            assert isinstance(config, ProviderConfig), (
                f"Registry entry '{name}' is not a ProviderConfig instance"
            )


@pytest.mark.unit
class TestGroqRegistryEntry:
    """Groq-specific registry metadata."""

    def test_groq_uses_openai_client(self):
        from tradingagents.llm_clients.openai_client import OpenAIClient
        config = PROVIDER_REGISTRY["groq"]
        assert config.client_class is OpenAIClient

    def test_groq_base_url(self):
        config = PROVIDER_REGISTRY["groq"]
        assert config.base_url == "https://api.groq.com/openai/v1"

    def test_groq_api_key_env(self):
        config = PROVIDER_REGISTRY["groq"]
        assert config.api_key_env == "GROQ_API_KEY"


@pytest.mark.unit
class TestSpecialProviders:
    """Edge cases — ollama has no API key, azure has no fixed base_url."""

    def test_ollama_has_no_api_key(self):
        config = PROVIDER_REGISTRY["ollama"]
        assert config.api_key_env is None

    def test_azure_base_url_is_none(self):
        """Azure resolves base_url from AZURE_OPENAI_ENDPOINT env var."""
        config = PROVIDER_REGISTRY["azure"]
        assert config.base_url is None
