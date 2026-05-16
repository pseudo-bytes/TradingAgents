# Groq Test-Driven Refactor: Implementation Plan (REVISED)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete Groq provider integration with verified feature support (chat, tools, structured output, async) while building an extensible provider registry system for future providers.

**Architecture:** Test-driven refactor — write tests following existing repo patterns (direct client instantiation + payload/bound-kwarg inspection), then add a provider registry as a single source of truth for metadata, then verify Groq features work. Backward compatible with existing factory API.

**Tech Stack:** LangChain (ChatOpenAI), pytest (test runner with `unit` marker), dataclasses (ProviderConfig). No new dependencies required.

**Repository conventions verified:**
- Tests are FLAT under `tests/` (no subdirectories)
- Tests use `@pytest.mark.unit` marker
- Tests use direct client instantiation + inspect `_get_request_payload()` and bound kwargs
- `conftest.py` autouses `_dummy_api_keys` fixture (sets all API keys to "placeholder")

---

## Phase 0: Pre-Phase Research & Setup (BLOCKING)

### Task 0.1: Research Groq API Capabilities

- [ ] **Step 1: Check Groq tool calling support**

Visit: https://console.groq.com/docs/tool-use

Verify and document:
- Does Groq accept `tools` array? (Expected: YES, since 3.1+)
- Does it support `tool_choice` parameter? (Most likely YES, but verify)
- Are there any model-specific restrictions?

- [ ] **Step 2: Check Groq structured output support**

Visit: https://console.groq.com/docs/text-chat (look for response_format)

Verify and document:
- Does Groq support `response_format={"type": "json_object"}`? (Expected: YES)
- Does it support `response_format={"type": "json_schema", ...}`? (Likely NO for Llama)

- [ ] **Step 3: Confirm streaming & async**

Verify Groq's OpenAI-compatible endpoint supports SSE streaming (it does — same protocol as OpenAI).

- [ ] **Step 4: Document findings inline**

Update this plan's "Research Findings" section below with answers. These directly inform Task 9 (capabilities).

**Research Findings (fill in after research):**
```
Tool calling: __ (Yes/No)
Tool choice param: __ (Yes/No — note any model exceptions)
JSON mode: __ (Yes/No)
JSON schema: __ (Yes/No)
Streaming: __ (Yes/No)
Notes: __
```

---

### Task 0.2: Update conftest.py with GROQ_API_KEY

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Add GROQ_API_KEY to the autouse fixture**

In `tests/conftest.py`, add `"GROQ_API_KEY"` to the `_API_KEY_ENV_VARS` tuple:

```python
_API_KEY_ENV_VARS = (
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "ANTHROPIC_API_KEY",
    "XAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_CN_API_KEY",
    "ZHIPU_API_KEY",
    "ZHIPU_CN_API_KEY",
    "MINIMAX_API_KEY",
    "MINIMAX_CN_API_KEY",
    "OPENROUTER_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "GROQ_API_KEY",               # NEW
    "ALPHA_VANTAGE_API_KEY",
)
```

**Note on conftest behavior:** The autouse fixture uses
`monkeypatch.setenv(var, os.environ.get(var, "placeholder"))`. If the
developer running tests has a real `GROQ_API_KEY` in their shell, that
real value is propagated (not overwritten with "placeholder"). The
`test_missing_api_key_raises` test handles this by explicitly calling
`monkeypatch.delenv("GROQ_API_KEY", raising=False)` — both monkeypatch
operations share the same function-scoped fixture, so removal works
regardless of who set the value first.

- [ ] **Step 2: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add GROQ_API_KEY to conftest dummy-key fixture"
```

---

## Phase 1: Verification Tests (Tests First)

**Note on TDD vs verification:** Groq already routes through `OpenAIClient`
in the current code (`groq` is in `_OPENAI_COMPATIBLE`). Most tests in
Tasks 1-4 will **PASS on first run** — they document and protect existing
behavior rather than driving new implementation. The genuinely failing
test is Task 5 (registry), which forces Phase 2's implementation.
This is verification-driven development, not strict TDD — both are valid
strategies for solidifying integration of existing code.

**Test directory:** `tests/` (flat — no subdirectories)  
**Test pattern:** Follow `test_minimax.py` and `test_capabilities.py` — direct instantiation, payload/bound-kwarg inspection. NO HTTP mocking.

### Task 1: Write Groq Client Construction Tests

**Files:**
- Create: `tests/test_groq.py`

- [ ] **Step 1: Write tests for Groq client construction**

File: `tests/test_groq.py`

```python
"""Tests for Groq provider integration.

Verifies that the Groq provider correctly routes through OpenAIClient
with the right base URL, API key env var, and that returned LLM
instances are properly configured.

Follows the repo convention of direct client instantiation and
payload-inspection over HTTP mocking (see test_minimax.py).
"""

import pytest

from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.openai_client import (
    OpenAIClient,
    NormalizedChatOpenAI,
)


@pytest.mark.unit
class TestGroqClientConstruction:
    """Verify the factory routes 'groq' through OpenAIClient with
    Groq's base URL and the GROQ_API_KEY env var."""

    def test_factory_returns_openai_client(self):
        """Groq is OpenAI-compatible, so it uses OpenAIClient."""
        client = create_llm_client("groq", "llama-3.3-70b-versatile")
        assert isinstance(client, OpenAIClient)
        assert client.provider == "groq"
        assert client.model == "llama-3.3-70b-versatile"

    def test_llm_has_groq_base_url(self, monkeypatch):
        """get_llm() should configure ChatOpenAI with Groq's endpoint."""
        monkeypatch.setenv("GROQ_API_KEY", "test_key_xyz")
        client = create_llm_client("groq", "llama-3.3-70b-versatile")
        llm = client.get_llm()
        assert isinstance(llm, NormalizedChatOpenAI)
        # base_url is stored on the langchain client
        assert str(llm.openai_api_base) == "https://api.groq.com/openai/v1"

    def test_llm_uses_groq_api_key_from_env(self, monkeypatch):
        """GROQ_API_KEY env var is propagated to the langchain client."""
        monkeypatch.setenv("GROQ_API_KEY", "test_key_xyz")
        client = create_llm_client("groq", "llama-3.3-70b-versatile")
        llm = client.get_llm()
        # langchain stores api_key as a SecretStr — access via get_secret_value
        assert llm.openai_api_key.get_secret_value() == "test_key_xyz"

    def test_missing_api_key_raises(self, monkeypatch):
        """If GROQ_API_KEY is unset, get_llm() raises with a helpful message."""
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        client = create_llm_client("groq", "llama-3.3-70b-versatile")
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            client.get_llm()

    def test_custom_base_url_override(self, monkeypatch):
        """Explicit base_url overrides Groq's default endpoint."""
        monkeypatch.setenv("GROQ_API_KEY", "test_key_xyz")
        client = create_llm_client(
            "groq",
            "llama-3.3-70b-versatile",
            base_url="https://custom.example.com/v1",
        )
        llm = client.get_llm()
        assert str(llm.openai_api_base) == "https://custom.example.com/v1"


@pytest.mark.unit
class TestGroqModelValidation:
    """Curated Groq models must pass validate_model() so users don't
    see a spurious RuntimeWarning when picking a known model from the
    catalog. validate_model() reads from MODEL_OPTIONS via
    get_known_models() — this test catches catalog/validator drift."""

    @pytest.mark.parametrize("model", [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-specdec",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "llama-3.1-70b-versatile",
    ])
    def test_curated_model_validates(self, model):
        from tradingagents.llm_clients.validators import validate_model
        assert validate_model("groq", model) is True, (
            f"Curated Groq model {model!r} failed validation — "
            f"check tradingagents/llm_clients/model_catalog.py:_GROQ_MODELS"
        )

    def test_unknown_groq_model_does_not_validate(self):
        """Sanity: a made-up Groq model returns False (not an exception)."""
        from tradingagents.llm_clients.validators import validate_model
        assert validate_model("groq", "made-up-model-id-xyz") is False
```

- [ ] **Step 2: Run tests to verify they pass or fail meaningfully**

```bash
pytest tests/test_groq.py -v
```

Expected: Tests should largely **PASS** since Groq routing already exists. Any failures indicate gaps in current Groq support.

- [ ] **Step 3: Commit**

```bash
git add tests/test_groq.py
git commit -m "test: add groq client construction tests"
```

---

### Task 2: Write Groq Tool Calling Test

**Files:**
- Modify: `tests/test_groq.py` (append)

- [ ] **Step 1: Append tool calling tests**

Append to `tests/test_groq.py`:

```python
from pydantic import BaseModel


@pytest.mark.unit
class TestGroqToolCalling:
    """Verify Groq Llama models can bind tools via with_structured_output().
    
    Follows the pattern from test_minimax.py — inspect bound kwargs to
    confirm the schema is sent as a tool. Does NOT call the API.
    """

    class _Decision(BaseModel):
        action: str
        confidence: float

    def _bound_kwargs(self, runnable):
        """Extract the kwargs passed to the underlying bind() call."""
        first = runnable.steps[0] if hasattr(runnable, "steps") else runnable
        return getattr(first, "kwargs", {})

    def test_groq_binds_schema_as_tool(self, monkeypatch):
        """Structured output should bind the pydantic schema as a tool."""
        monkeypatch.setenv("GROQ_API_KEY", "test_key_xyz")
        client = create_llm_client("groq", "llama-3.3-70b-versatile")
        llm = client.get_llm()
        bound = llm.with_structured_output(self._Decision)
        tools = self._bound_kwargs(bound).get("tools", [])
        assert any(
            t.get("function", {}).get("name") == "_Decision" for t in tools
        ), f"schema not bound as tool: {tools}"
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_groq.py::TestGroqToolCalling -v
```

Expected: **PASS** if Groq follows default capabilities. Failures here would indicate Groq has unique tool-calling requirements.

- [ ] **Step 3: Commit**

```bash
git add tests/test_groq.py
git commit -m "test: add groq tool-calling structured-output test"
```

---

### Task 3: Write Groq Capabilities Test

**Files:**
- Modify: `tests/test_capabilities.py` (append)

- [ ] **Step 1: Append Groq capability tests**

Append to `tests/test_capabilities.py`:

```python
@pytest.mark.unit
class TestGroqCapabilities:
    """Verify capability resolution for Groq Llama models.
    
    Update expected values after Phase 0 research. As of writing,
    the assumption is that Groq Llama models support default behavior
    (tools + json_object) and can use the _DEFAULT permissive profile.
    """

    def test_groq_llama_3_3_uses_default(self):
        """llama-3.3-70b-versatile uses default capabilities (full support)."""
        caps = get_capabilities("llama-3.3-70b-versatile")
        # Update these based on Phase 0 research findings
        assert caps.supports_tool_choice is True
        assert caps.supports_json_mode is True
        assert caps.preferred_structured_method == "function_calling"

    def test_groq_llama_3_1_uses_default(self):
        """llama-3.1-8b-instant uses default capabilities."""
        caps = get_capabilities("llama-3.1-8b-instant")
        assert caps.supports_tool_choice is True
        assert caps.preferred_structured_method == "function_calling"

    def test_groq_llama_4_uses_default(self):
        """Llama 4 models use default capabilities."""
        caps = get_capabilities("meta-llama/llama-4-scout-17b-16e-instruct")
        assert caps.supports_tool_choice is True
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_capabilities.py::TestGroqCapabilities -v
```

Expected: **PASS** — Groq Llama models fall through to `_DEFAULT` capabilities (permissive). If research reveals quirks (e.g., tool_choice unsupported), update Task 9 to add explicit `_BY_ID` entries.

- [ ] **Step 3: Commit**

```bash
git add tests/test_capabilities.py
git commit -m "test: add groq capability resolution tests"
```

---

### Task 4: Write Groq Async/Streaming Smoke Tests

**Files:**
- Modify: `tests/test_groq.py` (append)

- [ ] **Step 1: Append async/streaming tests**

Append to `tests/test_groq.py`:

```python
@pytest.mark.unit
class TestGroqAsyncStreaming:
    """Verify Groq LLM exposes async and streaming methods.
    
    Does NOT call the API — only verifies the langchain interface
    is present. Real async/streaming is exercised by smoke tests.
    """

    def test_async_invoke_method_exists(self, monkeypatch):
        """The returned LLM has the async invoke method."""
        monkeypatch.setenv("GROQ_API_KEY", "test_key_xyz")
        client = create_llm_client("groq", "llama-3.3-70b-versatile")
        llm = client.get_llm()
        assert callable(getattr(llm, "ainvoke", None))

    def test_stream_methods_exist(self, monkeypatch):
        """The returned LLM exposes stream() and astream()."""
        monkeypatch.setenv("GROQ_API_KEY", "test_key_xyz")
        client = create_llm_client("groq", "llama-3.3-70b-versatile")
        llm = client.get_llm()
        assert callable(getattr(llm, "stream", None))
        assert callable(getattr(llm, "astream", None))
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_groq.py::TestGroqAsyncStreaming -v
```

Expected: **PASS** — these methods exist on ChatOpenAI by default.

- [ ] **Step 3: Commit**

```bash
git add tests/test_groq.py
git commit -m "test: add groq async/stream interface smoke tests"
```

---

### Task 5: Write Provider Registry Tests

**Files:**
- Create: `tests/test_provider_registry.py`
- Create: `tradingagents/llm_clients/provider_registry.py` (stub)

- [ ] **Step 1: Create a minimal stub for provider_registry.py**

File: `tradingagents/llm_clients/provider_registry.py`

```python
"""Provider registry: single source of truth for provider metadata.

Stub created by Phase 1. Full implementation arrives in Phase 2 (Task 7).
"""

from dataclasses import dataclass
from typing import Dict, Optional, Type


@dataclass(frozen=True)
class ProviderConfig:
    """Metadata for a single provider."""
    name: str
    client_class: type
    base_url: Optional[str]
    api_key_env: Optional[str]


PROVIDER_REGISTRY: Dict[str, ProviderConfig] = {}
```

- [ ] **Step 2: Write tests for the registry**

File: `tests/test_provider_registry.py`

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/test_provider_registry.py -v
```

Expected: **FAIL** — registry is empty, parametrized tests fail for each missing provider.

- [ ] **Step 4: Commit**

```bash
git add tests/test_provider_registry.py tradingagents/llm_clients/provider_registry.py
git commit -m "test: add provider registry coverage tests (failing)"
```

---

## Phase 2: Provider Registry Implementation

### Task 6: Implement Provider Registry

**Files:**
- Modify: `tradingagents/llm_clients/provider_registry.py`

- [ ] **Step 1: Implement full ProviderConfig and registry**

Replace the stub in `tradingagents/llm_clients/provider_registry.py`:

```python
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
from typing import Dict, Optional, Type


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
        "ollama":     ("http://localhost:11434/v1",                                None),
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
```

- [ ] **Step 2: Run registry tests**

```bash
pytest tests/test_provider_registry.py -v
```

Expected: **ALL PASS** — every expected provider is registered with correct metadata.

- [ ] **Step 3: Commit**

```bash
git add tradingagents/llm_clients/provider_registry.py
git commit -m "feat(llm_clients): introduce provider registry with full provider coverage"
```

---

### Task 7: Refactor Factory to Use Registry

**Files:**
- Modify: `tradingagents/llm_clients/factory.py`

- [ ] **Step 1: Replace factory.py implementation**

File: `tradingagents/llm_clients/factory.py`

```python
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
```

- [ ] **Step 2: Run all LLM-related tests**

```bash
pytest tests/test_groq.py tests/test_provider_registry.py tests/test_capabilities.py tests/test_minimax.py tests/test_deepseek_reasoning.py tests/test_ollama_base_url.py tests/test_api_key_env.py tests/test_model_validation.py -v
```

Expected: **ALL PASS** — Groq, registry, and existing provider tests all green.

- [ ] **Step 3: Run the full test suite to confirm no regressions**

```bash
pytest tests/ -v --tb=short
```

Expected: All previously-passing tests still pass.

- [ ] **Step 4: Commit**

```bash
git add tradingagents/llm_clients/factory.py
git commit -m "refactor(factory): dispatch through provider registry (backward compatible)"
```

---

## Phase 3: Groq Capability Adjustments (if needed)

### Task 8: Update Capabilities Matrix (Conditional on Research)

**Files:**
- Conditionally modify: `tradingagents/llm_clients/capabilities.py`

- [ ] **Step 1: Decide if any changes are needed**

Review Phase 0 research findings:

- **If Groq Llama models behave like the `_DEFAULT` profile** (support tools, tool_choice, json_object): NO CHANGES NEEDED. Skip to Step 4 and commit an empty change-note.

- **If Groq Llama has quirks** (e.g., rejects `tool_choice`, no json_object, etc.): Continue to Step 2.

- [ ] **Step 2: Add Groq-specific ModelCapabilities entries (if needed)**

In `tradingagents/llm_clients/capabilities.py`, add a profile and `_BY_ID` entries. Example (only if research shows quirks):

```python
# Add near other profiles:
_GROQ_LLAMA = ModelCapabilities(
    supports_tool_choice=True,    # set to False if research shows otherwise
    supports_json_mode=True,      # set to False if Llama models lack json_object
    supports_json_schema=False,   # Llama models on Groq don't support json_schema
    preferred_structured_method="function_calling",
)

# In _BY_ID, after existing entries:
_BY_ID.update({
    "llama-3.3-70b-versatile": _GROQ_LLAMA,
    "llama-3.1-70b-versatile": _GROQ_LLAMA,
    "llama-3.1-8b-instant":    _GROQ_LLAMA,
    "llama-3.3-70b-specdec":   _GROQ_LLAMA,
    "meta-llama/llama-4-scout-17b-16e-instruct":    _GROQ_LLAMA,
    "meta-llama/llama-4-maverick-17b-128e-instruct": _GROQ_LLAMA,
})
```

- [ ] **Step 3: Update tests to match (if you changed defaults)**

Edit the `TestGroqCapabilities` class in `tests/test_capabilities.py` to assert the new expected values (e.g., if you set `supports_json_schema=False`, add `assert caps.supports_json_schema is False`).

- [ ] **Step 4: Run capability and Groq tests**

```bash
pytest tests/test_capabilities.py tests/test_groq.py -v
```

Expected: **ALL PASS**.

- [ ] **Step 5: Commit**

```bash
git add tradingagents/llm_clients/capabilities.py tests/test_capabilities.py
# If no changes were needed, skip the commit entirely.
git commit -m "feat(capabilities): record Groq Llama model quirks (if any)"
```

---

## Phase 4: Documentation

### Task 9: Create Provider Documentation Directory

**Files:**
- Create: `docs/providers/README.md`
- Create: `docs/providers/architecture.md`
- Create: `docs/providers/adding-a-provider.md`
- Create: `docs/providers/groq-setup-and-usage.md`
- Create: `docs/providers/capability-matrix.md`
- Create: `docs/providers/troubleshooting.md`

- [ ] **Step 1: Create docs/providers/README.md**

File: `docs/providers/README.md`

```markdown
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
```

- [ ] **Step 2: Create docs/providers/architecture.md**

File: `docs/providers/architecture.md`

```markdown
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
```

- [ ] **Step 3: Create docs/providers/adding-a-provider.md**

File: `docs/providers/adding-a-provider.md`

```markdown
# Adding a New Provider

Two paths depending on whether the provider exposes an OpenAI-compatible
endpoint.

## Path A: OpenAI-Compatible Provider (~15 minutes)

Most third-party providers (Groq, xAI, DeepSeek, OpenRouter, etc.) expose
an OpenAI-compatible Chat Completions API. You can reuse `OpenAIClient`.

### Step 1: Register the provider

`tradingagents/llm_clients/provider_registry.py` — add an entry to the
`openai_compatible` dict inside `_build_registry()`:

```python
"newprovider": ("https://api.newprovider.com/v1", "NEWPROVIDER_API_KEY"),
```

### Step 2: Mirror the base URL in OpenAIClient's internal map

`tradingagents/llm_clients/openai_client.py` — add to `_PROVIDER_BASE_URL`:

```python
"newprovider": "https://api.newprovider.com/v1",
```

### Step 3: Add the API-key env var

`tradingagents/llm_clients/api_key_env.py` — add to `PROVIDER_API_KEY_ENV`:

```python
"newprovider": "NEWPROVIDER_API_KEY",
```

### Step 4: Add to factory's OpenAI-compatible tuple

`tradingagents/llm_clients/factory.py` — already routes through the registry,
no change needed.

### Step 5: Add models to the catalog

`tradingagents/llm_clients/model_catalog.py`:

```python
_NEWPROVIDER_MODELS: Dict[str, List[ModelOption]] = {
    "quick": [
        ("Fast model", "newprovider-fast"),
        ("Custom model ID", "custom"),
    ],
    "deep": [
        ("Best model", "newprovider-pro"),
        ("Custom model ID", "custom"),
    ],
}

MODEL_OPTIONS["newprovider"] = _NEWPROVIDER_MODELS
```

### Step 6: Add to the CLI provider list

`cli/utils.py` — inside `select_llm_provider()`, add a tuple in the providers
list with display name, key, and default base URL.

### Step 7: Capabilities (optional)

If models have unusual quirks (e.g., reject `tool_choice`), add entries to
`capabilities._BY_ID`. Otherwise they inherit the permissive `_DEFAULT`.

### Step 8: Add tests

Create `tests/test_newprovider.py` following the pattern in `tests/test_groq.py`:

```python
@pytest.mark.unit
class TestNewProviderClientConstruction:
    def test_factory_returns_openai_client(self):
        client = create_llm_client("newprovider", "model-x")
        assert isinstance(client, OpenAIClient)
        assert client.provider == "newprovider"
```

Also extend `tests/test_provider_registry.py::_EXPECTED_PROVIDERS` to include
your provider.

### Step 9: Update docs

- Add a row to `docs/providers/README.md` provider table.
- Add a row to `docs/providers/capability-matrix.md`.
- Optionally create `docs/providers/newprovider-setup-and-usage.md`.

### Step 10: Verify

```bash
pytest tests/test_newprovider.py tests/test_provider_registry.py -v
pytest tests/ -v
```

---

## Path B: Native (non-OpenAI-compatible) Provider (~2 hours)

If the provider has a custom API (like Anthropic's Messages API or Google's
GenAI), you'll need a dedicated client class.

### Step 1: Create the client class

`tradingagents/llm_clients/newprovider_client.py`, modeled on
`anthropic_client.py`:

```python
from typing import Any, Optional

from .base_client import BaseLLMClient, normalize_content


class NewProviderClient(BaseLLMClient):
    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        self.warn_if_unknown_model()
        # Instantiate the langchain wrapper / SDK client here.
        ...

    def validate_model(self) -> bool:
        from .validators import validate_model
        return validate_model("newprovider", self.model)
```

### Step 2: Register in provider_registry.py

```python
from tradingagents.llm_clients.newprovider_client import NewProviderClient
registry["newprovider"] = ProviderConfig(
    name="newprovider", client_class=NewProviderClient,
    base_url=None, api_key_env="NEWPROVIDER_API_KEY",
)
```

The factory's "special-case OpenAIClient" branch ensures your client
receives only `(model, base_url, **kwargs)` — it won't get an unwanted
`provider` kwarg.

### Step 3-10: Same as Path A.
```

- [ ] **Step 4: Create docs/providers/groq-setup-and-usage.md**

File: `docs/providers/groq-setup-and-usage.md`

```markdown
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
```

- [ ] **Step 5: Create docs/providers/capability-matrix.md**

File: `docs/providers/capability-matrix.md`

**Important:** The Groq rows below are placeholders. Before committing
this file, update each cell using the findings from Phase 0 research
(Task 0.1). The values shown reflect the most likely defaults based on
Groq's published API behavior, but should not be treated as authoritative
until verified.

```markdown
# Provider Capability Matrix

What each provider/model accepts at the API layer. Entries marked `?`
need verification; update as research finishes.

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

Legend: ✅ supported · ❌ not supported · `?` unverified · n/a not applicable.

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
```

- [ ] **Step 6: Create docs/providers/troubleshooting.md**

File: `docs/providers/troubleshooting.md`

```markdown
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
```

- [ ] **Step 7: Commit all documentation**

```bash
git add docs/providers/
git commit -m "docs: add provider system guide (architecture, Groq, capability matrix, troubleshooting)"
```

---

### Task 10: Link Docs from README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Find an appropriate section in README.md**

Open `README.md`. Look for a section about LLM provider configuration (likely near setup / env vars).

- [ ] **Step 2: Add a docs link**

Insert a short paragraph near the provider configuration content (use Edit with enough surrounding context to disambiguate):

```markdown
For detailed provider setup, architecture, and a feature support matrix,
see [`docs/providers/`](docs/providers/README.md).
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs(readme): link to providers documentation"
```

---

## Phase 5: Verification

### Task 11: Live Smoke Test with Groq

**Prerequisites:** Real `GROQ_API_KEY` set in env / `.env`.

- [ ] **Step 1: Run the CLI**

```bash
python main.py
```

- [ ] **Step 2: Select Groq and a model**

When prompted:
- Provider: **Groq (Free tier — Llama 3.3 / Llama 4)**
- Mode: **quick** or **deep**
- Model: **`llama-3.3-70b-versatile`** (a known-good default)

- [ ] **Step 3: Run a small trading task**

Use a simple ticker the system already supports. Observe:
- ✅ Calls succeed (no `ValueError` from missing key).
- ✅ Responses come back as strings (content normalization works).
- ✅ Any structured-output / tool-using agents in the run produce
      valid outputs.

- [ ] **Step 4: Record findings**

If anything fails, file an issue or add a follow-up task to this plan.
If the smoke test succeeds, note the model + task + observations
inline in the PR description for the change.

---

### Task 12: Final Verification

- [ ] **Step 1: Full test suite**

```bash
pytest tests/ -v --tb=short
```

Expected: **ALL PASS**.

- [ ] **Step 2: Verify backward compatibility manually**

Spot-check that legacy callers still work:

```python
# Should still return clients (no exceptions).
from tradingagents.llm_clients.factory import create_llm_client

create_llm_client("openai", "gpt-5.5")
create_llm_client("anthropic", "claude-sonnet-4-6")
create_llm_client("google", "gemini-2.5-flash")
create_llm_client("deepseek", "deepseek-chat")
create_llm_client("ollama", "qwen3:latest")
```

- [ ] **Step 3: Inspect commit history**

```bash
git log --oneline origin/main..HEAD
```

Verify commits are coherent: research / dep / conftest / tests / registry / factory / capabilities / docs / readme.

- [ ] **Step 4: (Optional) Open a PR**

```bash
gh pr create --title "feat: complete Groq integration via provider registry" --body "$(cat <<'EOF'
## Summary

- Verified Groq provider works end-to-end (chat / tools / structured output / async).
- Introduced `provider_registry.py` as the single source of truth for provider metadata.
- Factory now dispatches through the registry while preserving backward compatibility.
- Added comprehensive provider documentation under `docs/providers/`.

## Test plan

- [ ] `pytest tests/ -v` passes locally.
- [ ] Smoke test with a real `GROQ_API_KEY` against `llama-3.3-70b-versatile`.
- [ ] CI green.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## File Structure Summary

**Created:**

```
tradingagents/llm_clients/
  provider_registry.py            (NEW)

tests/
  test_groq.py                    (NEW)
  test_provider_registry.py       (NEW)

docs/providers/
  README.md                       (NEW)
  architecture.md                 (NEW)
  adding-a-provider.md            (NEW)
  groq-setup-and-usage.md         (NEW)
  capability-matrix.md            (NEW)
  troubleshooting.md              (NEW)
```

**Modified:**

```
tradingagents/llm_clients/
  factory.py                      (dispatch via registry)
  capabilities.py                 (Groq entries — only if research shows quirks)

tests/
  test_capabilities.py            (append Groq capability tests)
  conftest.py                     (add GROQ_API_KEY to dummy-key fixture)

README.md                         (link to docs/providers/)
```

**Unchanged (intentionally):**

```
tradingagents/llm_clients/openai_client.py     (already handles Groq)
tradingagents/llm_clients/api_key_env.py       (GROQ_API_KEY already mapped)
tradingagents/llm_clients/model_catalog.py     (Groq models already listed)
cli/utils.py                                   (Groq already in CLI list)
```

---

## Success Checklist

- [ ] Phase 0 research findings documented in this plan
- [ ] `tests/conftest.py` includes `GROQ_API_KEY` in dummy-key fixture
- [ ] All new and existing tests pass (`pytest tests/ -v`)
- [ ] `tests/test_groq.py::TestGroqModelValidation` confirms catalog/validator alignment
- [ ] `tests/test_provider_registry.py` covers every legacy provider
- [ ] Live smoke test with a real Groq key succeeds
- [ ] Documentation under `docs/providers/` is complete and linked from README
- [ ] All commits are coherent and follow repo convention

---

## Estimated Timeline

| Phase | Work | Time |
|-------|------|------|
| 0 | Research + conftest update | 20 min |
| 1 | Verification tests (tasks 1–5) | 1.5 h |
| 2 | Registry + factory refactor | 1 h |
| 3 | Capabilities adjustments (conditional, often skipped) | 0–30 min |
| 4 | Documentation | 1.5 h |
| 5 | Smoke test + final verification | 30 min |

**Total:** ~5 hours (4.5 if capabilities are unchanged).

---

## Notes for the Implementer

- **Mocking philosophy:** Follow `test_minimax.py` and `test_capabilities.py`.
  Inspect bound kwargs and request payloads — do NOT mock HTTP. The repo
  uses `monkeypatch.setenv()` for env vars (per `conftest.py` convention),
  not `@patch.dict`.
- **Backward compatibility is sacrosanct.** The factory's public API
  signature is unchanged; every legacy provider call must keep working.
- **Base-URL duplication:** `_PROVIDER_BASE_URL` in `openai_client.py`
  and the `openai_compatible` dict in `provider_registry.py` will hold
  the same URLs. Keep them in sync; consider a follow-up to consolidate.
  Don't try to fix it in this PR — the duplication is contained and
  removing it cleanly requires a separate refactor.
- **Async tests are synchronous in this plan.** Task 4's "async" tests
  just verify method existence (`callable(getattr(llm, "ainvoke", ...))`),
  so `pytest-asyncio` is NOT a required dependency. If a future
  integration test actually awaits a Groq call, add `pytest-asyncio>=0.23`
  and set `asyncio_mode = "auto"` in `[tool.pytest.ini_options]`.

---

## Two Execution Options

**1. Subagent-Driven (recommended)** — Fresh subagent per task, review
between tasks, fastest iteration.  
**REQUIRED SUB-SKILL:** `superpowers:subagent-driven-development`

**2. Inline Execution** — Execute in this session with checkpoints.  
**REQUIRED SUB-SKILL:** `superpowers:executing-plans`

**Which approach?**
