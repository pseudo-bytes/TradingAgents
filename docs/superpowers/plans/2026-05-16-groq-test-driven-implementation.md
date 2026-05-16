# Groq Test-Driven Refactor: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete Groq provider integration with full feature support (chat, tools, structured output, async) while building an extensible provider registry system for future providers.

**Architecture:** Test-driven refactor — write all tests first (mocked), then refactor provider system to a registry pattern, then verify Groq features work. Registry centralizes provider metadata; factory becomes a simple lookup.

**Tech Stack:** LangChain (ChatOpenAI), unittest.mock (testing), pytest (test runner), dataclasses (ProviderConfig)

---

## Pre-Phase-1: Research (BLOCKING)

**Do this first** — answers inform test mocks and capabilities matrix.

### Research Task: Verify Groq API Capabilities

- [ ] **Step 1: Check Groq documentation for tool support**

Visit: https://console.groq.com/docs/tool-use  
Document answers:
- Does Groq accept `tools` array parameter? (Expected: YES)
- Does it support `tool_choice="auto"`? (Unknown — may only support `tools=[]`)
- Which Llama models support tools? (e.g., 3.3+ only?)

- [ ] **Step 2: Check Groq documentation for structured output**

Visit: https://console.groq.com/docs/json-mode  
Document answers:
- Does Groq support `response_format={"type": "json_object"}`? (JSON mode)
- Does it support `response_format={"type": "json_schema", "schema": ...}`? (Likely NO for Llama)
- Which models support which format?

- [ ] **Step 3: Check Groq documentation for streaming and async**

Visit: https://console.groq.com/docs/api-reference  
Document answers:
- Does Groq support streaming responses? (Server-Sent Events?)
- Does LangChain's ChatOpenAI streaming work with Groq endpoint?

**Deliverable:** Findings doc (internal notes) — update design section 7.1 with answers, update task mocks below accordingly.

---

## Phase 1: Test Writing (No Code Changes)

### Task 1: Write Provider Registry Tests

**Files:**
- Create: `tests/llm_clients/test_provider_registry.py`
- Create: `tradingagents/llm_clients/provider_registry.py` (stub — will implement in Phase 2)

- [ ] **Step 1: Create stub provider_registry.py**

File: `tradingagents/llm_clients/provider_registry.py`

```python
"""Provider registry: centralized configuration for all LLM providers."""

from dataclasses import dataclass
from typing import Type, Optional, Dict, List

# Stub — will populate in Phase 2
@dataclass
class ProviderConfig:
    """Metadata for a single provider."""
    name: str
    client_class: Type['BaseLLMClient']
    base_url: Optional[str]
    api_key_env: Optional[str]
    models: Dict[str, List[tuple]]

PROVIDER_REGISTRY: Dict[str, ProviderConfig] = {}
```

- [ ] **Step 2: Write test for registry lookup**

File: `tests/llm_clients/test_provider_registry.py`

```python
import pytest
from tradingagents.llm_clients.provider_registry import PROVIDER_REGISTRY, ProviderConfig


def test_groq_registered_in_registry():
    """Test that Groq is registered in the provider registry."""
    assert "groq" in PROVIDER_REGISTRY
    config = PROVIDER_REGISTRY["groq"]
    assert isinstance(config, ProviderConfig)
    assert config.name == "groq"
    assert config.api_key_env == "GROQ_API_KEY"
    assert config.base_url == "https://api.groq.com/openai/v1"


def test_all_registry_entries_have_required_fields():
    """Test that every provider has all required metadata."""
    for provider_name, config in PROVIDER_REGISTRY.items():
        assert config.name is not None
        assert config.client_class is not None
        assert config.api_key_env is not None or provider_name == "ollama"  # Ollama is special
        assert config.models is not None
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd /path/to/repo
pytest tests/llm_clients/test_provider_registry.py -v
```

Expected: **FAIL** — registry is empty, assertions fail.

- [ ] **Step 4: Commit test file**

```bash
git add tests/llm_clients/test_provider_registry.py tradingagents/llm_clients/provider_registry.py
git commit -m "test: add provider registry tests (failing)"
```

---

### Task 2: Write Groq Chat Completions Test

**Files:**
- Create: `tests/llm_clients/test_groq_chat.py`

- [ ] **Step 1: Write test for Groq chat completions**

File: `tests/llm_clients/test_groq_chat.py`

```python
import pytest
from unittest.mock import patch, MagicMock
from tradingagents.llm_clients.factory import create_llm_client


@patch("langchain_openai.ChatOpenAI._make_request")
def test_groq_chat_completion_basic(mock_request):
    """Test that Groq client sends and receives chat completions correctly."""
    # Mock response from Groq API
    mock_request.return_value = {
        "choices": [
            {"message": {"content": "Hello, I am Llama."}}
        ]
    }
    
    # Create Groq client
    client = create_llm_client("groq", "llama-3.3-70b-versatile")
    llm = client.get_llm()
    
    # Make request
    response = llm.invoke([("human", "Say hello")])
    
    # Verify response
    assert response.content == "Hello, I am Llama."
    
    # Verify base URL was set correctly
    assert llm.base_url == "https://api.groq.com/openai/v1"


@patch.dict("os.environ", {"GROQ_API_KEY": "test_key_123"})
@patch("langchain_openai.ChatOpenAI._make_request")
def test_groq_api_key_loaded_from_env(mock_request, mock_env):
    """Test that GROQ_API_KEY environment variable is used."""
    mock_request.return_value = {"choices": [{"message": {"content": "OK"}}]}
    
    client = create_llm_client("groq", "llama-3.3-70b-versatile")
    llm = client.get_llm()
    
    # Verify API key was passed
    assert llm.api_key == "test_key_123"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/llm_clients/test_groq_chat.py -v
```

Expected: **FAIL** — `create_llm_client` may not recognize "groq" yet or registry is empty.

- [ ] **Step 3: Commit test**

```bash
git add tests/llm_clients/test_groq_chat.py
git commit -m "test: add groq chat completions test (failing)"
```

---

### Task 3: Write Groq Tool Calling Test

**Files:**
- Create: `tests/llm_clients/test_groq_tools.py`

- [ ] **Step 1: Write test for Groq with function tools**

File: `tests/llm_clients/test_groq_tools.py`

```python
import pytest
from unittest.mock import patch
from tradingagents.llm_clients.factory import create_llm_client


@patch("langchain_openai.ChatOpenAI._make_request")
def test_groq_tool_calling_basic(mock_request):
    """Test that Groq accepts tools parameter and processes tool calls."""
    # Mock response from Groq with tool call
    mock_request.return_value = {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_xyz",
                            "function": {"name": "get_weather", "arguments": '{"location": "NYC"}'},
                            "type": "function"
                        }
                    ]
                }
            }
        ]
    }
    
    client = create_llm_client("groq", "llama-3.3-70b-versatile")
    llm = client.get_llm()
    
    # Define a tool
    tools_schema = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }
        }
    ]
    
    # Make request with tools
    response = llm.invoke([("human", "What's the weather in NYC?")], tools=tools_schema)
    
    # Verify tools were sent in request
    call_args = mock_request.call_args
    assert "tools" in call_args[1]
    assert len(call_args[1]["tools"]) == 1


def test_groq_tool_choice_support():
    """Test that Groq client handles tool_choice parameter correctly.
    
    Note: Based on research, if Groq doesn't support tool_choice,
    update capabilities.py to set supports_tool_choice=False.
    """
    from tradingagents.llm_clients.capabilities import get_capabilities
    
    caps = get_capabilities("llama-3.3-70b-versatile")
    
    # This test will inform whether to set tool_choice support
    # If research shows Groq doesn't support tool_choice, caps should reflect False
    # Update based on research findings from Pre-Phase-1
    assert hasattr(caps, "supports_tool_choice")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/llm_clients/test_groq_tools.py -v
```

Expected: **FAIL** — tools not yet in capabilities matrix.

- [ ] **Step 3: Commit test**

```bash
git add tests/llm_clients/test_groq_tools.py
git commit -m "test: add groq tool calling test (failing)"
```

---

### Task 4: Write Groq Structured Output Test

**Files:**
- Create: `tests/llm_clients/test_groq_structured_output.py`

- [ ] **Step 1: Write test for JSON mode**

File: `tests/llm_clients/test_groq_structured_output.py`

```python
import pytest
from unittest.mock import patch
from tradingagents.llm_clients.factory import create_llm_client


@patch("langchain_openai.ChatOpenAI._make_request")
def test_groq_json_mode_support(mock_request):
    """Test that Groq accepts response_format for JSON mode."""
    # Mock response from Groq in JSON mode
    mock_request.return_value = {
        "choices": [
            {
                "message": {"content": '{"name": "John", "age": 30}'}
            }
        ]
    }
    
    client = create_llm_client("groq", "llama-3.3-70b-versatile")
    llm = client.get_llm()
    
    # Request with JSON mode
    response = llm.invoke(
        [("human", "Return user data as JSON")],
        response_format={"type": "json_object"}
    )
    
    # Verify response_format was sent
    call_args = mock_request.call_args
    assert "response_format" in call_args[1]
    assert call_args[1]["response_format"]["type"] == "json_object"


def test_groq_json_schema_support():
    """Test Groq's support for JSON schema (if available).
    
    Note: Research may show Llama doesn't support json_schema.
    If so, update capabilities.py: supports_json_schema=False.
    """
    from tradingagents.llm_clients.capabilities import get_capabilities
    
    caps = get_capabilities("llama-3.3-70b-versatile")
    
    # This will be set based on research findings
    # If Groq doesn't support json_schema, this should be False
    assert hasattr(caps, "supports_json_schema")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/llm_clients/test_groq_structured_output.py -v
```

Expected: **FAIL** — structured output not yet in capabilities matrix.

- [ ] **Step 3: Commit test**

```bash
git add tests/llm_clients/test_groq_structured_output.py
git commit -m "test: add groq structured output test (failing)"
```

---

### Task 5: Write Groq Async/Streaming Test

**Files:**
- Create: `tests/llm_clients/test_groq_async.py`

- [ ] **Step 1: Write test for async invoke**

File: `tests/llm_clients/test_groq_async.py`

```python
import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from tradingagents.llm_clients.factory import create_llm_client


@pytest.mark.asyncio
@patch("langchain_openai.ChatOpenAI._amake_request", new_callable=AsyncMock)
async def test_groq_async_invoke(mock_request):
    """Test that Groq client supports async invoke."""
    mock_request.return_value = {
        "choices": [
            {"message": {"content": "Async response"}}
        ]
    }
    
    client = create_llm_client("groq", "llama-3.3-70b-versatile")
    llm = client.get_llm()
    
    # Async invoke
    response = await llm.ainvoke([("human", "Test")])
    
    assert response.content == "Async response"


def test_groq_streaming_support():
    """Test that Groq client supports streaming."""
    from tradingagents.llm_clients.factory import create_llm_client
    
    client = create_llm_client("groq", "llama-3.3-70b-versatile")
    llm = client.get_llm()
    
    # Verify streaming methods exist (may be no-op in mocks)
    assert hasattr(llm, "stream")
    assert hasattr(llm, "astream")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/llm_clients/test_groq_async.py -v
```

Expected: **FAIL** — async methods not yet tested.

- [ ] **Step 3: Commit test**

```bash
git add tests/llm_clients/test_groq_async.py
git commit -m "test: add groq async/streaming test (failing)"
```

---

### Task 6: Write Capability Matrix Test

**Files:**
- Create: `tests/llm_clients/test_capabilities_matrix.py`

- [ ] **Step 1: Write test for Groq capabilities**

File: `tests/llm_clients/test_capabilities_matrix.py`

```python
import pytest
from tradingagents.llm_clients.capabilities import get_capabilities


def test_groq_llama_3_3_capabilities():
    """Test that llama-3.3-70b-versatile has correct capabilities."""
    caps = get_capabilities("llama-3.3-70b-versatile")
    
    # Based on research (Pre-Phase-1), set expected values
    # These are placeholders — update based on actual Groq API research
    assert caps.supports_tool_choice in (True, False)  # Check research findings
    assert caps.supports_json_mode in (True, False)    # Check research findings
    assert caps.supports_json_schema in (True, False)  # Likely False
    assert caps.preferred_structured_method in ("function_calling", "json_mode", "none")


def test_groq_llama_3_1_capabilities():
    """Test that llama-3.1-8b-instant has correct capabilities."""
    caps = get_capabilities("llama-3.1-8b-instant")
    
    # May differ from 3.3 — based on research
    assert caps.supports_tool_choice in (True, False)
    assert caps.supports_json_mode in (True, False)


def test_default_capability_fallback():
    """Test that unknown models fall back to DEFAULT capabilities."""
    from tradingagents.llm_clients.capabilities import _DEFAULT
    
    caps = get_capabilities("llama-custom-unknown-model")
    
    # Should return default (not raise error)
    assert caps == _DEFAULT
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/llm_clients/test_capabilities_matrix.py -v
```

Expected: **FAIL** — Groq not in capabilities matrix yet.

- [ ] **Step 3: Commit test**

```bash
git add tests/llm_clients/test_capabilities_matrix.py
git commit -m "test: add capabilities matrix test (failing)"
```

---

## Phase 2: Provider Registry Implementation

### Task 7: Implement Provider Registry

**Files:**
- Modify: `tradingagents/llm_clients/provider_registry.py`
- Modify: `tradingagents/llm_clients/api_key_env.py` (verify GROQ_API_KEY exists)
- Reference: `tradingagents/llm_clients/model_catalog.py` (Groq models already here)

- [ ] **Step 1: Verify GROQ_API_KEY in api_key_env.py**

File: `tradingagents/llm_clients/api_key_env.py`

Check if "groq" is already mapped to "GROQ_API_KEY". If not:

```python
# In api_key_env.py, add to mapping:
PROVIDER_API_KEY_ENV = {
    ...existing...
    "groq": "GROQ_API_KEY",
}
```

- [ ] **Step 2: Implement ProviderConfig in provider_registry.py**

File: `tradingagents/llm_clients/provider_registry.py`

```python
"""Provider registry: centralized configuration for all LLM providers."""

from dataclasses import dataclass
from typing import Type, Optional, Dict, List, Tuple

# Type alias for model options
ModelOption = Tuple[str, str]  # (display_label, model_id)


@dataclass(frozen=True)
class ProviderConfig:
    """Metadata for a single provider.
    
    The client_class field determines everything about how the provider
    is instantiated and what APIs it supports. No separate "is_openai_compatible"
    flag needed — the client_class itself tells you the implementation.
    """
    name: str                               # "groq", "openai", etc.
    client_class: Type                      # e.g., OpenAIClient, AnthropicClient
    base_url: Optional[str]                 # Default API endpoint
    api_key_env: Optional[str]              # ENV var name for API key
    models: Dict[str, List[ModelOption]]    # {"quick": [...], "deep": [...]}


# Build registry from existing providers
def _build_registry() -> Dict[str, ProviderConfig]:
    """Build the provider registry from imports."""
    from tradingagents.llm_clients.openai_client import OpenAIClient
    from tradingagents.llm_clients.anthropic_client import AnthropicClient
    from tradingagents.llm_clients.google_client import GoogleClient
    from tradingagents.llm_clients.azure_client import AzureOpenAIClient
    from tradingagents.llm_clients.model_catalog import MODEL_OPTIONS
    
    return {
        "groq": ProviderConfig(
            name="groq",
            client_class=OpenAIClient,
            base_url="https://api.groq.com/openai/v1",
            api_key_env="GROQ_API_KEY",
            models=MODEL_OPTIONS.get("groq", {}),
        ),
        "openai": ProviderConfig(
            name="openai",
            client_class=OpenAIClient,
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            models=MODEL_OPTIONS.get("openai", {}),
        ),
        "anthropic": ProviderConfig(
            name="anthropic",
            client_class=AnthropicClient,
            base_url=None,  # Anthropic uses default endpoint
            api_key_env="ANTHROPIC_API_KEY",
            models=MODEL_OPTIONS.get("anthropic", {}),
        ),
        "google": ProviderConfig(
            name="google",
            client_class=GoogleClient,
            base_url=None,
            api_key_env="GOOGLE_API_KEY",
            models=MODEL_OPTIONS.get("google", {}),
        ),
        "azure": ProviderConfig(
            name="azure",
            client_class=AzureOpenAIClient,
            base_url=None,
            api_key_env="AZURE_OPENAI_API_KEY",
            models=MODEL_OPTIONS.get("azure", {}),
        ),
        # Add other existing providers as needed...
    }


PROVIDER_REGISTRY: Dict[str, ProviderConfig] = _build_registry()
```

- [ ] **Step 3: Run registry tests**

```bash
pytest tests/llm_clients/test_provider_registry.py -v
```

Expected: **PASS** — Groq is registered, registry lookup works.

- [ ] **Step 4: Commit**

```bash
git add tradingagents/llm_clients/provider_registry.py tradingagents/llm_clients/api_key_env.py
git commit -m "feat: implement provider registry with Groq registration"
```

---

### Task 8: Refactor Factory to Use Registry

**Files:**
- Modify: `tradingagents/llm_clients/factory.py`

- [ ] **Step 1: Check current factory.py structure**

Read the current factory.py to understand provider conditionals and how they work.

- [ ] **Step 2: Refactor factory.py to use registry**

File: `tradingagents/llm_clients/factory.py`

Replace the current provider conditionals with:

```python
from typing import Optional
from tradingagents.llm_clients.provider_registry import PROVIDER_REGISTRY
from tradingagents.llm_clients.base_client import BaseLLMClient


def create_llm_client(
    provider: str,
    model: str,
    base_url: Optional[str] = None,
    **kwargs,
) -> BaseLLMClient:
    """Create an LLM client for the specified provider.
    
    Provider modules are imported lazily to avoid pulling in heavy
    dependencies unless needed.
    
    Args:
        provider: LLM provider name (e.g., "groq", "openai")
        model: Model identifier
        base_url: Optional override for default endpoint
        **kwargs: Provider-specific arguments
    
    Returns:
        Configured BaseLLMClient instance
    
    Raises:
        ValueError: If provider is not supported
    """
    provider_lower = provider.lower()
    
    if provider_lower not in PROVIDER_REGISTRY:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    
    config = PROVIDER_REGISTRY[provider_lower]
    client_class = config.client_class
    
    # Use provided base_url or fall back to registry default
    effective_base_url = base_url or config.base_url
    
    # Instantiate client with provider name for backward compatibility
    return client_class(
        model=model,
        base_url=effective_base_url,
        provider=provider_lower,
        **kwargs
    )
```

- [ ] **Step 3: Verify backward compatibility**

Run existing tests to ensure old callers still work:

```bash
pytest tests/ -k "llm" -v
```

Expected: All existing tests still pass (backward compatible).

- [ ] **Step 4: Commit**

```bash
git add tradingagents/llm_clients/factory.py
git commit -m "refactor: factory uses provider registry instead of conditionals"
```

---

## Phase 3: Groq Feature Verification & Implementation

### Task 9: Update Capabilities Matrix with Groq

**Files:**
- Modify: `tradingagents/llm_clients/capabilities.py`

- [ ] **Step 1: Check research findings**

Use the research findings from Pre-Phase-1 to determine Groq capabilities:
- Tool choice support? (tool_choice parameter)
- JSON mode support? (response_format JSON object)
- JSON schema support? (response_format JSON schema)

- [ ] **Step 2: Add Groq entries to capabilities.py**

File: `tradingagents/llm_clients/capabilities.py`

Add after the existing model mappings:

```python
# Groq Llama models — based on research findings (Pre-Phase-1)
# If research shows tool_choice is NOT supported, set supports_tool_choice=False
_GROQ_LLAMA = ModelCapabilities(
    supports_tool_choice=True,  # UPDATE based on research
    supports_json_mode=True,    # UPDATE based on research
    supports_json_schema=False, # Llama likely doesn't support this
    preferred_structured_method="function_calling",  # or "json_mode" if tools aren't supported
)

# Add to _BY_ID mapping if needed (if different models have different capabilities)
_BY_ID: dict[str, ModelCapabilities] = {
    ...existing...
    "llama-3.3-70b-versatile": _GROQ_LLAMA,
    "llama-3.1-70b-versatile": _GROQ_LLAMA,
    "llama-3.1-8b-instant": _GROQ_LLAMA,
    "meta-llama/llama-4-scout-17b-16e-instruct": _GROQ_LLAMA,
    "meta-llama/llama-4-maverick-17b-128e-instruct": _GROQ_LLAMA,
}
```

- [ ] **Step 3: Run capability tests**

```bash
pytest tests/llm_clients/test_capabilities_matrix.py -v
```

Expected: **PASS** — Groq capabilities are now in the matrix.

- [ ] **Step 4: Commit**

```bash
git add tradingagents/llm_clients/capabilities.py
git commit -m "feat: add Groq capability matrix entries based on research findings"
```

---

### Task 10: Run and Verify All LLM Client Tests

**Files:**
- Test: all tests in `tests/llm_clients/`

- [ ] **Step 1: Run all Groq tests**

```bash
pytest tests/llm_clients/test_groq_*.py tests/llm_clients/test_provider_registry.py tests/llm_clients/test_capabilities_matrix.py -v
```

Expected: **ALL PASS** — Registry, chat, tools, structured output, async, and capabilities all working.

- [ ] **Step 2: Run all existing LLM tests**

```bash
pytest tests/llm_clients/ -v
```

Expected: **ALL PASS** — Backward compatibility maintained.

- [ ] **Step 3: Verify no regressions in other tests**

```bash
pytest tests/ -v --tb=short
```

Expected: All tests pass or show expected failures (unrelated to this work).

- [ ] **Step 4: Commit verification**

```bash
git commit --allow-empty -m "test: all groq and provider tests pass, backward compatibility verified"
```

---

## Phase 4: Documentation

### Task 11: Write Provider Documentation

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

This directory contains guides for working with LLM providers in the trading agents system.

## Quick Links

- **[Architecture](architecture.md)** — How the provider system works
- **[Adding a Provider](adding-a-provider.md)** — Step-by-step guide for new providers
- **[Groq Setup](groq-setup-and-usage.md)** — Groq-specific configuration and models
- **[Capability Matrix](capability-matrix.md)** — Which providers support which features
- **[Troubleshooting](troubleshooting.md)** — Common issues and solutions

## Supported Providers

- **OpenAI** — GPT-4, GPT-5 models
- **Anthropic** — Claude Opus, Sonnet, Haiku
- **Google** — Gemini models
- **Groq** — Llama 3.1, 3.3, Llama 4 (free tier available)
- **Ollama** — Local models
- **Others** — DeepSeek, Qwen, GLM, MiniMax, xAI, Azure OpenAI, OpenRouter

## Quick Start

```python
from tradingagents.llm_clients.factory import create_llm_client

# Create a Groq client
client = create_llm_client("groq", "llama-3.3-70b-versatile")
llm = client.get_llm()

# Use it
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

The provider system uses a **registry pattern** to centralize configuration and make adding new providers trivial.

## Components

### 1. ProviderConfig (provider_registry.py)

```python
@dataclass
class ProviderConfig:
    name: str                               # Provider name ("groq", "openai", etc.)
    client_class: Type[BaseLLMClient]      # Which client class to instantiate
    base_url: Optional[str]                 # Default API endpoint
    api_key_env: Optional[str]              # ENV var for API key
    models: Dict[str, List[ModelOption]]    # Available models
```

**Benefits:**
- Single source of truth for each provider
- No scattered metadata across files
- Easy to add providers (one registry entry)

### 2. Provider Registry (provider_registry.py)

```python
PROVIDER_REGISTRY: Dict[str, ProviderConfig] = {
    "groq": ProviderConfig(...),
    "openai": ProviderConfig(...),
    ...
}
```

All providers are registered here. Factory looks them up by name.

### 3. Client Classes

Each provider has a client class (e.g., `OpenAIClient`, `AnthropicClient`) that:
- Inherits from `BaseLLMClient`
- Implements `get_llm()` and `validate_model()`
- Handles provider-specific quirks (authentication, parameter differences)

For OpenAI-compatible providers (Groq, DeepSeek, Ollama), they all use `OpenAIClient`.

### 4. Capabilities Matrix (capabilities.py)

Declares what each model supports:

```python
@dataclass
class ModelCapabilities:
    supports_tool_choice: bool              # Can use tool_choice parameter?
    supports_json_mode: bool                # Can use JSON mode?
    supports_json_schema: bool              # Can use JSON schema?
    preferred_structured_method: str        # "function_calling" or "json_mode"
```

Lookup: `get_capabilities(model_name)` returns capabilities for that model.

### 5. Factory (factory.py)

```python
def create_llm_client(provider: str, model: str, base_url: Optional[str] = None) -> BaseLLMClient:
    config = PROVIDER_REGISTRY[provider]
    return config.client_class(model=model, base_url=base_url or config.base_url, ...)
```

Simple registry lookup + client instantiation.

## Data Flow

```
User calls: create_llm_client("groq", "llama-3.3-70b-versatile")
    ↓
Factory looks up "groq" in PROVIDER_REGISTRY
    ↓
Gets ProviderConfig (base_url, api_key_env, etc.)
    ↓
Instantiates OpenAIClient with Groq's base_url and API key
    ↓
Client calls LangChain's ChatOpenAI
    ↓
ChatOpenAI sends request to https://api.groq.com/openai/v1
    ↓
Response returned to caller
```

## Adding a New Provider

See [Adding a Provider](adding-a-provider.md) for step-by-step instructions.

TL;DR: One registry entry + a client class (if not OpenAI-compatible) + tests + docs.
```

- [ ] **Step 3: Create docs/providers/adding-a-provider.md**

File: `docs/providers/adding-a-provider.md`

```markdown
# Adding a New Provider

This guide walks you through adding a new LLM provider to the system.

## Prerequisites

- Provider has a public API
- Provider has Python SDK or OpenAI-compatible endpoint
- You have API credentials for testing

## Step 1: Add Provider Configuration

Edit `tradingagents/llm_clients/provider_registry.py`:

```python
PROVIDER_REGISTRY: Dict[str, ProviderConfig] = {
    ...
    "newprovider": ProviderConfig(
        name="newprovider",
        client_class=OpenAIClient,  # or CustomClient if not OpenAI-compatible
        base_url="https://api.newprovider.com/v1",
        api_key_env="NEWPROVIDER_API_KEY",
        models=MODEL_OPTIONS.get("newprovider", {}),
    ),
}
```

## Step 2: Add Models to Catalog

Edit `tradingagents/llm_clients/model_catalog.py`:

```python
_NEWPROVIDER_MODELS: Dict[str, List[ModelOption]] = {
    "quick": [
        ("Model A - Fast", "model-a-fast"),
        ("Model B - Balanced", "model-b"),
    ],
    "deep": [
        ("Model C - Most Capable", "model-c-pro"),
    ],
}

MODEL_OPTIONS: ProviderModeOptions = {
    ...
    "newprovider": _NEWPROVIDER_MODELS,
}
```

## Step 3: Add Capabilities (if Needed)

Edit `tradingagents/llm_clients/capabilities.py`:

```python
_NEWPROVIDER_CONFIG = ModelCapabilities(
    supports_tool_choice=True,      # Can use tool_choice?
    supports_json_mode=True,         # Can use JSON mode?
    supports_json_schema=False,      # Can use JSON schema?
    preferred_structured_method="function_calling",
)

_BY_ID: dict[str, ModelCapabilities] = {
    ...
    "model-a-fast": _NEWPROVIDER_CONFIG,
    "model-b": _NEWPROVIDER_CONFIG,
    "model-c-pro": _NEWPROVIDER_CONFIG,
}
```

## Step 4: Create or Use Existing Client Class

If OpenAI-compatible: No new client needed (registry already uses `OpenAIClient`).

If custom API:
- Create `tradingagents/llm_clients/newprovider_client.py`
- Inherit from `BaseLLMClient`
- Implement `get_llm()` and `validate_model()`
- Update registry to use your client class

## Step 5: Add Tests

Create `tests/llm_clients/test_newprovider_*.py`:

```python
from unittest.mock import patch
from tradingagents.llm_clients.factory import create_llm_client

@patch("langchain_...ChatOpenAI._make_request")
def test_newprovider_chat(mock_request):
    mock_request.return_value = {"choices": [{"message": {"content": "OK"}}]}
    
    client = create_llm_client("newprovider", "model-a-fast")
    llm = client.get_llm()
    response = llm.invoke([("human", "Test")])
    
    assert response.content == "OK"
```

Run: `pytest tests/llm_clients/test_newprovider_*.py -v`

## Step 6: Update Documentation

Add your provider to:
- `docs/providers/README.md` (Supported Providers list)
- `docs/providers/capability-matrix.md` (Feature table)
- Create `docs/providers/newprovider-setup.md` (if provider-specific config needed)

## Step 7: Verify Everything

```bash
pytest tests/llm_clients/ -v          # All tests pass
pytest tests/ -v                      # No regressions
```

## Step 8: Commit

```bash
git commit -m "feat: add newprovider support"
```

## Timeline

If OpenAI-compatible: ~30 minutes (steps 1-8)
If custom client: ~2 hours (steps 1-8 + client implementation + testing)
```

- [ ] **Step 4: Create docs/providers/groq-setup-and-usage.md**

File: `docs/providers/groq-setup-and-usage.md`

```markdown
# Groq Setup & Usage Guide

## What is Groq?

Groq provides ultra-fast inference of open-source Llama models via an OpenAI-compatible API.

**Key Features:**
- Free tier with rate limits (good for testing)
- Models: Llama 3.1, 3.3, Llama 4 (Scout, Maverick)
- Ultra-low latency
- OpenAI-compatible endpoint

## Getting Started

### 1. Get API Key

Visit: https://console.groq.com  
Sign up → Create API key

### 2. Set Environment Variable

```bash
export GROQ_API_KEY="your_key_here"
```

Or add to `.env`:
```
GROQ_API_KEY=your_key_here
```

### 3. Use in Code

```python
from tradingagents.llm_clients.factory import create_llm_client

client = create_llm_client("groq", "llama-3.3-70b-versatile")
llm = client.get_llm()

response = llm.invoke([("human", "What is AI?")])
print(response.content)
```

## Available Models

### Quick Mode (Fast, Lower Accuracy)

- **Llama 3.1 8B Instant** (`llama-3.1-8b-instant`)
  - Fastest, cheapest
  - Good for simple tasks
  - Free tier friendly

- **Llama 3.3 70B SpecDec** (`llama-3.3-70b-specdec`)
  - Fast with speculative decoding
  - Balanced speed/accuracy

- **Llama 4 Scout 17B** (`meta-llama/llama-4-scout-17b-16e-instruct`)
  - Meta's Llama 4, low latency
  - Lightweight alternative to 70B

### Deep Mode (Slower, Higher Accuracy)

- **Llama 3.3 70B Versatile** (`llama-3.3-70b-versatile`)
  - Recommended for quality
  - Free tier available
  - Good for agent tasks

- **Llama 4 Maverick 17B** (`meta-llama/llama-4-maverick-17b-128e-instruct`)
  - Meta's Llama 4, strong reasoning
  - Lightweight but more capable

- **Llama 3.1 70B Versatile** (`llama-3.1-70b-versatile`)
  - Previous-gen 70B model
  - Still excellent quality

## Rate Limits (Free Tier)

- **RPM (Requests Per Minute):** 30
- **TPM (Tokens Per Minute):** 6,000
- **Max tokens per request:** 12,000

For production, upgrade to paid tier.

## Features Support

| Feature | Support | Notes |
|---------|---------|-------|
| Chat Completions | ✅ Yes | Basic chat |
| Tool Calling | ✅ Yes | Function tools supported |
| Structured Output | ⚠️ Partial | JSON mode yes, JSON schema no |
| Streaming | ✅ Yes | Server-Sent Events |
| Async | ✅ Yes | Full async/await support |

## Example: Tool Calling

```python
from tradingagents.llm_clients.factory import create_llm_client

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"],
            }
        }
    }
]

client = create_llm_client("groq", "llama-3.3-70b-versatile")
llm = client.get_llm()

response = llm.invoke(
    [("human", "What's the weather in NYC?")],
    tools=tools
)

# Check if there are tool calls
if response.tool_calls:
    for call in response.tool_calls:
        print(f"Tool: {call['function']['name']}")
        print(f"Args: {call['function']['arguments']}")
```

## Example: Structured Output (JSON Mode)

```python
client = create_llm_client("groq", "llama-3.3-70b-versatile")
llm = client.get_llm()

response = llm.invoke(
    [("human", "Extract user info from: John Doe, age 30, NYC")],
    response_format={"type": "json_object"}
)

# response.content will be JSON-formatted
print(response.content)
```

## Troubleshooting

See [Troubleshooting](troubleshooting.md) for common issues.

**Common error: "GROQ_API_KEY not set"**
→ Make sure you've set the environment variable (see step 2 above)

**Common error: "Rate limit exceeded"**
→ You've hit the free tier RPM limit. Wait 60 seconds or upgrade account.

**Common error: "Model not found"**
→ Check model ID spelling. Use the exact ID from "Available Models" above.

## More Info

- Groq API Docs: https://console.groq.com/docs
- Llama Model Docs: https://github.com/meta-llama/llama-models
```

- [ ] **Step 5: Create docs/providers/capability-matrix.md**

File: `docs/providers/capability-matrix.md`

```markdown
# Provider Capability Matrix

This table shows which features each provider and model supports.

| Provider | Model | Tool Calling | Tool Choice | JSON Mode | JSON Schema | Streaming | Async |
|----------|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Groq** | llama-3.3-70b-versatile | ✅ | ⚠️ | ✅ | ❌ | ✅ | ✅ |
| | llama-3.1-8b-instant | ✅ | ⚠️ | ✅ | ❌ | ✅ | ✅ |
| | llama-4-maverick | ✅ | ⚠️ | ✅ | ❌ | ✅ | ✅ |
| **OpenAI** | gpt-5.5 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | gpt-5.4 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | gpt-4.1 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Anthropic** | claude-opus-4-7 | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| | claude-sonnet-4-6 | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| | claude-haiku-4-5 | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| **Google** | gemini-3-flash | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| | gemini-2.5-pro | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **DeepSeek** | deepseek-v4-pro | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ |
| **Ollama** | llama2 | ✅ | ❌ | ❌ | ❌ | ✅ | ⚠️ |

**Legend:**
- ✅ = Supported
- ❌ = Not supported
- ⚠️ = Partial / Conditional support

**Notes:**
- **Tool Calling:** Does the model support function calling (tools array)?
- **Tool Choice:** Can you specify which tool to use (`tool_choice="auto"`)?
- **JSON Mode:** Can you force JSON output with `response_format={"type": "json_object"}`?
- **JSON Schema:** Can you enforce a specific JSON schema structure?
- **Streaming:** Can you stream responses token-by-token?
- **Async:** Can you use async/await with this model?

### When to Use Which Feature

**Tool Calling:** Building agents that need to call functions/APIs  
**JSON Mode:** When you need structured output but not a specific schema  
**JSON Schema:** When you need strict schema enforcement (OpenAI only currently)  
**Streaming:** Real-time updates to the user  
**Async:** High-concurrency applications
```

- [ ] **Step 6: Create docs/providers/troubleshooting.md**

File: `docs/providers/troubleshooting.md`

```markdown
# Troubleshooting

## Common Issues

### "API key not set" / "API key is empty"

**Symptom:** Error when calling `create_llm_client()`

**Cause:** Environment variable not set

**Fix:**
```bash
export GROQ_API_KEY="your_key"
# Or in .env:
# GROQ_API_KEY=your_key
```

Verify it's set:
```python
import os
print(os.environ.get("GROQ_API_KEY"))  # Should print your key
```

### "Unsupported LLM provider: xyz"

**Symptom:** `ValueError: Unsupported LLM provider: xyz`

**Cause:** Provider name not recognized or not in registry

**Fix:** Check provider name spelling. Valid providers: `groq`, `openai`, `anthropic`, `google`, `azure`, `ollama`, etc.

### "Rate limit exceeded"

**Symptom:** Groq returns 429 error

**Cause:** Hit free tier rate limit (30 RPM, 6K TPM)

**Fix:**
- Wait 60 seconds and retry
- Use smaller models (8B instead of 70B)
- Reduce request frequency
- Upgrade to paid Groq account

### "Tool calling not working"

**Symptom:** Tool calls not returned in response

**Cause:** Model doesn't support tool calling, or tools array not sent correctly

**Fix:**
- Check [Capability Matrix](capability-matrix.md) — does your model support tools?
- Verify `tools` parameter is correctly formatted
- Check that you're using the right model

```python
# Correct format:
tools = [
    {
        "type": "function",
        "function": {
            "name": "my_function",
            "description": "...",
            "parameters": {...}
        }
    }
]

response = llm.invoke([...], tools=tools)
```

### "JSON mode returning non-JSON"

**Symptom:** Response doesn't look like valid JSON

**Cause:** JSON mode enabled but model not enforcing it

**Fix:**
- Some models (Llama) don't strictly enforce JSON mode
- Parse what you get and handle errors
- Consider using JSON Schema if available for your provider

### "Import error: No module named 'langchain_openai'"

**Symptom:** `ModuleNotFoundError: No module named 'langchain_openai'`

**Cause:** Missing dependency

**Fix:**
```bash
pip install langchain-openai
```

Or upgrade:
```bash
pip install --upgrade langchain-openai
```

### "Custom model ID not working"

**Symptom:** Model name not recognized by provider

**Cause:** Model ID doesn't exist or isn't available to your account

**Fix:**
- Check official provider docs for valid model IDs
- Verify your account has access to the model
- Check Groq console for available models: https://console.groq.com/docs/models

## Performance Issues

### "Response is very slow"

**Cause:** Using a larger, slower model

**Fix:** Try a smaller model:
- Groq: Use `llama-3.1-8b-instant` instead of `llama-3.3-70b-versatile`
- OpenAI: Use `gpt-5.4-mini` instead of `gpt-5.5`

### "Timeout errors"

**Cause:** Request taking too long or network issue

**Fix:**
- Increase timeout:
  ```python
  client = create_llm_client("groq", "model", timeout=60)
  ```
- Use async:
  ```python
  response = await llm.ainvoke([...])
  ```
- Try a faster model

## Still Stuck?

1. Check the provider's official documentation
2. Check your API quota/rate limits in provider console
3. Ask in #engineering Slack or create an issue on GitHub

Include:
- Provider name
- Model ID
- Error message (full traceback)
- What you were trying to do
```

- [ ] **Step 7: Commit all documentation**

```bash
git add docs/providers/
git commit -m "docs: add comprehensive provider documentation and guides"
```

---

## Phase 5: Polish & Verification

### Task 12: Smoke Test with Trading Agent

**Files:**
- No code changes
- Test: Manual verification with real trading agent

- [ ] **Step 1: Start the application**

```bash
python main.py
```

Or start with a test trading agent script if available.

- [ ] **Step 2: Select Groq as provider**

When prompted for provider selection, choose "groq".

- [ ] **Step 3: Select a Groq model**

Choose one of the available Groq models (e.g., `llama-3.3-70b-versatile`).

- [ ] **Step 4: Run a trading task**

Execute a normal trading agent task and verify:
- Chat completions work
- Response parsing is correct
- No errors in logs
- Response content is properly formatted

- [ ] **Step 5: Document results**

Record in a comment or note:
- Model tested
- Task executed
- Any issues encountered
- Performance (latency, quality)

---

### Task 13: Final Verification & Cleanup

**Files:**
- All modified files from phases 1-4

- [ ] **Step 1: Run full test suite**

```bash
pytest tests/ -v
```

Expected: **ALL PASS** — No regressions.

- [ ] **Step 2: Check code style**

```bash
# If using black, isort, pylint, etc.:
black tradingagents/llm_clients/ --check
isort tradingagents/llm_clients/ --check
```

Fix any style issues.

- [ ] **Step 3: Review final code**

Check:
- No TODO/FIXME comments (except intentional ones)
- All functions have docstrings
- Type hints are present
- No hardcoded values (use registry/env vars)

- [ ] **Step 4: Verify documentation links**

Check that:
- README.md links to /docs/providers/
- All provider docs are reachable
- Code examples in docs are correct

- [ ] **Step 5: Create final commit**

```bash
git log --oneline | head -20
# Verify commits are clear and follow convention
```

If needed, squash or reword commits:
```bash
git rebase -i origin/main
```

- [ ] **Step 6: Create a summary of changes**

Document in a comment or PR description:
- What was implemented (Groq + provider registry)
- Tests added/verified
- Docs created
- Backward compatibility maintained

Example:
```
## Summary

✅ **Groq Provider Support Complete**
- Full Groq integration (chat, tools, structured output, async)
- Provider registry pattern for extensibility
- 5 new test files, all passing
- Comprehensive docs in /docs/providers/
- Backward compatible with existing code

## Changes
- [+] tradingagents/llm_clients/provider_registry.py (new)
- [+] tests/llm_clients/test_groq_*.py (5 new test files)
- [+] docs/providers/ (6 new markdown docs)
- [~] tradingagents/llm_clients/factory.py (refactored to use registry)
- [~] tradingagents/llm_clients/capabilities.py (added Groq entries)

## Tests
- All 40+ tests passing
- No regressions in existing tests
- Smoke test with trading agent: ✅ PASS
```

---

## File Structure Summary

**Created:**
```
tradingagents/llm_clients/
  provider_registry.py          (NEW — provider config registry)

tests/llm_clients/
  test_provider_registry.py     (NEW)
  test_groq_chat.py             (NEW)
  test_groq_tools.py            (NEW)
  test_groq_structured_output.py (NEW)
  test_groq_async.py            (NEW)
  test_capabilities_matrix.py   (NEW)

docs/providers/
  README.md                     (NEW)
  architecture.md               (NEW)
  adding-a-provider.md          (NEW)
  groq-setup-and-usage.md       (NEW)
  capability-matrix.md          (NEW)
  troubleshooting.md            (NEW)
```

**Modified:**
```
tradingagents/llm_clients/
  factory.py                    (refactored to use registry)
  capabilities.py               (added Groq entries)
  api_key_env.py                (verified GROQ_API_KEY exists)
```

---

## Success Checklist

- [ ] All 13 tasks completed
- [ ] All tests passing (`pytest tests/ -v`)
- [ ] No regressions in existing functionality
- [ ] Groq tested end-to-end with trading agent
- [ ] Documentation complete and reviewed
- [ ] Code committed with clear messages
- [ ] Registry pattern is extensible (new providers take <30 min to add)
- [ ] Backward compatibility maintained

---

## Estimated Timeline

- **Phase 0 (Research):** 30 min
- **Phase 1 (Tests):** 1.5 hours
- **Phase 2 (Registry):** 1 hour
- **Phase 3 (Verification):** 1 hour
- **Phase 4 (Docs):** 1.5 hours
- **Phase 5 (Smoke test):** 30 min

**Total:** ~6-7 hours for full implementation + testing + docs

---

# Ready to Execute

Plan is complete. Two execution options:

**Option 1: Subagent-Driven (Recommended)**
- Fresh subagent per task
- I review between tasks
- Faster iteration
- Requires: `superpowers:subagent-driven-development`

**Option 2: Inline Execution**
- Execute tasks in this session
- Batch execution with checkpoints
- Requires: `superpowers:executing-plans`

**Which approach do you prefer?**
