# Groq Provider: Test-Driven Refactor & Extensible Provider System

**Date:** 2026-05-16  
**Goal:** Complete Groq integration (chat, tools, structured output, async) while building an extensible provider system that makes adding future providers trivial.  
**Approach:** Test-driven refactor — write tests first, refactor system to pass them, document as we go.

---

## 1. Current State

**Groq Status:** Partially integrated
- ✅ Listed in OpenAI-compatible providers (factory.py)
- ✅ Base URL configured (openai_client.py: `https://api.groq.com/openai/v1`)
- ✅ Models catalogued (Llama 3.1, 3.3, Llama 4 Scout/Maverick)
- ❓ Tool calling support — needs verification
- ❓ Structured output (JSON mode/schema) — needs verification
- ❓ Async/streaming — needs verification

**Provider System Status:** Implicit patterns
- Factory uses hardcoded provider conditionals
- Capabilities are implicit (no explicit matrix)
- No formal "provider interface" beyond BaseLLMClient
- Adding a new provider requires editing factory.py, api_key_env.py, model_catalog.py

---

## 2. Design: Extensible Provider System

### 2.1 Architecture

**New: Provider Registry Pattern**

```python
# tradingagents/llm_clients/provider_registry.py
from dataclasses import dataclass
from typing import Type, Optional, Dict, List

@dataclass
class ProviderConfig:
    """Metadata for a single provider.
    
    client_class determines everything (OpenAIClient for OpenAI-compatible,
    AnthropicClient for Anthropic, etc.); no need for is_openai_compatible flag.
    """
    name: str                          # "groq", "openai", etc.
    client_class: Type[BaseLLMClient]  # The client to instantiate (determines impl)
    base_url: Optional[str]            # Default API endpoint (can be overridden)
    api_key_env: Optional[str]         # ENV var for API key
    models: Dict[str, List[ModelOption]]  # Quick/deep model lists

PROVIDER_REGISTRY: Dict[str, ProviderConfig] = {
    "groq": ProviderConfig(
        name="groq",
        client_class=OpenAIClient,
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        models=_GROQ_MODELS,
    ),
    "openai": ProviderConfig(...),
    # ... etc
}
```

**Benefits:**
- Single source of truth for each provider
- Easy to add providers: register once in the registry
- Metadata is explicit, not scattered across files
- Factory becomes simple: `registry.get(provider)` instead of conditionals

### 2.2 Refactored Files

| File | Change | Reason |
|------|--------|--------|
| `provider_registry.py` | **NEW** | Centralize provider config; eliminate scattered metadata |
| `factory.py` | Refactor to use registry | Remove provider conditionals |
| `capabilities.py` | Add Groq feature matrix | Explicitly declare what Groq supports |
| `openai_client.py` | No change (already handles OpenAI-compatible) | Groq inherits support |
| `api_key_env.py` | Verify Groq entry | Should already have GROQ_API_KEY |
| `model_catalog.py` | No change (Groq models already listed) | — |

### 2.3 Groq Feature Verification Matrix

**Capabilities to verify (in tests):**

| Feature | Status | Notes |
|---------|--------|-------|
| **Chat Completions** | ✅ Likely working | Basic OpenAI-compat endpoint |
| **Tool Calling** | ? TBD | Does Groq accept `tools=` parameter? Does it support `tool_choice`? |
| **Structured Output** | ? TBD | JSON mode? JSON schema? (Llama models may have limits) |
| **Async Invoke** | ? TBD | LangChain ChatOpenAI async support |
| **Streaming** | ? TBD | Does Groq support streaming? LangChain stream methods? |

**Research before Phase 1:** Check Groq API docs for:
- Tool calling support (tools array + tool_choice parameter)
- Structured output (JSON mode vs JSON schema — Llama models may not support schema)
- Streaming (does Groq support streaming responses?)

**Capabilities.py update:** Based on findings, add Groq models to capabilities matrix:
```python
# If all Groq models support tools + JSON mode (best case):
_DEFAULT = ModelCapabilities(
    supports_tool_choice=True,
    supports_json_mode=True,
    supports_json_schema=False,  # Llama may not support schema
    preferred_structured_method="function_calling",
)

# If some models differ (e.g., 8B vs 70B):
_BY_ID: dict[str, ModelCapabilities] = {
    "llama-3.1-8b-instant": ModelCapabilities(...),
    "llama-3.3-70b-versatile": ModelCapabilities(...),
    # ... etc
}
```

**Test coverage:** One integration test per feature; each test is isolated and can run without API calls (mocked).

---

## 3. Test Structure

### 3.1 Test Files Created

```
tests/llm_clients/
├── test_provider_registry.py        # Registry lookup, registration
├── test_groq_chat.py                # Basic chat completions
├── test_groq_tools.py               # Function tool calling
├── test_groq_structured_output.py   # JSON mode & schema
├── test_groq_async.py               # Streaming & async invoke
└── test_capabilities_matrix.py      # Feature matrix validation
```

### 3.2 Test Strategy

**Mocking approach:** Tests use `unittest.mock.patch()` to mock the OpenAI client's HTTP layer (no live API calls):
```python
# Example mocking pattern:
@patch("langchain_openai.ChatOpenAI._make_request")
def test_groq_tools(mock_request):
    # Mock fixture: recorded response from Groq API for tool call
    mock_request.return_value = {
        "choices": [{"message": {"tool_calls": [...]}}]
    }
    
    client = create_llm_client("groq", "llama-3.3-70b-versatile")
    response = client.invoke([...with tools...])
    
    # Verify request format (tools array sent)
    assert "tools" in mock_request.call_args[1]
```

**Test isolation:** Each test is independent; mocks are function-scoped (no cross-test pollution).

**Test order (by implementation priority):**
1. Registry tests (most basic)
2. Chat completions (foundation)
3. Tool calling
4. Structured output
5. Async/streaming

Each test:
- Mocks the OpenAI client's HTTP layer (no live calls)
- Verifies client instantiation with Groq config from registry
- Checks request format sent to Groq (e.g., tools array, response_format field)
- Validates response parsing (content normalization)
- **CI runs offline:** No API keys needed in test environment

---

## 4. Documentation Structure

**Location:** `/docs/providers/` (markdown files)

### 4.1 Files to Create

```
docs/providers/
├── README.md                          # Overview & quick links
├── architecture.md                    # System design, diagrams, concepts
├── adding-a-provider.md               # Step-by-step: how to add a new provider
├── groq-setup-and-usage.md            # Groq-specific: setup, models, quirks
├── capability-matrix.md               # Feature table: which providers support what
└── troubleshooting.md                 # Common issues & solutions
```

### 4.2 Key Sections

**architecture.md:**
- Provider registry pattern
- How factory discovers and instantiates clients
- Where capabilities live (capabilities.py)
- Extensibility points

**adding-a-provider.md:**
1. Define provider config (name, base_url, models, api_key_env)
2. Choose client class (OpenAIClient for compatible, or create custom)
3. Register in provider_registry.py
4. Update capabilities.py if needed
5. Add tests
6. Verify model_catalog.py is updated

**groq-setup-and-usage.md:**
- API key setup (GROQ_API_KEY)
- Available models & performance tiers
- Known limitations (if any)
- Free tier rate limits

---

## 5. Implementation Phases

### Phase 1: Test Writing (No code changes yet)
- Write all failing tests (test-driven approach)
- Tests define expected behavior
- Mocked API responses for offline testing

### Phase 2: Provider Registry Refactor
- Create `provider_registry.py` with ProviderConfig dataclass and PROVIDER_REGISTRY dict
- Refactor `factory.py` to use registry (replace provider conditionals)
- Move provider metadata from factory/api_key_env into registry
- **Backward compatibility:** Old API surface (`create_llm_client(...)`) still works; factory internally delegates to registry
- **Migration path:** Existing callers need zero changes; internal factory code is simplified from ~50 lines to ~10
- Verify all tests pass (both old unit tests and new registry tests)

### Phase 3: Groq Feature Verification & Implementation
- Run tests against Groq models
- Implement missing features (tool calling, structured output, async)
- Update `capabilities.py` with Groq feature matrix
- All tests pass ✅

### Phase 4: Documentation
- Write /docs/providers/*.md files
- Include code examples
- Link from README.md

### Phase 5: Polish & Verification
- Smoke test: manually test Groq with a trading agent
- Verify all tests pass
- Check code is clean & documented

---

## 6. Success Criteria

✅ All tests passing (chat, tools, structured output, async)  
✅ Groq works end-to-end with trading agents  
✅ Provider system is extensible (adding a new provider takes <30 mins)  
✅ Documentation is complete (architecture, Groq guide, extension cookbook)  
✅ Code is merged to main branch  

---

## 7. Assumptions & Constraints

- **Groq API compatibility:** Will verify before Phase 1; assume Llama 3.3+ supports tools, JSON mode; JSON schema unknown
- **Tool choice support:** Will research; if Groq doesn't support `tool_choice`, update capabilities.py accordingly
- **LangChain version:** Current version supports async/streaming for OpenAI-compatible endpoints
- **Testing:** Offline tests use mocked responses; no live API calls in CI; tests are runnable without API keys
- **Future extensibility:** Registry pattern allows adding 10+ providers without code churn
- **Backward compatibility:** Existing `create_llm_client()` API remains unchanged; old conditionals in factory are replaced, not augmented

## 7.1 Pre-Phase-1 Research (BLOCKING)

Before writing tests, answer:
1. Check Groq API documentation: Does it support `tools` array + `tool_choice` parameter?
2. Check Groq API documentation: Does it support `response_format={"type": "json_mode"}`? JSON schema?
3. Verify LangChain's ChatOpenAI can stream/async with Groq endpoint

(These answers directly inform the mocks we write and the capabilities matrix.)

---

## 8. Open Questions & Decisions

### Questions (answer before Phase 1)
1. **Tool choice parameter:** Does Groq's Llama models support `tool_choice="auto"` or only `tools=[]`?
2. **JSON schema:** Does Groq support `response_format={"type": "json_schema", "schema": ...}` or only JSON mode?

### Decisions (already made)
3. **Rate limits:** ✅ Document Groq's free-tier rate limits in `groq-setup-and-usage.md`
4. **Custom base URL:** ✅ Yes, users can override via `create_llm_client(..., base_url="...")` (registry default is override-able). Same pattern as current code.

---

## 9. Related Files

- Current implementation: `tradingagents/llm_clients/openai_client.py:154` (Groq base URL)
- Model catalog: `tradingagents/llm_clients/model_catalog.py:80-93` (Groq models)
- Factory: `tradingagents/llm_clients/factory.py:11` (Groq in _OPENAI_COMPATIBLE)
- API key env: `tradingagents/llm_clients/api_key_env.py` (should have GROQ_API_KEY)
