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

### Step 4: Add models to the catalog

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

### Step 5: Add to the CLI provider list

`cli/utils.py` — inside `select_llm_provider()`, add a tuple in the providers
list with display name, key, and default base URL.

### Step 6: Capabilities (optional)

If models have unusual quirks (e.g., reject `tool_choice`), add entries to
`capabilities._BY_ID`. Otherwise they inherit the permissive `_DEFAULT`.

### Step 7: Add tests

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

### Step 8: Update docs

- Add a row to `docs/providers/README.md` provider table.
- Add a row to `docs/providers/capability-matrix.md`.
- Optionally create `docs/providers/newprovider-setup-and-usage.md`.

### Step 9: Verify

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

### Steps 3-9: Same as Path A.
