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
        assert str(llm.openai_api_base) == "https://api.groq.com/openai/v1"

    def test_llm_uses_groq_api_key_from_env(self, monkeypatch):
        """GROQ_API_KEY env var is propagated to the langchain client."""
        monkeypatch.setenv("GROQ_API_KEY", "test_key_xyz")
        client = create_llm_client("groq", "llama-3.3-70b-versatile")
        llm = client.get_llm()
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
