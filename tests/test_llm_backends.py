"""Backends openai/groq: configuracao do cliente (sem chamar a API)."""
import pytest

pytest.importorskip("langchain_openai")

from assistant import llm  # noqa: E402
from core.config import settings  # noqa: E402


def test_groq_client_uses_groq_endpoint_and_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_teste")
    chat = llm._build_groq()
    assert chat.model_name == settings.groq_model
    assert str(chat.openai_api_base).startswith("https://api.groq.com")
    assert chat.max_tokens == settings.groq_max_tokens
    assert chat.openai_api_key.get_secret_value() == "gsk_teste"


def test_openai_client_uses_openai_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-teste")
    chat = llm._build_openai()
    assert chat.model_name == settings.openai_model
    assert chat.max_tokens == settings.openai_max_tokens
    assert chat.openai_api_key.get_secret_value() == "sk-teste"


@pytest.mark.parametrize("builder,env", [(llm._build_groq, "GROQ_API_KEY"), (llm._build_openai, "OPENAI_API_KEY")])
def test_missing_key_gives_clear_error(monkeypatch, builder, env):
    monkeypatch.delenv(env, raising=False)
    with pytest.raises(RuntimeError, match=env):
        builder()


def test_model_description_names_provider():
    assert llm.model_description("groq") == f"groq:{settings.groq_model}"
    assert llm.model_description("openai") == f"openai:{settings.openai_model}"
