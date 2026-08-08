from __future__ import annotations

import pytest

from sharek_agents.common import llm


class FakeChatModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture(autouse=True)
def clear_llm_cache():
    llm.clear_cache()
    yield
    llm.clear_cache()


def configure_alibaba(monkeypatch) -> None:
    monkeypatch.setattr(llm.settings, "ai_provider", "alibaba")
    monkeypatch.setattr(llm.settings, "default_model", "")
    monkeypatch.setattr(llm.settings, "alibaba_model", "qwen3.7-plus")
    monkeypatch.setattr(llm.settings, "alibaba_enable_thinking", False)
    monkeypatch.setattr(llm.settings, "alibaba_api_key", "test-secret")
    monkeypatch.setattr(
        llm.settings,
        "alibaba_base_url",
        "https://workspace.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1",
    )


def test_alibaba_uses_openai_compatible_client(monkeypatch) -> None:
    configure_alibaba(monkeypatch)
    monkeypatch.setattr(llm, "ChatOpenAI", FakeChatModel)

    client = llm.get_llm()

    assert client.kwargs == {
        "model": "qwen3.7-plus",
        "api_key": "test-secret",
        "base_url": (
            "https://workspace.ap-southeast-1.maas.aliyuncs.com/"
            "compatible-mode/v1"
        ),
        "timeout": llm.settings.ai_skill_profile_timeout_seconds,
        "temperature": 0.0,
        "extra_body": {"enable_thinking": False},
    }


def test_alibaba_model_wins_over_legacy_default_model(monkeypatch) -> None:
    configure_alibaba(monkeypatch)
    monkeypatch.setattr(llm.settings, "default_model", "legacy/provider-model")
    monkeypatch.setattr(llm, "ChatOpenAI", FakeChatModel)

    client = llm.get_llm()

    assert client.kwargs["model"] == "qwen3.7-plus"


def test_alibaba_client_is_cached_by_provider_model_and_base_url(
    monkeypatch,
) -> None:
    configure_alibaba(monkeypatch)
    created: list[FakeChatModel] = []

    def create_client(**kwargs):
        client = FakeChatModel(**kwargs)
        created.append(client)
        return client

    monkeypatch.setattr(llm, "ChatOpenAI", create_client)

    first = llm.get_llm()
    second = llm.get_llm()

    assert first is second
    assert len(created) == 1


def test_alibaba_rejects_missing_credentials(monkeypatch) -> None:
    configure_alibaba(monkeypatch)
    monkeypatch.setattr(llm.settings, "alibaba_api_key", "")

    with pytest.raises(llm.LLMConfigurationError, match="ALIBABA_API_KEY"):
        llm.get_llm()


def test_alibaba_rejects_non_compatible_base_url(monkeypatch) -> None:
    configure_alibaba(monkeypatch)
    monkeypatch.setattr(
        llm.settings,
        "alibaba_base_url",
        "https://workspace.ap-southeast-1.maas.aliyuncs.com/api/v1",
    )

    with pytest.raises(llm.LLMConfigurationError, match="OpenAI-compatible"):
        llm.get_llm()


def test_openrouter_provider_is_preserved(monkeypatch) -> None:
    monkeypatch.setattr(llm.settings, "ai_provider", "openrouter")
    monkeypatch.setattr(llm.settings, "default_model", "")
    monkeypatch.setattr(llm.settings, "openrouter_model", "test/model")
    monkeypatch.setattr(llm.settings, "openrouter_api_key", "test-secret")
    monkeypatch.setattr(llm, "ChatOpenRouter", FakeChatModel)

    client = llm.get_llm()

    assert client.kwargs["model"] == "test/model"
    assert client.kwargs["api_key"] == "test-secret"


def test_groq_provider_is_preserved(monkeypatch) -> None:
    monkeypatch.setattr(llm.settings, "ai_provider", "groq")
    monkeypatch.setattr(llm.settings, "default_model", "")
    monkeypatch.setattr(llm.settings, "groq_model", "openai/gpt-oss-120b")
    monkeypatch.setattr(llm.settings, "groq_api_key", "test-secret")
    monkeypatch.setattr(llm, "ChatOpenAI", FakeChatModel)

    client = llm.get_llm()

    assert client.kwargs["model"] == "openai/gpt-oss-120b"
    assert client.kwargs["api_key"] == "test-secret"
    assert client.kwargs["base_url"] == "https://api.groq.com/openai/v1"


def test_unknown_provider_fails_before_network_access(monkeypatch) -> None:
    monkeypatch.setattr(llm.settings, "ai_provider", "unknown")
    monkeypatch.setattr(llm.settings, "default_model", "some-model")

    with pytest.raises(llm.LLMConfigurationError, match="Unsupported AI_PROVIDER"):
        llm.get_llm()
