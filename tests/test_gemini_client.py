"""Tests for GeminiLLMClient (SDK replaced with fakes: no network, no package)."""

import sys
import types

import pytest

from app.llm.client import GeminiLLMClient, LLMError


def _install_fake_gemini(monkeypatch, text="hello", raises=None, fail_first=0):
    calls = {"n": 0}

    class _Models:
        def generate_content(self, model, contents, config):
            calls["n"] += 1
            calls.update(model=model, contents=contents, config=config)
            if raises and calls["n"] > fail_first:
                raise raises
            if fail_first and calls["n"] <= fail_first:
                raise RuntimeError("503 UNAVAILABLE. high demand")
            return types.SimpleNamespace(text=text)

    class _Client:
        def __init__(self, api_key):
            calls["api_key"] = api_key
            self.models = _Models()

    class _Config:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    genai = types.ModuleType("google.genai")
    genai.Client = _Client
    genai_types = types.ModuleType("google.genai.types")
    genai_types.GenerateContentConfig = _Config
    genai.types = genai_types
    google = types.ModuleType("google")
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", genai_types)
    return calls


def _client(**kw):
    client = GeminiLLMClient(api_key="k", **kw)
    client._sleep = lambda seconds: None
    return client


def test_requires_api_key():
    with pytest.raises(LLMError):
        GeminiLLMClient(api_key="  ")


def test_construction_never_touches_network_or_sdk():
    assert GeminiLLMClient(api_key="k") is not None


def test_complete_returns_text_and_passes_settings(monkeypatch):
    calls = _install_fake_gemini(monkeypatch, text="  answer  ")
    client = GeminiLLMClient(api_key="k", model="some-model", temperature=0.1)
    assert client.complete("prompt", system="sys", max_tokens=123) == "answer"
    assert calls["model"] == "some-model"
    assert calls["contents"] == "prompt"
    assert calls["config"].system_instruction == "sys"
    assert calls["config"].max_output_tokens == 123
    assert calls["config"].temperature == 0.1


def test_rejects_empty_prompt():
    with pytest.raises(LLMError):
        GeminiLLMClient(api_key="k").complete("   ")


def test_empty_response_raises(monkeypatch):
    _install_fake_gemini(monkeypatch, text="")
    with pytest.raises(LLMError):
        _client().complete("prompt")


def test_sdk_errors_become_llmerror(monkeypatch):
    _install_fake_gemini(monkeypatch, raises=RuntimeError("401 bad key"))
    with pytest.raises(LLMError, match="LLM request failed"):
        _client().complete("prompt")


def test_error_message_never_contains_api_key(monkeypatch):
    _install_fake_gemini(monkeypatch, raises=RuntimeError("boom"))
    client = GeminiLLMClient(api_key="SECRET-KEY-123")
    client._sleep = lambda s: None
    with pytest.raises(LLMError) as info:
        client.complete("prompt")
    assert "SECRET-KEY-123" not in str(info.value)


def test_missing_sdk_gives_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "google", None)
    with pytest.raises(LLMError, match="google-genai"):
        _client().complete("prompt")


def test_retries_transient_503_then_succeeds(monkeypatch):
    calls = _install_fake_gemini(monkeypatch, text="finally", fail_first=2)
    client = GeminiLLMClient(api_key="k")
    waits = []
    client._sleep = waits.append
    assert client.complete("prompt") == "finally"
    assert calls["n"] == 3
    assert waits == [2.0, 4.0]


def test_gives_up_after_retries_on_persistent_503(monkeypatch):
    _install_fake_gemini(monkeypatch, raises=RuntimeError("503 UNAVAILABLE"))
    client = GeminiLLMClient(api_key="k")
    waits = []
    client._sleep = waits.append
    with pytest.raises(LLMError, match="503"):
        client.complete("prompt")
    assert waits == [2.0, 4.0, 8.0]


def test_does_not_retry_rate_limit_429(monkeypatch):
    _install_fake_gemini(monkeypatch, raises=RuntimeError("429 RESOURCE_EXHAUSTED quota"))
    client = GeminiLLMClient(api_key="k")
    waits = []
    client._sleep = waits.append
    with pytest.raises(LLMError):
        client.complete("prompt")
    assert waits == []


def test_build_app_context_uses_gemini_client(tmp_path):
    from app.config.settings import Settings
    from app.context import build_app_context

    settings = Settings(llm_api_key="k", vector_db_path=tmp_path / "vs", log_path=tmp_path / "logs")
    ctx = build_app_context(settings)
    assert isinstance(ctx.llm_client, GeminiLLMClient)
    assert ctx.using_fake_llm is False


def test_build_app_context_without_key_uses_erroring_fake(tmp_path):
    from app.config.settings import Settings
    from app.context import build_app_context

    settings = Settings(llm_api_key="", vector_db_path=tmp_path / "vs", log_path=tmp_path / "logs")
    ctx = build_app_context(settings)
    assert ctx.using_fake_llm is True
    with pytest.raises(LLMError, match="No LLM API key"):
        ctx.llm_client.complete("prompt")
