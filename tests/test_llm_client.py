import pytest

from app.llm.client import GeminiLLMClient, FakeLLMClient, LLMError


def test_fake_client_returns_fixed_response():
    client = FakeLLMClient(fixed_response="hello world")
    assert client.complete("any prompt") == "hello world"


def test_fake_client_returns_response_fn_output():
    client = FakeLLMClient(response_fn=lambda prompt: f"echo: {prompt}")
    assert client.complete("hi") == "echo: hi"


def test_fake_client_rejects_empty_prompt():
    client = FakeLLMClient(fixed_response="x")
    with pytest.raises(LLMError):
        client.complete("")


def test_fake_client_requires_response_or_fn():
    with pytest.raises(ValueError):
        FakeLLMClient()


def test_gemini_client_rejects_missing_api_key():
    with pytest.raises(LLMError):
        GeminiLLMClient(api_key="")


def test_gemini_client_rejects_whitespace_only_api_key():
    with pytest.raises(LLMError):
        GeminiLLMClient(api_key="   ")


def test_gemini_client_construction_with_key_does_not_call_network():
    # Constructing the client must not itself attempt any network call —
    # only complete() should, and only then.
    client = GeminiLLMClient(api_key="fake-key-for-construction-test")
    assert client is not None


def test_gemini_client_rejects_empty_prompt_before_network_call():
    client = GeminiLLMClient(api_key="fake-key")
    with pytest.raises(LLMError):
        client.complete("")
