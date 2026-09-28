"""
LLM Client

What it is: a provider-agnostic interface for "send this prompt, get text
back" (LLMClient), a real implementation backed by Google's Gemini API
(GeminiLLMClient), and a fake for tests (FakeLLMClient).

Why required: TestbenchGenerator and DebugAnalyzer need to call an LLM, but
should never import a provider SDK directly. Keeping that behind LLMClient
means testing with a fake client that returns canned responses -- or adding
another provider later -- never requires touching application logic.

Input: a fully-assembled prompt string (see prompts.py) plus optional
generation parameters (system instructions, max_tokens).
Output: the model's raw text response (str). Parsing that text into a
structured result is verification/testbench_generator.py and
debug_analyzer.py's job, not this module's.

What would happen if it were removed: every caller would construct its own
provider client directly, scattering API calls and error handling
throughout the application.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Optional


class LLMError(Exception):
    """Raised for any LLM-call failure (auth, network, rate limit, malformed
    response) -- one exception type so callers only need to catch one thing."""


# Errors that are usually temporary on the provider's side (overload / brief
# outage). Rate-limit and quota errors (429) are deliberately NOT retried:
# they won't clear in a few seconds and retrying just burns free-tier quota.
_TRANSIENT_MARKERS = ("503", "502", "504", "unavailable", "high demand", "overloaded")


def _is_transient(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_MARKERS)


def _call_with_retry(fn, retries: int = 3, base_delay: float = 2.0, sleep=time.sleep):
    """Call fn(); on a transient provider error wait 2s, 4s, 8s (exponential
    backoff) and retry, up to `retries` extra attempts. Any other error, or
    the last failed attempt, is re-raised unchanged."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if attempt < retries and _is_transient(exc):
                sleep(base_delay * (2 ** attempt))
                continue
            raise


class LLMClient(ABC):
    """Interface every LLM backend must implement: prompt in, text out."""

    @abstractmethod
    def complete(self, prompt: str, system: Optional[str] = None, max_tokens: int = 2048) -> str:
        """
        Send prompt (and optional system instructions) to the LLM and return
        its text response. Raises LLMError on any failure -- missing/invalid
        API key, network failure, rate limiting, or an empty response.
        """


class GeminiLLMClient(LLMClient):
    """
    Real LLMClient backend for Google's Gemini API (free tier via Google AI
    Studio), using the `google-genai` SDK. The SDK is imported lazily inside
    complete(), so constructing this object needs neither the package nor the
    network -- only an actual call does. Every failure (auth, rate limit /
    HTTP 429, network, empty response) surfaces as LLMError, and temporary
    overload errors (503) are retried with backoff first.
    """

    def __init__(self, api_key: str, model: str = "gemini-3.8-flash", temperature: float = 0.2):
        if not api_key or not api_key.strip():
            raise LLMError(
                "No LLM API key configured. Set LLM_API_KEY in your .env file "
                "(see .env.example) before using GeminiLLMClient."
            )
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._sleep = time.sleep  # replaceable in tests

    def complete(self, prompt: str, system: Optional[str] = None, max_tokens: int = 2048) -> str:
        if not prompt or not prompt.strip():
            raise LLMError("Cannot send an empty prompt to the LLM")
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise LLMError(
                "The 'google-genai' package is not installed. Install it with "
                "`pip install google-genai` to use GeminiLLMClient."
            ) from exc

        def _generate() -> str:
            client = genai.Client(api_key=self._api_key)
            response = client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=self._temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return (response.text or "").strip()

        try:
            text = _call_with_retry(_generate, sleep=self._sleep)
        except Exception as exc:  # noqa: BLE001 - wrap every provider failure, never leak the key
            raise LLMError(f"LLM request failed: {exc}") from exc

        if not text:
            raise LLMError("LLM returned an empty or non-text response")
        return text


class FakeLLMClient(LLMClient):
    """
    Test/demo LLMClient backend. Returns a pre-configured, fixed response (or
    one produced by a caller-supplied function of the prompt) instead of
    calling any real API. Used by unit/integration tests so they need no
    network access or API key, and never presented as real model output.
    """

    def __init__(self, fixed_response: Optional[str] = None, response_fn=None):
        if fixed_response is None and response_fn is None:
            raise ValueError("FakeLLMClient requires either fixed_response or response_fn")
        self._fixed_response = fixed_response
        self._response_fn = response_fn

    def complete(self, prompt: str, system: Optional[str] = None, max_tokens: int = 2048) -> str:
        if not prompt or not prompt.strip():
            raise LLMError("Cannot send an empty prompt to the LLM")
        if self._response_fn is not None:
            return self._response_fn(prompt)
        return self._fixed_response  # type: ignore[return-value]
