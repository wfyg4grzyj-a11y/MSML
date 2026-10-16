"""Centralized client and provider factory for alpha-lab.

Every module that needs an LLM provider should call ``get_provider()``
instead of constructing one directly.  This makes it easy to swap between
OpenAI and Anthropic (or any future provider) by changing config.

Authentication:
  - OpenAI: Set OPENAI_API_KEY env var (or pass api_key directly)
  - Anthropic: Set ANTHROPIC_API_KEY env var
"""

from __future__ import annotations

import logging
import os

from openai import OpenAI

from alpha_lab.provider import Provider

logger = logging.getLogger("alpha_lab.client")


def get_client(api_key: str | None = None) -> OpenAI:
    """Return a configured OpenAI client.

    Parameters
    ----------
    api_key : str, optional
        Explicit API key.  Falls back to ``OPENAI_API_KEY`` env var.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL")  # None -> default
    return OpenAI(api_key=key, base_url=base_url)


def get_anthropic_client():
    """Create an Anthropic client using ANTHROPIC_API_KEY."""
    import anthropic
    return anthropic.Anthropic()


def get_provider(
    provider_name: str = "openai",
    api_key: str | None = None,
    *, runtime_settings=None, ledger=None, runtime=None,
) -> Provider:
    """Return a configured Provider instance.

    Parameters
    ----------
    provider_name : str
        One of "openai" or "anthropic".
    api_key : str, optional
        Explicit API key for OpenAI.  Ignored for Anthropic.
    """
    if runtime is not None:
        if runtime_settings is not None or ledger is not None:
            raise ValueError("Pass either an existing runtime or new runtime settings")
        runtime_settings = runtime.settings
    if provider_name == "openai":
        from alpha_lab.provider_openai import OpenAIProvider
        if runtime_settings is not None:
            import httpx
            from alpha_lab.llm_runtime import LLMRuntime
            key = api_key or os.environ.get("OPENAI_API_KEY", "")
            if not key or not key.strip():
                raise ValueError("OPENAI_API_KEY must be set before configuring a bounded run")
            if runtime is None and ledger is None:
                raise ValueError("A bounded provider requires a fresh ledger or an explicitly shared runtime")
            control = runtime or LLMRuntime(runtime_settings, ledger)
            control.check_active()
            client = OpenAI(api_key=key,
                            base_url=os.environ.get("OPENAI_BASE_URL"), max_retries=0,
                            timeout=httpx.Timeout(min(runtime_settings.read_timeout_seconds,
                                                     runtime_settings.max_request_seconds),
                                                  connect=min(20,runtime_settings.max_request_seconds)))
            return OpenAIProvider(client, runtime=control)
        return OpenAIProvider(get_client(api_key))
    elif provider_name in ("anthropic", "bedrock"):
        if runtime_settings is not None:
            raise ValueError("Bounded LLM runtime requires the OpenAI Responses provider")
        from alpha_lab.provider_anthropic import AnthropicProvider
        return AnthropicProvider(
            anthropic_client=get_anthropic_client(),
            openai_client=get_client(api_key),
        )
    else:
        raise ValueError(f"Unknown provider: {provider_name!r}. Use 'openai' or 'anthropic'.")
