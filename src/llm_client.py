"""Provider-agnostic LLM client abstraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int


class LLMClient:
    """Abstract base. Subclasses implement ``generate``."""

    provider: str = "base"

    def generate(
        self, prompt: str, *, system: str = "",
        model: str | None = None, temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        raise NotImplementedError


class GrokClient(LLMClient):
    """Grok-3 via CMU LLM API (OpenAI-compatible endpoint)."""

    provider = "grok"

    def __init__(self) -> None:
        from openai import OpenAI
        from src.config import GROK_ENDPOINT, GROK_API_KEY
        self._client = OpenAI(base_url=GROK_ENDPOINT, api_key=GROK_API_KEY)

    def generate(
        self, prompt: str, *, system: str = "",
        model: str | None = None, temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        from src.config import GROK_MODEL, GENERATION_TEMPERATURE

        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})

        used_model = model or GROK_MODEL
        log.info("GrokClient.generate model=%s max_tokens=%s prompt_len=%d",
                 used_model, max_tokens, len(prompt))
        kwargs: dict = {
            "model": used_model,
            "messages": msgs,
            "temperature": temperature if temperature is not None else GENERATION_TEMPERATURE,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        resp = self._client.chat.completions.create(**kwargs)
        u = resp.usage
        r = LLMResponse(
            text=resp.choices[0].message.content or "",
            input_tokens=getattr(u, "prompt_tokens", 0) if u else 0,
            output_tokens=getattr(u, "completion_tokens", 0) if u else 0,
        )
        log.info("GrokClient.generate -> %d in / %d out tokens, response_len=%d",
                 r.input_tokens, r.output_tokens, len(r.text))
        return r


class AzureOpenAIClient(LLMClient):
    """Azure OpenAI (o4-mini)."""

    provider = "azure_openai"

    def __init__(self) -> None:
        from openai import AzureOpenAI
        from src.config import AZURE_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION
        self._client = AzureOpenAI(
            api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
        )

    def generate(
        self, prompt: str, *, system: str = "",
        model: str | None = None, temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        from src.config import AZURE_MODEL, GENERATION_TEMPERATURE

        deploy = model or AZURE_MODEL
        temp = temperature if temperature is not None else GENERATION_TEMPERATURE

        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})

        kwargs: dict = {"model": deploy, "messages": msgs}
        if deploy.startswith("o"):
            if max_tokens is not None:
                kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["temperature"] = temp
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

        log.info("AzureOpenAIClient.generate model=%s max_tokens=%s prompt_len=%d",
                 deploy, max_tokens, len(prompt))
        resp = self._client.chat.completions.create(**kwargs)
        u = resp.usage
        r = LLMResponse(
            text=resp.choices[0].message.content or "",
            input_tokens=getattr(u, "prompt_tokens", 0) if u else 0,
            output_tokens=getattr(u, "completion_tokens", 0) if u else 0,
        )
        log.info("AzureOpenAIClient.generate -> %d in / %d out tokens, response_len=%d",
                 r.input_tokens, r.output_tokens, len(r.text))
        return r


def get_llm_client() -> LLMClient:
    """Factory: return the correct client based on LLM_PROVIDER."""
    from src.config import LLM_PROVIDER
    if LLM_PROVIDER == "azure_openai":
        return AzureOpenAIClient()
    if LLM_PROVIDER == "grok":
        return GrokClient()
    raise ValueError(f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}")
