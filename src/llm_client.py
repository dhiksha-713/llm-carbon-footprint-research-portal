"""Provider-agnostic LLM client abstraction."""

from __future__ import annotations

from dataclasses import dataclass


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


class GeminiClient(LLMClient):
    provider = "gemini"

    def __init__(self) -> None:
        from google import genai
        self._client = genai.Client()

    def generate(
        self, prompt: str, *, system: str = "",
        model: str | None = None, temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        from google.genai import types
        from src.config import GEMINI_MODEL, GENERATION_TEMPERATURE, MAX_OUTPUT_TOKENS

        resp = self._client.models.generate_content(
            model=model or GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system or None,
                temperature=temperature if temperature is not None else GENERATION_TEMPERATURE,
                max_output_tokens=max_tokens or MAX_OUTPUT_TOKENS,
            ),
        )
        u = resp.usage_metadata
        return LLMResponse(
            text=resp.text or "",
            input_tokens=getattr(u, "prompt_token_count", 0),
            output_tokens=getattr(u, "candidates_token_count", 0),
        )


class AzureOpenAIClient(LLMClient):
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
        from src.config import AZURE_MODEL, GENERATION_TEMPERATURE, MAX_OUTPUT_TOKENS

        deploy = model or AZURE_MODEL
        temp = temperature if temperature is not None else GENERATION_TEMPERATURE
        tokens = max_tokens or MAX_OUTPUT_TOKENS

        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})

        kwargs: dict = {"model": deploy, "messages": msgs}
        if deploy.startswith("o"):
            kwargs["max_completion_tokens"] = tokens
        else:
            kwargs["temperature"] = temp
            kwargs["max_tokens"] = tokens

        resp = self._client.chat.completions.create(**kwargs)
        u = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            input_tokens=getattr(u, "prompt_tokens", 0) if u else 0,
            output_tokens=getattr(u, "completion_tokens", 0) if u else 0,
        )


def get_llm_client() -> LLMClient:
    """Factory: return the correct client based on LLM_PROVIDER."""
    from src.config import LLM_PROVIDER
    if LLM_PROVIDER == "azure_openai":
        return AzureOpenAIClient()
    if LLM_PROVIDER == "gemini":
        return GeminiClient()
    raise ValueError(f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}")
