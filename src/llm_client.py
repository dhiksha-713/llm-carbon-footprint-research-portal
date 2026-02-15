"""Provider-agnostic LLM client.

Supports Gemini and Azure OpenAI via the LLM_PROVIDER feature flag.
All RAG/eval code calls through this abstraction -- never directly to a
specific SDK.
"""

from __future__ import annotations

from dataclasses import dataclass


# ── Response ──────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LLMResponse:
    """Unified response from any LLM provider."""

    text: str
    input_tokens: int
    output_tokens: int


# ── Base ──────────────────────────────────────────────────────────────────
class LLMClient:
    """Abstract base.  Subclasses implement ``generate``."""

    provider: str = "base"

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        raise NotImplementedError


# ── Gemini ────────────────────────────────────────────────────────────────
class GeminiClient(LLMClient):
    """Google Gemini via ``google-genai`` SDK."""

    provider = "gemini"

    def __init__(self) -> None:
        from google import genai  # lazy so Azure-only installs skip this

        self._client = genai.Client()

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        temperature: float | None = None,
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
        usage = resp.usage_metadata
        return LLMResponse(
            text=resp.text or "",
            input_tokens=getattr(usage, "prompt_token_count", 0),
            output_tokens=getattr(usage, "candidates_token_count", 0),
        )


# ── Azure OpenAI ──────────────────────────────────────────────────────────
class AzureOpenAIClient(LLMClient):
    """Azure OpenAI via ``openai`` SDK."""

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
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        from src.config import AZURE_MODEL, GENERATION_TEMPERATURE, MAX_OUTPUT_TOKENS

        deployment = model or AZURE_MODEL
        temp = temperature if temperature is not None else GENERATION_TEMPERATURE
        tokens = max_tokens or MAX_OUTPUT_TOKENS

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # o-series reasoning models use max_completion_tokens and ignore temperature
        is_reasoning = deployment.startswith("o")
        kwargs: dict = {"model": deployment, "messages": messages}
        if is_reasoning:
            kwargs["max_completion_tokens"] = tokens
        else:
            kwargs["temperature"] = temp
            kwargs["max_tokens"] = tokens

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        usage = resp.usage
        return LLMResponse(
            text=choice.message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
        )


# ── Factory ───────────────────────────────────────────────────────────────
def get_llm_client() -> LLMClient:
    """Return the correct client based on ``LLM_PROVIDER`` in config."""
    from src.config import LLM_PROVIDER

    if LLM_PROVIDER == "azure_openai":
        return AzureOpenAIClient()
    if LLM_PROVIDER == "gemini":
        return GeminiClient()
    raise ValueError(
        f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}. "
        "Set to 'gemini' or 'azure_openai' in .env"
    )
