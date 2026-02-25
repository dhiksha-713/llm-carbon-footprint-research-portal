"""Provider-agnostic LLM client abstraction."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)


class LLMServiceError(RuntimeError):
    """Raised when an LLM call fails after retries or with non-retryable errors."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int


class LLMClient:
    """Abstract base. Subclasses implement ``generate``."""

    provider: str = "base"
    _last_call_ts: float = 0.0

    def _throttle(self) -> None:
        """Enforce a minimum interval between consecutive LLM calls."""
        from src.config import LLM_MIN_CALL_INTERVAL_S
        if LLM_MIN_CALL_INTERVAL_S <= 0:
            return
        elapsed = time.time() - self._last_call_ts
        if elapsed < LLM_MIN_CALL_INTERVAL_S:
            sleep_s = LLM_MIN_CALL_INTERVAL_S - elapsed
            log.debug("%s throttle sleep %.2fs", self.__class__.__name__, sleep_s)
            time.sleep(sleep_s)

    def _sleep_backoff(self, attempt: int) -> None:
        """Exponential backoff with jitter for transient failures."""
        from src.config import LLM_BACKOFF_BASE_S, LLM_BACKOFF_MAX_S
        base = max(0.1, LLM_BACKOFF_BASE_S)
        max_wait = max(base, LLM_BACKOFF_MAX_S)
        wait = min(max_wait, base * (2 ** attempt))
        jitter = random.uniform(0.0, base)
        time.sleep(wait + jitter)

    def _is_retryable_error(self, exc: Exception) -> tuple[bool, str]:
        """Detect transient OpenAI/Azure failures worth retrying."""
        name = exc.__class__.__name__
        status_code = getattr(exc, "status_code", None)
        body = str(exc).lower()
        retryable_status = {408, 409, 429, 500, 502, 503, 504}
        retryable_name_markers = (
            "RateLimitError",
            "APITimeoutError",
            "APIConnectionError",
            "InternalServerError",
        )
        if status_code in retryable_status:
            return True, f"status={status_code}"
        if any(marker in name for marker in retryable_name_markers):
            return True, name
        if "rate limit" in body or "too many requests" in body or "timeout" in body:
            return True, name
        return False, name

    def _chat_completion_with_retry(self, client, kwargs: dict):
        """Call chat.completions.create with retries/backoff for transient errors."""
        from src.config import LLM_MAX_RETRIES
        attempts = max(1, LLM_MAX_RETRIES + 1)
        last_exc: Exception | None = None

        for attempt in range(attempts):
            try:
                self._throttle()
                resp = client.chat.completions.create(**kwargs)
                self._last_call_ts = time.time()
                return resp
            except Exception as exc:  # OpenAI SDK uses provider-specific exception classes
                last_exc = exc
                retryable, reason = self._is_retryable_error(exc)
                is_last = attempt == attempts - 1
                if not retryable or is_last:
                    log.error(
                        "%s call failed (attempt %d/%d, retryable=%s, reason=%s): %s",
                        self.__class__.__name__, attempt + 1, attempts, retryable, reason, exc,
                    )
                    raise LLMServiceError(
                        f"{self.__class__.__name__} request failed after {attempt + 1}/{attempts} attempts: {exc}"
                    ) from exc
                log.warning(
                    "%s transient LLM error (attempt %d/%d, reason=%s). Retrying...",
                    self.__class__.__name__, attempt + 1, attempts, reason,
                )
                self._sleep_backoff(attempt)

        if last_exc is not None:
            raise LLMServiceError(
                f"{self.__class__.__name__} request failed after {attempts} attempts: {last_exc}"
            ) from last_exc

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
        resp = self._chat_completion_with_retry(self._client, kwargs)
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
        resp = self._chat_completion_with_retry(self._client, kwargs)
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
