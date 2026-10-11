"""Small wrapper around the official Google Gen AI Python SDK."""

from __future__ import annotations

import logging
from typing import Any

from config import DEFAULT_GEMINI_MODEL, get_api_key

logger = logging.getLogger(__name__)


class GeminiServiceError(RuntimeError):
    """A safe, user-facing Gemini configuration or request failure."""


class GeminiService:
    """Reusable client for one Streamlit action; the API key stays server-side."""

    def __init__(self, model_name: str | None = None, *, api_key: str | None = None) -> None:
        key = api_key or get_api_key()
        if not key:
            raise GeminiServiceError(
                "Gemini API key is missing. Add GEMINI_API_KEY to a local .env file and restart the app."
            )
        self.model_name = model_name or DEFAULT_GEMINI_MODEL
        try:
            from google import genai

            self._client = genai.Client(api_key=key)
        except ImportError as exc:
            raise GeminiServiceError("The Google Gen AI SDK is not installed. Install project dependencies first.") from exc
        except Exception as exc:
            logger.warning("Gemini client configuration failed (%s).", type(exc).__name__)
            raise GeminiServiceError("Gemini could not be configured. Check the API key and model setting.") from exc

    def generate_text(self, prompt: str) -> str:
        """Generate a text response and translate SDK failures to safe messages."""
        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            text = getattr(response, "text", None)
            if not isinstance(text, str) or not text.strip():
                raise GeminiServiceError("Gemini returned an empty response. Please try again.")
            return text.strip()
        except GeminiServiceError:
            raise
        except Exception as exc:  # SDK exception classes vary between releases.
            code = _provider_code(exc)
            logger.warning("Gemini request failed (code=%s, type=%s).", code, type(exc).__name__)
            raise GeminiServiceError(_safe_provider_message(code, exc)) from exc


def _provider_code(exc: Exception) -> int | None:
    value: Any = getattr(exc, "code", None)
    if value is None:
        response = getattr(exc, "response", None)
        value = getattr(response, "status_code", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_provider_message(code: int | None, exc: Exception) -> str:
    description = f"{type(exc).__name__} {exc}".lower()
    if code in (401, 403) or "api key" in description or "unauthorized" in description:
        return "Gemini rejected the API configuration. Check that the API key is valid and enabled."
    if code == 429 or "resource_exhausted" in description or "rate limit" in description:
        return "Gemini is rate-limiting this request or the available quota is exhausted. Wait and try again."
    if code == 404 or "not found" in description:
        return "The selected Gemini model is unavailable. Choose a currently supported model in config.py."
    if code is not None and code >= 500 or isinstance(exc, (TimeoutError, ConnectionError)):
        return "Gemini is temporarily unavailable or the network request timed out. Try again shortly."
    if code in (400, 422):
        return "Gemini could not accept this request. Check the document size and model configuration."
    return "The Gemini request failed. Check your connection and API configuration, then try again."
