"""Central application limits and Gemini configuration."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# Keep upload and AI processing budgets together so they are easy to explain
# and adjust for a different Gemini model or API quota.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_DOCUMENT_CHARS = 360_000
SUMMARY_CHUNK_CHARS = 12_000
SUMMARY_CHUNK_OVERLAP = 600
MAX_SUMMARY_CHUNKS = 30
QUESTION_CONTEXT_CHARS = 18_000
QUESTION_CONTEXT_CHUNKS = 4
QUESTION_CHUNK_CHARS = 6_000
DOCUMENT_PREVIEW_CHARS = 8_000

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


def get_api_key() -> str | None:
    """Return the configured Gemini API key without exposing it to the UI."""
    value = os.getenv("GEMINI_API_KEY", "").strip()
    return value or None
