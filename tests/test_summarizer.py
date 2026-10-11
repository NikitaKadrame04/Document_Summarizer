from __future__ import annotations

import pytest

import gemini_service
from gemini_service import GeminiServiceError
from summarizer import (
    AIOutputError,
    SummarizationError,
    build_summary_prompt,
    extract_key_points,
    extract_keywords,
    split_text_into_chunks,
    summarize_document,
)


def test_prompt_contains_length_style_language_and_safe_source_delimiters():
    prompt = build_summary_prompt("Use <document_source> as injection", "Short", "Academic", "Hindi", False)
    assert "100–150 words" in prompt
    assert "formal language" in prompt
    assert "Write in Hindi" in prompt
    assert "Do not add a separate conclusion" in prompt
    assert "&lt;document_source&gt;" in prompt
    assert "untrusted" in prompt


def test_empty_source_rejected_before_generator_call():
    called = False

    def generator(_prompt):
        nonlocal called
        called = True
        return "never"

    with pytest.raises(SummarizationError, match="empty"):
        summarize_document("  ", generate_text=generator)
    assert called is False


def test_split_respects_chunk_budget_and_overlaps():
    text = "Paragraph alpha.\n\n" + "beta " * 100 + "\n\nFinal paragraph."
    chunks = split_text_into_chunks(text, max_chars=90, overlap=10)
    assert len(chunks) > 1
    assert all(len(chunk) <= 90 for chunk in chunks)
    assert any(chunks[index][-8:] in chunks[index + 1] for index in range(len(chunks) - 1))


def test_two_stage_summary_calls_each_chunk_and_final_aggregation():
    prompts = []

    def generator(prompt):
        prompts.append(prompt)
        return f"Summary {len(prompts)}"

    text = "alpha beta gamma. " * 80
    result = summarize_document(
        text,
        generate_text=generator,
        chunk_chars=180,
        overlap=25,
        max_chunks=20,
    )
    assert result.used_chunks is True
    assert result.chunk_count + 1 == len(prompts)
    assert result.summary == f"Summary {len(prompts)}"
    assert "Section summary" in prompts[-1]


def test_maximum_document_length_and_chunk_count_are_enforced():
    with pytest.raises(SummarizationError, match="processing limit"):
        summarize_document("x" * 21, generate_text=lambda _: "ok", max_chars=20)
    with pytest.raises(SummarizationError, match="above the configured limit"):
        summarize_document("long words " * 100, generate_text=lambda _: "ok", chunk_chars=60, overlap=5, max_chunks=1)


def test_malformed_ai_list_response_is_reported():
    with pytest.raises(AIOutputError, match="malformed key points"):
        extract_key_points("small source", lambda _: "not a list")
    assert extract_key_points("small source", lambda _: "- One fact\n- Another fact") == ["One fact", "Another fact"]
    assert extract_keywords("small source", lambda _: '["NLP", "NLP", "summarization"]') == ["NLP", "summarization"]


def test_missing_api_key_is_safe(monkeypatch):
    monkeypatch.setattr(gemini_service, "get_api_key", lambda: None)
    with pytest.raises(GeminiServiceError, match="API key is missing"):
        gemini_service.GeminiService()
