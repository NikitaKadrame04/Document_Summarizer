from __future__ import annotations

import pytest

from question_answering import QuestionError, answer_question, select_relevant_context


def test_selects_question_relevant_passage():
    text = ("The coastal wetlands absorb storm water and protect nearby towns. " * 8) + "\n\n" + ("The telescope observed distant galaxies. " * 8)
    selected = select_relevant_context(text, "How do wetlands protect towns?", chunk_chars=100, max_context_chars=300)
    assert "wetlands" in selected.casefold()
    assert "telescope" not in selected.casefold()
    assert len(selected) <= 300


def test_no_overlap_has_no_relevant_context():
    assert select_relevant_context("The greenhouse stores plant samples.", "What was the tax rate?") == ""


def test_empty_question_rejected():
    with pytest.raises(QuestionError, match="Enter a question"):
        answer_question("Source text", "  ", generate_text=lambda _: "answer")


def test_question_over_long_document_uses_selected_context_only():
    text = "".join(f"Section {i}: marine algae convert sunlight to energy.\n" for i in range(60))
    prompts = []

    def generator(prompt):
        prompts.append(prompt)
        return "The document says marine algae convert sunlight to energy."

    result = answer_question(text, "How do algae use sunlight?", generate_text=generator, max_context_chars=250)
    assert "convert sunlight" in result
    assert len(prompts) == 1
    assert "marine algae" in prompts[0]
    assert "Section 59" not in prompts[0]


def test_no_relevant_passage_returns_clear_answer_without_generation():
    called = False

    def generator(_prompt):
        nonlocal called
        called = True
        return "unsupported"

    answer = answer_question(
        "A document about pottery and clay.",
        "What is the interest rate?",
        generate_text=generator,
        max_context_chars=10,
    )
    assert "couldn't find" in answer
    assert called is False
