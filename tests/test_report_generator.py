from __future__ import annotations

import importlib.util

import pytest

from report_generator import (
    generate_markdown_report,
    generate_pdf_report,
    generate_txt_report,
    sanitize_filename,
)


def test_txt_export_contents_and_unicode():
    report = generate_txt_report("résumé.pdf", "A useful summary.", ["First fact"], ["NLP"], "2026-10-11")
    decoded = report.decode("utf-8")
    assert "résumé.pdf" in decoded
    assert "SUMMARY\nA useful summary." in decoded
    assert "1. First fact" in decoded
    assert "KEYWORDS\nNLP" in decoded
    assert "API key" not in decoded


def test_markdown_report_structure():
    report = generate_markdown_report("notes.txt", "Main result", ["Point"], ["topic"])
    decoded = report.decode("utf-8")
    assert "# SmartDoc AI" in decoded
    assert "## Summary" in decoded
    assert "## Key Points" in decoded
    assert "## Keywords" in decoded


def test_filename_sanitization():
    assert sanitize_filename("../../my private report.pdf") == "my_private_report"
    assert sanitize_filename("../..") == "document"


def test_pdf_export_when_reportlab_is_installed():
    if importlib.util.find_spec("reportlab") is None:
        pytest.skip("ReportLab is an optional PDF dependency")
    long_summary = "Café & analysis: " + ("A long section with readable details. " * 1000)
    report = generate_pdf_report("study.pdf", long_summary, ["Main result"], ["NLP"])
    assert report.startswith(b"%PDF-")
    assert len(report) > 1000
