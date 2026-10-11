"""In-memory TXT, Markdown, and optional PDF report generation."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from html import escape
from io import BytesIO
from pathlib import PurePath
from typing import Iterable


def sanitize_filename(filename: str, max_length: int = 70) -> str:
    """Return a safe filename stem for use in download names."""
    stem = PurePath(filename or "document").stem
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return (safe or "document")[:max_length]


def _timestamp(generated_at: str | None) -> str:
    return generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _clean_items(items: Iterable[str] | None) -> list[str]:
    return [str(item).strip() for item in (items or []) if str(item).strip()]


def generate_txt_report(
    filename: str,
    summary: str,
    key_points: Iterable[str] | None = None,
    keywords: Iterable[str] | None = None,
    generated_at: str | None = None,
) -> bytes:
    points = _clean_items(key_points)
    terms = _clean_items(keywords)
    sections = [
        "SmartDoc AI — Intelligent Document Summarizer",
        f"Source file: {filename}",
        f"Generated: {_timestamp(generated_at)}",
        "AI-generated interpretation; review important details against the original.",
        "",
        "SUMMARY",
        summary.strip(),
    ]
    if points:
        sections.extend(["", "KEY POINTS", *[f"{i}. {value}" for i, value in enumerate(points, 1)]])
    if terms:
        sections.extend(["", "KEYWORDS", ", ".join(terms)])
    return ("\n".join(sections).rstrip() + "\n").encode("utf-8")


def generate_markdown_report(
    filename: str,
    summary: str,
    key_points: Iterable[str] | None = None,
    keywords: Iterable[str] | None = None,
    generated_at: str | None = None,
) -> bytes:
    points = _clean_items(key_points)
    terms = _clean_items(keywords)
    lines = [
        "# SmartDoc AI — Intelligent Document Summarizer",
        "",
        f"**Source file:** {filename}  ",
        f"**Generated:** {_timestamp(generated_at)}",
        "",
        "> AI-generated interpretation. Review important details against the original document.",
        "",
        "## Summary",
        "",
        summary.strip(),
    ]
    if points:
        lines.extend(["", "## Key Points", "", *[f"{i}. {value}" for i, value in enumerate(points, 1)]])
    if terms:
        lines.extend(["", "## Keywords", "", ", ".join(terms)])
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def generate_pdf_report(
    filename: str,
    summary: str,
    key_points: Iterable[str] | None = None,
    keywords: Iterable[str] | None = None,
    generated_at: str | None = None,
) -> bytes:
    """Build a paginated PDF in memory using the optional ReportLab package."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError as exc:
        raise RuntimeError("PDF export requires ReportLab. Install the optional report dependency.") from exc

    output = BytesIO()
    doc = SimpleDocTemplate(
        output,
        pagesize=letter,
        rightMargin=0.72 * inch,
        leftMargin=0.72 * inch,
        topMargin=0.72 * inch,
        bottomMargin=0.72 * inch,
        title="SmartDoc AI Report",
        author="SmartDoc AI",
    )
    base = getSampleStyleSheet()
    title = ParagraphStyle("SmartDocTitle", parent=base["Title"], textColor=colors.HexColor("#4338ca"), alignment=TA_LEFT)
    body = ParagraphStyle("SmartDocBody", parent=base["BodyText"], leading=15, spaceAfter=8)
    small = ParagraphStyle("SmartDocSmall", parent=body, fontSize=9, textColor=colors.HexColor("#475569"))
    story = [
        Paragraph("SmartDoc AI — Intelligent Document Summarizer", title),
        Spacer(1, 10),
        Paragraph(f"Source file: {escape(filename)}", small),
        Paragraph(f"Generated: {escape(_timestamp(generated_at))}", small),
        Spacer(1, 12),
        Paragraph("Summary", base["Heading2"]),
    ]
    story.extend(Paragraph(escape(part).replace("\n", "<br/>"), body) for part in summary.strip().split("\n\n") if part.strip())
    points = _clean_items(key_points)
    if points:
        story.extend([Spacer(1, 8), Paragraph("Key Points", base["Heading2"])])
        story.extend(Paragraph(f"{i}. {escape(point)}", body) for i, point in enumerate(points, 1))
    terms = _clean_items(keywords)
    if terms:
        story.extend([Spacer(1, 8), Paragraph("Keywords", base["Heading2"]), Paragraph(escape(", ".join(terms)), body)])

    def footer(canvas: object, page: object) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(0.72 * inch, 0.42 * inch, "AI-generated report — verify against the source document")
        canvas.drawRightString(7.78 * inch, 0.42 * inch, f"Page {page.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output.getvalue()
