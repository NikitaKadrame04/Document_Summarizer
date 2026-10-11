"""Streamlit user interface for SmartDoc AI."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime

import streamlit as st

from config import (
    DOCUMENT_PREVIEW_CHARS,
    MAX_DOCUMENT_CHARS,
    MAX_UPLOAD_BYTES,
    SUMMARY_CHUNK_CHARS,
    get_api_key,
)
from document_processor import DocumentExtractionResult, DocumentProcessingError, extract_document_text
from gemini_service import GeminiService, GeminiServiceError
from question_answering import QuestionError, answer_question
from report_generator import (
    generate_markdown_report,
    generate_pdf_report,
    generate_txt_report,
    sanitize_filename,
)
from summarizer import (
    AIOutputError,
    SummarizationError,
    extract_key_points,
    extract_keywords,
    summarize_document,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {
        --smartdoc-indigo: #4f46e5;
        --smartdoc-indigo-dark: #3730a3;
        --smartdoc-ink: #172033;
        --smartdoc-field: #ffffff;
        --smartdoc-field-border: #cbd5e1;
      }
      .stApp, [data-testid="stAppViewContainer"] { background: #fbfbfe !important; color: var(--smartdoc-ink) !important; }
      section[data-testid="stSidebar"],
      section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        background: #f1f0ff !important; border-right: 1px solid #e1defa;
      }
      section[data-testid="stSidebar"] h1,
      section[data-testid="stSidebar"] h2,
      section[data-testid="stSidebar"] h3 { color: #312e81 !important; }
      section[data-testid="stSidebar"] p,
      section[data-testid="stSidebar"] label,
      section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
      section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #37365f !important; }
      .block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 2rem; }
      h1, h2, h3 { color: #1e1b4b; letter-spacing: -0.025em; }
      p, label, [data-testid="stMarkdownContainer"] { color: #172033; }
      div[data-testid="stMetric"] { background: #ffffff !important; color: #172033 !important;
        border: 1px solid #e7e7f0; border-radius: 12px; padding: 0.85rem 1rem; }
      [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *,
      [data-testid="stMetricValue"], [data-testid="stMetricValue"] *,
      [data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * { color: #172033 !important; opacity: 1 !important; }
      div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--smartdoc-indigo); }
      div[data-testid="stFileUploader"] section { border: 1px dashed #a5b4fc; border-radius: 14px; background: white; }
      [data-testid="stFileUploaderDropzone"] { background: #ffffff !important; }
      [data-testid="stFileUploaderDropzone"] * { color: #334155 !important; }
      div[data-testid="stButton"] > button,
      div[data-testid="stDownloadButton"] > button,
      div[data-testid="stFileUploader"] button {
        background: #4f46e5 !important; color: #ffffff !important;
        border: 1px solid #4338ca !important; opacity: 1 !important;
      }
      div[data-testid="stButton"] > button *,
      div[data-testid="stDownloadButton"] > button *,
      div[data-testid="stFileUploader"] button * { color: #ffffff !important; }
      div[data-testid="stButton"] > button:hover,
      div[data-testid="stDownloadButton"] > button:hover,
      div[data-testid="stFileUploader"] button:hover { background: #3730a3 !important; }
      div[data-testid="stButton"] > button:disabled { background: #94a3b8 !important; border-color: #94a3b8 !important; }
      /* Keep every form control light and readable, even when the OS/browser prefers dark mode. */
      div[data-baseweb="select"] > div {
        background: var(--smartdoc-field) !important;
        color: var(--smartdoc-ink) !important;
        border-color: var(--smartdoc-field-border) !important;
      }
      div[data-baseweb="select"] input,
      div[data-baseweb="select"] svg { color: var(--smartdoc-ink) !important; fill: var(--smartdoc-ink) !important; }
      div[data-testid="stTextInput"] [data-baseweb="input"] > div,
      div[data-testid="stTextArea"] [data-baseweb="textarea"],
      div[data-testid="stNumberInput"] [data-baseweb="input"] > div,
      div[data-testid="stDateInput"] [data-baseweb="input"] > div,
      div[data-testid="stTimeInput"] [data-baseweb="input"] > div,
      div[data-testid="stChatInput"] [data-baseweb="textarea"] {
        background: var(--smartdoc-field) !important;
        border-color: var(--smartdoc-field-border) !important;
      }
      div[data-testid="stTextInput"] input,
      div[data-testid="stTextArea"] textarea,
      div[data-testid="stNumberInput"] input,
      div[data-testid="stDateInput"] input,
      div[data-testid="stTimeInput"] input,
      div[data-testid="stChatInput"] textarea {
        background: var(--smartdoc-field) !important;
        color: var(--smartdoc-ink) !important;
        border-color: var(--smartdoc-field-border) !important;
      }
      div[data-baseweb="select"] > div:focus-within,
      [data-testid="stTextInput"] [data-baseweb="input"] > div:focus-within,
      [data-testid="stTextArea"] [data-baseweb="textarea"]:focus-within,
      [data-testid="stNumberInput"] [data-baseweb="input"] > div:focus-within,
      [data-testid="stDateInput"] [data-baseweb="input"] > div:focus-within,
      [data-testid="stTimeInput"] [data-baseweb="input"] > div:focus-within,
      [data-testid="stChatInput"] [data-baseweb="textarea"]:focus-within,
      div[data-testid="stTextInput"] input:focus,
      div[data-testid="stTextArea"] textarea:focus,
      div[data-testid="stNumberInput"] input:focus,
      div[data-testid="stDateInput"] input:focus,
      div[data-testid="stTimeInput"] input:focus,
      div[data-testid="stChatInput"] textarea:focus {
        border-color: var(--smartdoc-indigo) !important;
        box-shadow: 0 0 0 1px var(--smartdoc-indigo) !important;
      }
      input::placeholder, textarea::placeholder { color: #64748b !important; opacity: 1 !important; }
      [data-baseweb="popover"] [role="listbox"],
      [data-baseweb="popover"] [role="option"] {
        background: #ffffff !important; color: var(--smartdoc-ink) !important;
      }
      [data-baseweb="popover"] [role="option"]:hover,
      [data-baseweb="popover"] [role="option"][aria-selected="true"] {
        background: #eef2ff !important; color: #312e81 !important;
      }
      div[data-testid="stCheckbox"] input { accent-color: var(--smartdoc-indigo) !important; }
      div[data-testid="stAlert"] { color: #1e3a8a !important; }
      .smartdoc-subtitle { color: #64748b; font-size: 1.07rem; margin-top: -0.55rem; }
      .smartdoc-card { background: white; border: 1px solid #e7e7f0; padding: 1rem 1.15rem;
        border-radius: 12px; margin: 0.35rem 0 1rem 0; }
      .smartdoc-footer { color: #64748b; font-size: 0.86rem; border-top: 1px solid #e7e7f0;
        padding-top: 1rem; margin-top: 2rem; }
      @media (max-width: 700px) { .block-container { padding: 1rem 0.8rem 1.5rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def _reset_document_state() -> None:
    for key, value in {
        "current_document_id": None,
        "document_result": None,
        "document_error": None,
        "summary_result": None,
        "summary_settings": None,
        "key_points": None,
        "keywords": None,
        "chat_history": [],
        "generated_at": None,
    }.items():
        st.session_state[key] = value


def _friendly_error(exc: Exception) -> str:
    if isinstance(exc, (GeminiServiceError, DocumentProcessingError, SummarizationError, QuestionError)):
        return str(exc)
    if isinstance(exc, AIOutputError):
        return str(exc)
    logger.warning("An application action failed (%s).", type(exc).__name__)
    return "Something went wrong while processing this request. Please try again or use a smaller document."


for key, default in {
    "upload_widget_version": 0,
    "current_document_id": None,
    "document_result": None,
    "document_error": None,
    "summary_result": None,
    "summary_settings": None,
    "key_points": None,
    "keywords": None,
    "chat_history": [],
    "generated_at": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.title("📄 SmartDoc AI")
st.markdown('<p class="smartdoc-subtitle">Transform lengthy documents into clear, actionable insights.</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Workspace")
    if get_api_key():
        st.success("Gemini API configured")
    else:
        st.warning("Gemini API key missing")
        st.caption("Add GEMINI_API_KEY to a local .env file, then restart the app to enable AI features.")
    st.divider()
    st.subheader("About")
    st.caption("A workspace for extracting documents, generating summaries, and asking document-grounded questions.")
    st.caption(f"Upload limit: {MAX_UPLOAD_BYTES // (1024 * 1024)} MB · Max AI input: {MAX_DOCUMENT_CHARS:,} characters")
    st.info("When you request an AI feature, the relevant document text is sent to Google Gemini. Avoid confidential documents unless you are authorized to share them.", icon="🔒")

st.subheader("1 · Upload a document")
st.write("Choose one PDF, DOCX, or plain text file. Processing stays in memory during this session.")
uploaded_file = st.file_uploader(
    "Drop a document here or browse files",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=False,
    max_upload_size=MAX_UPLOAD_BYTES // (1024 * 1024),
    key=f"document_uploader_{st.session_state.upload_widget_version}",
    help="Maximum file size: 10 MB. Scanned image-only PDFs are not supported because OCR is not included.",
)

if uploaded_file is None and st.session_state.current_document_id is not None:
    _reset_document_state()
elif uploaded_file is not None:
    upload_bytes = uploaded_file.getvalue()
    document_id = hashlib.sha256(uploaded_file.name.encode("utf-8", errors="replace") + b"\0" + upload_bytes).hexdigest()
    if document_id != st.session_state.current_document_id:
        _reset_document_state()
        st.session_state.current_document_id = document_id
        try:
            st.session_state.document_result = extract_document_text(uploaded_file)
        except DocumentProcessingError as exc:
            st.session_state.document_error = str(exc)

if st.session_state.document_error:
    st.error(st.session_state.document_error)

document: DocumentExtractionResult | None = st.session_state.document_result
if document is not None:
    meta = document.metadata
    st.subheader("2 · Extracted document")
    st.markdown(f'<div class="smartdoc-card"><strong>{meta.filename}</strong> · {meta.file_type}{f" · {meta.encoding}" if meta.encoding else ""}</div>', unsafe_allow_html=True)
    columns = st.columns(4)
    columns[0].metric("Characters", f"{meta.character_count:,}")
    columns[1].metric("Approx. words", f"{meta.word_count:,}")
    columns[2].metric("Pages", str(meta.page_count) if meta.page_count is not None else "—")
    columns[3].metric("File type", meta.file_type)
    if meta.low_text_pages:
        pages = ", ".join(map(str, meta.low_text_pages[:12]))
        suffix = " …" if len(meta.low_text_pages) > 12 else ""
        st.info(f"Little or no selectable text was found on PDF page(s) {pages}{suffix}. Scanned pages may need OCR, which is not included.")
    with st.expander("Preview extracted text", expanded=False):
        preview = document.text[:DOCUMENT_PREVIEW_CHARS]
        st.text_area("Extracted text preview", preview, height=260, disabled=True, label_visibility="collapsed")
        if len(document.text) > DOCUMENT_PREVIEW_CHARS:
            st.caption(f"Preview limited to {DOCUMENT_PREVIEW_CHARS:,} characters; the full extracted text remains available in Original Text.")

    st.subheader("3 · Configure and generate a summary")
    settings_cols = st.columns([1, 1.4, 1.2, 1.5])
    summary_length = settings_cols[0].selectbox("Length", ["Short", "Medium", "Detailed"], index=1)
    summary_style = settings_cols[1].selectbox("Style", ["General", "Academic", "Simple Language", "Bullet Points"])
    language = settings_cols[2].text_input("Output language", value="English", max_chars=40)
    include_conclusion = settings_cols[3].checkbox("Include conclusion when supported", value=True)
    can_use_ai = bool(get_api_key())
    if len(document.text) > MAX_DOCUMENT_CHARS:
        st.warning("This document is above the configured AI processing limit. Extraction is available, but AI requests are disabled for this document.")
    generate_disabled = not can_use_ai or len(document.text) > MAX_DOCUMENT_CHARS or not language.strip()
    if st.button("✨ Generate Summary", type="primary", disabled=generate_disabled, use_container_width=False):
        try:
            service = GeminiService()
            with st.spinner("Reading the document and preparing the summary…"):
                result = summarize_document(
                    document.text,
                    summary_length,
                    summary_style,
                    language.strip(),
                    include_conclusion,
                    generate_text=service.generate_text,
                )
            st.session_state.summary_result = result
            st.session_state.summary_settings = {
                "length": summary_length,
                "style": summary_style,
                "language": language.strip(),
            }
            st.session_state.key_points = None
            st.session_state.keywords = None
            st.session_state.generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
            st.success("Summary generated.")
        except Exception as exc:
            st.error(_friendly_error(exc))
    if not can_use_ai:
        st.caption("Configure the Gemini API key in .env to enable summary, key point, keyword, and question features.")
    if len(document.text) > SUMMARY_CHUNK_CHARS:
        st.caption("Long documents are summarized in overlapping sections, then combined. The full source must fit within the configured document and chunk limits.")

    st.subheader("4 · Results")
    summary_tab, points_tab, keywords_tab, ask_tab, original_tab = st.tabs(
        ["Summary", "Key Points", "Keywords", "Ask Document", "Original Text"]
    )
    with summary_tab:
        summary_result = st.session_state.summary_result
        if summary_result is None:
            st.info("Your summary will appear here. Choose the length and style above, then select Generate Summary.")
        else:
            settings = st.session_state.summary_settings or {}
            st.caption(f"{settings.get('style', 'General')} · {settings.get('length', 'Medium')} · {settings.get('language', 'English')} · {len(summary_result.summary.split()):,} words")
            if summary_result.used_chunks:
                st.info(f"Two-stage chunk summarization used {summary_result.chunk_count} overlapping sections. Each source section was processed before the intermediate summaries were combined.")
            else:
                st.caption("Processed in one Gemini request.")
            st.markdown(summary_result.summary)
            with st.expander("AI output note"):
                st.caption("This is an AI-generated interpretation and may contain mistakes. Review facts and figures against the original document.")

    key_source = document.text if len(document.text) <= SUMMARY_CHUNK_CHARS else (st.session_state.summary_result.summary if st.session_state.summary_result else "")
    key_source_label = "Full extracted document text" if len(document.text) <= SUMMARY_CHUNK_CHARS else "Generated summary (used for long-document extraction)"
    with points_tab:
        st.caption(f"Source: {key_source_label}")
        if st.session_state.key_points is None:
            st.info("Generate a summary first, then extract the document’s main arguments and findings.")
            if st.button("Extract Key Points", disabled=not can_use_ai or st.session_state.summary_result is None, key="extract_key_points"):
                try:
                    service = GeminiService()
                    with st.spinner("Extracting key points…"):
                        st.session_state.key_points = extract_key_points(key_source, service.generate_text)
                except Exception as exc:
                    st.error(_friendly_error(exc))
        else:
            for index, point in enumerate(st.session_state.key_points, 1):
                st.markdown(f"{index}. {point}")
            if st.button("Regenerate Key Points", disabled=not can_use_ai, key="regenerate_key_points"):
                try:
                    service = GeminiService()
                    with st.spinner("Extracting key points…"):
                        st.session_state.key_points = extract_key_points(key_source, service.generate_text)
                except Exception as exc:
                    st.error(_friendly_error(exc))
    with keywords_tab:
        st.caption(f"Source: {key_source_label}")
        if st.session_state.keywords is None:
            st.info("Generate a summary first, then extract useful recurring terms and concepts.")
            if st.button("Extract Keywords", disabled=not can_use_ai or st.session_state.summary_result is None, key="extract_keywords"):
                try:
                    service = GeminiService()
                    with st.spinner("Finding central keywords…"):
                        st.session_state.keywords = extract_keywords(key_source, service.generate_text)
                except Exception as exc:
                    st.error(_friendly_error(exc))
        else:
            st.markdown("\n".join(f"- `{value}`" for value in st.session_state.keywords))
            if st.button("Regenerate Keywords", disabled=not can_use_ai, key="regenerate_keywords"):
                try:
                    service = GeminiService()
                    with st.spinner("Finding central keywords…"):
                        st.session_state.keywords = extract_keywords(key_source, service.generate_text)
                except Exception as exc:
                    st.error(_friendly_error(exc))
    with ask_tab:
        st.caption("Answers use the document as the source of truth. Long documents are searched by simple word overlap; this is not a guarantee of factual accuracy.")
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if can_use_ai:
            question = st.chat_input("Ask a question about this document", key="document_question")
            if question:
                st.session_state.chat_history.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)
                try:
                    service = GeminiService()
                    with st.spinner("Finding relevant passages…"):
                        answer = answer_question(
                            document.text,
                            question,
                            generate_text=service.generate_text,
                            history=st.session_state.chat_history[:-1],
                        )
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                except Exception as exc:
                    error = _friendly_error(exc)
                    st.session_state.chat_history.append({"role": "assistant", "content": error})
                    st.error(error)
        else:
            st.info("Configure GEMINI_API_KEY to ask questions about this document.")
    with original_tab:
        st.caption(f"{meta.character_count:,} extracted characters · shown in a scrollable field")
        st.text_area("Full extracted text", document.text, height=420, disabled=True, label_visibility="collapsed")

    if st.session_state.summary_result is not None:
        st.subheader("5 · Export report")
        result = st.session_state.summary_result
        download_stem = sanitize_filename(meta.filename)
        points = st.session_state.key_points
        keywords = st.session_state.keywords
        export_cols = st.columns(3)
        txt = generate_txt_report(meta.filename, result.summary, points, keywords, st.session_state.generated_at)
        md = generate_markdown_report(meta.filename, result.summary, points, keywords, st.session_state.generated_at)
        export_cols[0].download_button("Download TXT", txt, file_name=f"{download_stem}_smartdoc_report.txt", mime="text/plain", use_container_width=True)
        export_cols[1].download_button("Download Markdown", md, file_name=f"{download_stem}_smartdoc_report.md", mime="text/markdown", use_container_width=True)
        try:
            pdf = generate_pdf_report(meta.filename, result.summary, points, keywords, st.session_state.generated_at)
            export_cols[2].download_button("Download PDF", pdf, file_name=f"{download_stem}_smartdoc_report.pdf", mime="application/pdf", use_container_width=True)
        except RuntimeError as exc:
            export_cols[2].button("PDF unavailable", disabled=True, help=str(exc), use_container_width=True)
        st.caption("Reports include only generated results and the source filename. API credentials are never included.")

    if st.button("Clear document", key="clear_document"):
        _reset_document_state()
        st.session_state.upload_widget_version += 1
        st.rerun()
elif not st.session_state.document_error:
    st.markdown(
        '<div class="smartdoc-card"><strong>Ready when you are.</strong><br/>Upload a document to inspect its text and metadata. No summary is generated until you request one.</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="smartdoc-footer">AI-generated insights can be inaccurate; verify important information against the original source. Document text is sent to Google Gemini only when an AI feature is requested.</div>',
    unsafe_allow_html=True,
)
