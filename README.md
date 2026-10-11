# SmartDoc AI — Intelligent Document Summarizer

SmartDoc AI is an academic mini-project that extracts text from a document and uses Google Gemini to produce a summary, key points, keywords, and grounded answers to questions. It demonstrates document processing, prompt design, API integration, a small Streamlit interface, and offline unit testing.

## Problem statement

Long reports and research documents can take time to review. This application helps a student inspect the main ideas by extracting document text and requesting a concise AI-generated interpretation. It also lets the user ask questions about the uploaded source.

## Motivation and objectives

- Demonstrate PDF, Word, and plain-text extraction in Python.
- Explore text normalization and character-based chunking.
- Practice safe prompt construction and calls through Google's official Gen AI SDK.
- Provide document-grounded summaries and answers with clear limitations.
- Keep the application understandable for an intermediate Python developer.

## Features

- Upload one PDF, DOCX, or TXT file, up to 10 MB.
- Display the extracted text, character and word counts, and PDF page count.
- Choose Short, Medium, or Detailed summaries; General, Academic, Simple Language, or Bullet Points style; an output language; and whether to include a conclusion.
- Process longer documents in overlapping sections, then combine the section summaries in one final request.
- Explicitly request key points or keywords and ask multiple questions in a chat-style tab.
- Download a TXT, Markdown, or PDF report.
- Keep uploads and generated results in Streamlit's per-session memory; the app does not save uploaded documents to disk.
- Show friendly errors for invalid files, missing API configuration, and provider errors.

## Technology stack

- Python 3.11 or newer
- Streamlit
- Google Gen AI Python SDK (`google-genai`) and Gemini API
- PyMuPDF for PDF text extraction
- `python-docx` for DOCX text and table extraction
- `python-dotenv` for local environment configuration
- Python's standard `json` module for validating model list responses
- ReportLab for PDF exports
- pytest for offline tests

## System architecture

```text
Streamlit UI (app.py)
  ├── document_processor.py  → in-memory validation, extraction, normalization
  ├── summarizer.py          → prompts, chunk splitting, summaries, points, keywords
  ├── question_answering.py  → keyword-overlap selection and grounded answers
  ├── gemini_service.py     → API key, SDK client, safe provider errors
  └── report_generator.py   → in-memory TXT, Markdown, and PDF reports

config.py → shared model and processing limits
tests/    → synthetic documents and mocked/offline AI behavior
```

## Processing workflow

1. Streamlit accepts a single upload and the application computes an in-memory fingerprint. A changed document clears its old summary, points, keywords, and chat history.
2. The document processor checks the extension, byte size, and practical file structure. It extracts PDF pages with PyMuPDF, DOCX paragraphs and tables with `python-docx`, or UTF-8/Windows-1252 text from TXT files.
3. Text cleaning normalizes line endings and whitespace while retaining punctuation and paragraph boundaries. The app displays the extracted text and available metadata.
4. Only after the user selects an AI action does the application create a Gemini client and send the relevant text to Google. The local upload is not written to disk.
5. Exports are generated in memory and contain the source filename, generated content, and AI-generated disclaimer. They never include the API key.

### Gemini API integration

`gemini_service.py` uses the official `google-genai` SDK and reads `GEMINI_API_KEY` from the process environment or `.env`. The client is created for an explicit AI action and reused across calls in that action (including a long-document summary). The API key is not shown in the page, prompts, logs, or reports. The configured model defaults to `gemini-3.8-flash`; set `GEMINI_MODEL` in `.env` to change it. The default was selected from Google's [Gemini API model guide](https://ai.google.dev/gemini-api/docs/models), which describes the available model IDs; model availability can change, so use a model supported by your API project.

Every source is marked as untrusted document content in the prompt. The summary instructions ask the model to preserve facts and acknowledge ambiguity instead of inventing details. AI responses can still be wrong and should be checked against the source.

### Two-stage summarization for long documents

The configured summary input budget is 12,000 characters per chunk, with up to 600 characters of overlap. If the full text fits in one chunk, one request produces the summary. Otherwise, the app summarizes each source chunk in 100–150 words, then sends the intermediate summaries to a final request with the user's chosen summary settings. The app reports when it used chunks. It rejects inputs above 360,000 characters or more than 30 chunks; it never silently drops source sections and calls the result a full-document summary. The process makes one API request per chunk plus one final request.

Chunk and character limits are conservative application settings in `config.py`. They can be adjusted for a model, quota, or class demonstration, but should be kept within the selected model's context limit.

### Lightweight document question answering

Short documents are supplied as a whole. For longer documents, the application splits text into passages and scores them by overlap with non-common words in the question. It sends a few top passages to Gemini rather than using embeddings or a vector database. If no relevant passage is found, the app says it could not locate enough evidence. This is basic document-grounded question answering, not a guarantee of factual accuracy or a sophisticated search system.

## Project structure

```text
.
├── app.py
├── config.py
├── document_processor.py
├── gemini_service.py
├── question_answering.py
├── report_generator.py
├── summarizer.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── tests/
    ├── test_document_processor.py
    ├── test_question_answering.py
    ├── test_report_generator.py
    └── test_summarizer.py
```

## Prerequisites

- Python 3.11 or newer (`python --version`)
- Internet access for installing packages and using Gemini features
- A Gemini API key from Google AI Studio for AI actions

## Installation and run on Windows PowerShell

From the project directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` in a text editor and replace the placeholder value for `GEMINI_API_KEY` with your own key. Keep `.env` private and out of source control. If the configured Gemini model is unavailable for your API project, set `GEMINI_MODEL` to a supported model identifier in `.env` and restart Streamlit.

Start the app and run the offline tests:

```powershell
streamlit run app.py
python -m pytest -q
```

If PowerShell blocks virtual-environment activation, avoid a system-wide policy change. For the current terminal only, you may run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, or skip activation and invoke `.venv\Scripts\python.exe -m pip ...` and `.venv\Scripts\streamlit.exe run app.py` directly.

## Installation and run on Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` to add your API key. Then start the app and test it:

```bash
streamlit run app.py
python -m pytest -q
```

## Publish this project to GitHub

Create a new **empty** repository on GitHub (do not initialize it with a README, license, or `.gitignore`, since this folder already contains those files). From PowerShell in this project folder, run:

```powershell
git init
git add .
git status --short
git commit -m "Add SmartDoc AI document summarizer"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/smartdoc-ai.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username and use the repository name you created. Before committing, inspect `git status --short`: `.env`, `.venv`, and Python cache files should not appear. `.env.example` is safe to publish because it contains only a placeholder. If an API key is ever committed, revoke it in Google AI Studio and remove it from the repository history before sharing the repository.

## Supported formats and limitations

- **PDF:** selectable text is extracted page by page. Encrypted PDFs are rejected with guidance. Scanned/image-only pages may contain no text; OCR is not part of this version.
- **DOCX:** paragraphs and table cells are extracted. Macros and embedded active content are not executed.
- **TXT:** UTF-8 (including a byte-order mark) is preferred, with Windows-1252 fallback.
- Uploads are limited to 10 MB. AI processing is limited to 360,000 extracted characters and 30 summary chunks.
- Long document summaries require one request per chunk plus a final aggregation request. A failure stops processing; partial summaries are not displayed as complete.
- PDF export uses ReportLab. If it is not installed, TXT and Markdown exports remain available.
- Gemini model names, quotas, prices, and availability can change. The configured model may need to be adjusted.

## Privacy and API cost

Document content remains in the Streamlit session unless an AI feature is requested. When you generate a summary, key points, keywords, or an answer, the relevant extracted text is sent to Google Gemini for processing. Do not upload confidential material unless you are authorized to share it. No unrelated service or database is used. Longer documents require more model calls and may incur higher API usage; check your Gemini API account's current quota and pricing before processing large files.

## Troubleshooting

- **API key missing:** check that `.env` is beside `app.py`, the variable is spelled `GEMINI_API_KEY`, and Streamlit was restarted after editing the file.
- **Invalid key/model:** verify the key and set `GEMINI_MODEL` to a model enabled for your API project, then restart Streamlit.
- **Quota or rate limit:** wait and retry later, reduce the document size, or check your API account quota.
- **PDF has no text:** it may be scanned. Convert it to a text-based PDF or use a smaller text export; OCR is not included.
- **Corrupt or unsupported upload:** re-export as a standard PDF, DOCX, or UTF-8 TXT file.
- **PowerShell activation blocked:** use the process-only policy command above, or run the virtual environment's Python/Streamlit executables without activation.
- **A test reports a missing library:** run `python -m pip install -r requirements.txt` in the active project environment.

## College demonstration outline

1. Show the empty state and API configuration status.
2. Upload a small synthetic PDF, DOCX, and TXT file in turn; point out the metadata and extracted text preview.
3. Generate a medium academic summary, then show the chunk status for a longer sample.
4. Extract key points and keywords, then ask a question whose answer is present and another whose answer is absent.
5. Download a Markdown or PDF report and explain the AI disclaimer.
6. Run `python -m pytest -q` to show that extraction and prompt logic can be tested without a Gemini key or network calls.

## Future enhancements

OCR support, citation locations for answers, richer language choices, and configurable user-level model settings could be added later. These features are not implemented in this version.
