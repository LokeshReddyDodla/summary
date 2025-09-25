# Report Summarization Bot

A FastAPI service that ingests up to five PDF, DOCX, or TXT documents, summarizes them with OpenAI GPT models, and returns an aggregated report summary plus a downloadable text file.

## Features
- Upload up to five documents per request (PDF, DOCX, TXT)
- Automatic text extraction (PyPDF2, python-docx, raw text)
- Normalization and chunking to stay within model context limits
- Chunk-wise GPT summaries combined into a final report
- Downloadable `.txt` summary token endpoint
- Basic `/health` check for uptime monitoring

## Requirements
- Python 3.10+
- OpenAI API key with access to the configured chat completion model

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration
Environment variables are loaded automatically from a local `.env` file if present. Duplicate `.env` from the template and update it with your credentials:

```bash
cp .env .env.local  # optional step if you prefer a non-tracked file
```

Set environment variables before running the app (either via `.env` or shell exports):

| Variable | Description | Default |
| --- | --- | --- |
| `OPENAI_API_KEY` | OpenAI API key (required) | `None` |
| `OPENAI_MODEL` | Chat completion model identifier | `gpt-4o-mini` |
| `CHUNK_TOKEN_LIMIT` | Max token-like word count per chunk | `2500` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |

Example (macOS/Linux):

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini
```

## Running the Service
Start the FastAPI app with uvicorn:

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/summarize-report` | Upload 1-5 files and receive the aggregated summary and download token |
| `GET` | `/download-summary/{token}` | Download the generated summary as a `.txt` file |
| `GET` | `/health` | Lightweight health probe |

### `/summarize-report`
- Request: `multipart/form-data` with one to five `files`
- Response JSON:
  ```json
  {
    "summary": "...",
    "download_token": "UUID",
    "message": "Summary generated successfully."
  }
  ```

### `/download-summary/{token}`
- Use the `download_token` from the summarize response
- Returns a `.txt` attachment containing the final summary

## Local Testing
Serve the HTML form at `static/test-client.html` or use tools like curl/Postman to upload files. The form lets you queue up to five documents per request and configure the API base URL (default `http://127.0.0.1:8000`). Example curl request:

```bash
curl -X POST \
  -F "files=@report.pdf" \
  -F "files=@appendix.docx" \
  http://127.0.0.1:8000/summarize-report
```

## Notes
- Large documents are chunked by word count approximation. Adjust `CHUNK_TOKEN_LIMIT` and `CHUNK_OVERLAP` via environment variables if needed.
- Summaries are kept in memory only for download tokens; restart clears stored summaries.
- Ensure network access to the OpenAI API from the host running the service.

## Flutter Frontend (Optional)
A minimal Flutter client lives in `flutter_frontend/` for a richer cross-platform experience.

### Prerequisites
- Flutter SDK 3.19+ installed and configured (`flutter doctor` passes)

### Run the App
```bash
cd flutter_frontend
flutter pub get
flutter run -d chrome  # or another device id
```

The Flutter UI lets you:
- Configure the FastAPI base URL (defaults to `http://127.0.0.1:8000`)
- Pick up to five PDF/DOCX/TXT files (uses `file_picker`)
- Submit to `/summarize-report` and display the combined summary
- Launch the `/download-summary/{token}` link in an external browser/app
