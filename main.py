"""FastAPI application for generating summaries from uploaded reports."""
from __future__ import annotations

import asyncio
import re
import uuid
from functools import lru_cache
from io import BytesIO
from threading import Lock
from typing import Dict, Iterable, List, Optional

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi import File as FastAPIFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from docx import Document
from docx.opc.exceptions import PackageNotFoundError

load_dotenv()  # Load environment variables from a .env file if present.

app = FastAPI(title="Report Summarization Bot", version="1.0.0")

# CORS middleware with production-ready settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        "https://summary.lokeshreddydodla.workers.dev",
        "*",  # Allow all origins for development - restrict in production
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


class Settings(BaseSettings):
    """Application configuration sourced from environment variables."""

    openai_api_key: Optional[str] = Field(default=None)
    openai_model: str = Field(default="gpt-4o-mini")
    chunk_token_limit: int = Field(default=2500)
    chunk_overlap: int = Field(default=200)

    class Config:
        env_file = ".env"
        extra = "forbid"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


class SummaryStore:
    """In-memory storage for generated summaries with download tokens."""

    def __init__(self) -> None:
        self._store: Dict[str, str] = {}
        self._lock = Lock()

    def save(self, summary_text: str) -> str:
        token = str(uuid.uuid4())
        with self._lock:
            self._store[token] = summary_text
        return token

    def get(self, token: str) -> Optional[str]:
        with self._lock:
            return self._store.get(token)


class GPTSummarizer:
    """Wrapper around the GPT API for chunked report summarization."""


    async def summarize_chunk(self, text: str) -> str:
        prompt = (
            "Summarize the following text into key points and a concise narrative:\n"
            f"{text}"
        )
        return await self._complete(prompt)

    async def summarize_overview(self, chunk_summaries: Iterable[str]) -> str:
        combined = "\n\n".join(chunk_summaries)
        prompt = (
            "Combine the following partial summaries into a single cohesive report summary with key points "
            "and a concise narrative:\n"
            f"{combined}"
        )
        return await self._complete(prompt)

    async def _complete(self, prompt: str) -> str:
        def _call_openai() -> str:
            completion = self._client.chat.completions.create(
                model=self._model,
                temperature=0.3,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a world-class assistant that distills long reports into accurate "
                            "summaries with clear key points and narratives."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return completion.choices[0].message.content.strip()

        try:
            return await asyncio.to_thread(_call_openai)
        except OpenAIError as exc:
            raise HTTPException(
                status_code=502,
                detail="Unable to generate summary via GPT at this time. Please try again later.",
            ) from exc


summary_store = SummaryStore()
_summarizer_lock = Lock()
_summarizer_instance: Optional[GPTSummarizer] = None


async def get_summarizer(settings: Settings = Depends(get_settings)) -> GPTSummarizer:
    global _summarizer_instance
    if _summarizer_instance is None:
        with _summarizer_lock:
            if _summarizer_instance is None:
                _summarizer_instance = GPTSummarizer(settings)
    return _summarizer_instance


SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".txt": "txt",
}
SUPPORTED_CONTENT_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain": "txt",
}
async def read_upload(upload_file: UploadFile) -> bytes:
    raw = await upload_file.read()
    if not raw:
        raise ValueError("The uploaded file is empty.")
    return raw


def detect_file_kind(upload_file: UploadFile) -> str:
    filename = upload_file.filename or ""
    ext = filename.lower().rsplit(".", 1)
    ext = f".{ext[-1]}" if len(ext) == 2 else ""
    if ext in SUPPORTED_EXTENSIONS:
        return SUPPORTED_EXTENSIONS[ext]
    content_type = (upload_file.content_type or "").lower()
    if content_type in SUPPORTED_CONTENT_TYPES:
        return SUPPORTED_CONTENT_TYPES[content_type]
    raise ValueError("Unsupported file format. Allowed types: PDF, DOCX, TXT.")


def extract_text(file_kind: str, raw: bytes) -> str:
    if file_kind == "pdf":
        return extract_text_from_pdf(raw)
    if file_kind == "docx":
        return extract_text_from_docx(raw)
    if file_kind == "txt":
        return extract_text_from_txt(raw)
    raise ValueError("Unsupported file format.")


def extract_text_from_pdf(raw: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(raw))
    except PdfReadError as exc:
        raise ValueError("The PDF file appears to be corrupted or unreadable.") from exc

    pages: List[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - PyPDF2 internal error handling
            raise ValueError("Failed to extract text from one of the PDF pages.") from exc
        if page_text:
            pages.append(page_text)

    if not pages:
        raise ValueError("No extractable text found in PDF.")

    return "\n".join(pages)


def extract_text_from_docx(raw: bytes) -> str:
    try:
        document = Document(BytesIO(raw))
    except PackageNotFoundError as exc:
        raise ValueError("The DOCX file appears to be corrupted or unreadable.") from exc

    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    if not paragraphs:
        raise ValueError("No readable text found in DOCX.")

    return "\n".join(paragraphs)


def extract_text_from_txt(raw: bytes) -> str:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw.decode("latin-1")
        except UnicodeDecodeError as exc:
            raise ValueError("Unable to decode TXT file. Please provide UTF-8 or Latin-1 encoded text.") from exc

    if not text.strip():
        raise ValueError("The TXT file is empty.")

    return text


def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines: List[str] = []
    header_footer_pattern = re.compile(r"^(page\s+\d+|\d+)$", re.IGNORECASE)

    for line in lines:
        if not line:
            continue
        if header_footer_pattern.match(line):
            continue
        cleaned_lines.append(line)

    normalized = " ".join(cleaned_lines)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def chunk_text(text: str, max_tokens: int, overlap: int) -> List[str]:
    if not text:
        return []

    tokens = text.split()
    if not tokens:
        return []

    if max_tokens <= 0:
        raise ValueError("Chunk token limit must be greater than zero.")

    overlap = max(0, min(overlap, max_tokens // 2))

    chunks: List[str] = []
    start = 0
    total_tokens = len(tokens)

    while start < total_tokens:
        end = min(total_tokens, start + max_tokens)
        chunk_tokens = tokens[start:end]
        chunks.append(" ".join(chunk_tokens))
        if end == total_tokens:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


@app.post("/summarize-report")
async def summarize_report(
    files: List[UploadFile] = FastAPIFile(..., description="Upload up to 5 documents for summarization."),
    summarizer: GPTSummarizer = Depends(get_summarizer),
    settings: Settings = Depends(get_settings),
) -> JSONResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one document must be uploaded.")
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="You can upload a maximum of 5 documents at once.")

    extracted_texts: List[str] = []
    for upload in files:
        try:
            file_kind = detect_file_kind(upload)
            raw = await read_upload(upload)
            extracted = extract_text(file_kind, raw)
            normalized = normalize_text(extracted)
            if not normalized:
                raise ValueError("The document does not contain any usable text after preprocessing.")
            extracted_texts.append(normalized)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"{upload.filename}: {exc}") from exc
        finally:
            await upload.close()

    aggregated_text = " \n ".join(extracted_texts)

    chunks = chunk_text(
        aggregated_text,
        max_tokens=settings.chunk_token_limit,
        overlap=settings.chunk_overlap,
    )

    if not chunks:
        raise HTTPException(status_code=400, detail="Unable to extract meaningful text from the uploaded documents.")

    chunk_summaries: List[str] = []
    for chunk in chunks:
        chunk_summary = await summarizer.summarize_chunk(chunk)
        chunk_summaries.append(chunk_summary)

    if len(chunk_summaries) == 1:
        final_summary = chunk_summaries[0]
    else:
        final_summary = await summarizer.summarize_overview(chunk_summaries)

    download_token = summary_store.save(final_summary)

    return JSONResponse(
        status_code=200,
        content={
            "summary": final_summary,
            "download_token": download_token,
            "message": "Summary generated successfully.",
        },
    )


@app.get("/download-summary/{token}")
async def download_summary(token: str) -> Response:
    summary_text = summary_store.get(token)
    if summary_text is None:
        raise HTTPException(status_code=404, detail="Summary not found or has expired.")

    filename = f"summary-{token}.txt"
    headers = {
        "Content-Disposition": f"attachment; filename=\"{filename}\"",
        "X-Content-Type-Options": "nosniff",
    }
    return Response(content=summary_text, media_type="text/plain", headers=headers)


@app.get("/")
async def root() -> Dict[str, str]:
    return {
        "message": "Report Summarization Bot API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "summarize": "/summarize-report",
            "web_client": "/static/test-client.html"
        }
    }

@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}
