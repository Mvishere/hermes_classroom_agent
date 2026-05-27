"""Centralized configuration and environment handling.

Loads .env values and exposes settings for other modules.
"""

from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv(override=True)

BASE_DIR = Path(__file__).resolve().parent

GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
GOOGLE_TOKEN_PATH = os.getenv("GOOGLE_TOKEN_PATH", str(BASE_DIR / "token.json"))

RAG_CHUNK_SIZE_CHARS = int(os.getenv("RAG_CHUNK_SIZE_CHARS", "").strip())
RAG_CHUNK_OVERLAP_CHARS = int(os.getenv("RAG_CHUNK_OVERLAP_CHARS", "").strip())
RAG_EXTRACT_MAX_CHARS = int(os.getenv("RAG_EXTRACT_MAX_CHARS", "").strip())
RAG_CONTEXT_MAX_CHARS = int(os.getenv("RAG_CONTEXT_MAX_CHARS", "").strip())
CHAT_MAX_HISTORY_TURNS = int(os.getenv("CHAT_MAX_HISTORY_TURNS", "").strip())

POLL_INTERVAL_MINUTES = int(os.getenv("POLL_INTERVAL_MINUTES", "5"))

DOWNLOAD_DIRECTORY = Path(os.getenv("DOWNLOAD_DIRECTORY", str(BASE_DIR / "downloads")))
DATA_DIRECTORY = Path(os.getenv("DATA_DIRECTORY", str(BASE_DIR / "data")))
LOG_DIRECTORY = Path(os.getenv("LOG_DIRECTORY", str(BASE_DIR / "logs")))
STATE_PATH = Path(os.getenv("STATE_PATH", str(DATA_DIRECTORY / "state" / "processed.json")))
USER_STATUS_PATH = Path(
    os.getenv("USER_STATUS_PATH", str(DATA_DIRECTORY / "state" / "user_status.json"))
)
RAG_INDEX_PATH = Path(
    os.getenv("RAG_INDEX_PATH", str(DATA_DIRECTORY / "rag" / "materials_index.json"))
)
RAG_SUMMARIES_PATH = Path(
    os.getenv("RAG_SUMMARIES_PATH", str(DATA_DIRECTORY / "rag" / "summaries.json"))
)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "5"))
API_BACKOFF_BASE_SECONDS = float(os.getenv("API_BACKOFF_BASE_SECONDS", "1.0"))
API_THROTTLE_SECONDS = float(os.getenv("API_THROTTLE_SECONDS", "0.2"))
OAUTH_LOCAL_SERVER_PORT = int(os.getenv("OAUTH_LOCAL_SERVER_PORT", "0"))
FETCH_SUBMISSIONS = os.getenv("FETCH_SUBMISSIONS", "0").strip() == "1"

RAG_ENABLED = os.getenv("RAG_ENABLED", "1").strip() == "1"
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "").strip()
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", "").strip()
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RAG_MAX_NEW_TOKENS = int(os.getenv("RAG_MAX_NEW_TOKENS", "256"))
RAG_TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.2"))
RAG_DEVICE = os.getenv("RAG_DEVICE", "cpu").strip()
PDF_EXTRACT_ENABLED = os.getenv("PDF_EXTRACT_ENABLED", "1").strip() == "1"
PDF_EXTRACT_MAX_CHARS = int(os.getenv("PDF_EXTRACT_MAX_CHARS", "6000"))
RAG_COMBINE_MAX_CHARS = int(os.getenv("RAG_COMBINE_MAX_CHARS", "6000"))
TOPIC_EXTRACT_MAX_CHARS = int(os.getenv("TOPIC_EXTRACT_MAX_CHARS", "4000"))
TOPIC_EXTRACT_LLM_ENABLED = os.getenv("TOPIC_EXTRACT_LLM_ENABLED", "1").strip() == "1"
TOPIC_EXTRACT_KEYWORD_LIMIT = int(os.getenv("TOPIC_EXTRACT_KEYWORD_LIMIT", "12"))

SCOPES = [
    "https://www.googleapis.com/auth/classroom.courses.readonly",
    "https://www.googleapis.com/auth/classroom.coursework.me.readonly",
    "https://www.googleapis.com/auth/classroom.courseworkmaterials.readonly",
    "https://www.googleapis.com/auth/classroom.student-submissions.me.readonly",
    "https://www.googleapis.com/auth/classroom.announcements.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]


def ensure_directories() -> None:
    """Create local directories used by the agent."""
    DOWNLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for folder in ("assignments", "materials", "announcements"):
        (DATA_DIRECTORY / folder).mkdir(parents=True, exist_ok=True)
    (DATA_DIRECTORY / "courses").mkdir(parents=True, exist_ok=True)
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    USER_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAG_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAG_SUMMARIES_PATH.parent.mkdir(parents=True, exist_ok=True)


def validate_config() -> None:
    """Validate required configuration values."""
    missing = []
    if not GOOGLE_APPLICATION_CREDENTIALS:
        missing.append("GOOGLE_APPLICATION_CREDENTIALS")
    if missing:
        raise ValueError("Missing required environment variables: " + ", ".join(missing))
