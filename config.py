"""Centralized configuration and environment handling.

Loads .env values and exposes settings for other modules.
"""

from pathlib import Path
import os

from dotenv import load_dotenv

# Load .env but do not override explicit environment variables set in the shell.
load_dotenv(override=False)

BASE_DIR = Path(__file__).resolve().parent


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default

GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
GOOGLE_TOKEN_PATH = os.getenv("GOOGLE_TOKEN_PATH", str(BASE_DIR / "token.json"))

RAG_CHUNK_SIZE_CHARS = _env_int("RAG_CHUNK_SIZE_CHARS", 2000)
RAG_CHUNK_OVERLAP_CHARS = _env_int("RAG_CHUNK_OVERLAP_CHARS", 200)
RAG_EXTRACT_MAX_CHARS = _env_int("RAG_EXTRACT_MAX_CHARS", 6000)
RAG_CONTEXT_MAX_CHARS = _env_int("RAG_CONTEXT_MAX_CHARS", 6000)
CHAT_MAX_HISTORY_TURNS = _env_int("CHAT_MAX_HISTORY_TURNS", 3)

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
QUIZ_STORAGE_PATH = Path(
    os.getenv("QUIZ_STORAGE_PATH", str(DATA_DIRECTORY / "quizzes" / "quizzes.json"))
)
QUIZ_BROWSER_PROFILE_DIR = Path(
    os.getenv(
        "QUIZ_BROWSER_PROFILE_DIR",
        str(DATA_DIRECTORY / "browser_profiles" / "google_forms"),
    )
)
QUIZ_BROWSER_HEADLESS = os.getenv("QUIZ_BROWSER_HEADLESS", "0").strip() == "1"
QUIZ_BROWSER_TIMEOUT_MS = int(os.getenv("QUIZ_BROWSER_TIMEOUT_MS", "45000"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "5"))
API_BACKOFF_BASE_SECONDS = _env_float("API_BACKOFF_BASE_SECONDS", 1.0)
API_THROTTLE_SECONDS = _env_float("API_THROTTLE_SECONDS", 0.2)
OAUTH_LOCAL_SERVER_PORT = int(os.getenv("OAUTH_LOCAL_SERVER_PORT", "0"))
FETCH_SUBMISSIONS = os.getenv("FETCH_SUBMISSIONS", "0").strip() == "1"

RAG_ENABLED = os.getenv("RAG_ENABLED", "1").strip() == "1"
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "").strip()
QWEN_MODEL_PATH = os.getenv(
    "QWEN_MODEL_PATH",
    str(BASE_DIR / "models" / "qwen2.5-7b-instruct"),
).strip()
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", QWEN_MODEL_PATH).strip()
LLM_BACKEND = os.getenv("LLM_BACKEND", "transformers").strip().lower()
LLM_CONTEXT_LENGTH = _env_int("LLM_CONTEXT_LENGTH", 8192)
LLM_MAX_NEW_TOKENS = _env_int("LLM_MAX_NEW_TOKENS", 1024)
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.2)
LLM_TOP_P = _env_float("LLM_TOP_P", 0.9)
LLM_DEVICE = os.getenv("LLM_DEVICE", "cpu").strip()
LLM_GPU_LAYERS = _env_int("LLM_GPU_LAYERS", 0)
LLM_NUM_THREADS = _env_int("LLM_NUM_THREADS", 8)
LLM_QUANTIZATION = os.getenv("LLM_QUANTIZATION", "auto").strip().lower()
LLM_ENABLE_STREAMING = os.getenv("LLM_ENABLE_STREAMING", "0").strip() == "1"
LLM_TIMEOUT_SECONDS = _env_int("LLM_TIMEOUT_SECONDS", 120)
LLM_SYSTEM_PROMPT = os.getenv(
    "LLM_SYSTEM_PROMPT",
    "You are a grounded educational assistant. Only answer using retrieved course data. If information is missing, explicitly say so.",
).strip()
RAG_TOP_K = _env_int("RAG_TOP_K", 3)
RAG_MAX_NEW_TOKENS = _env_int("RAG_MAX_NEW_TOKENS", 256)
RAG_TEMPERATURE = _env_float("RAG_TEMPERATURE", 0.2)
RAG_DEVICE = os.getenv("RAG_DEVICE", "cpu").strip()
PDF_EXTRACT_ENABLED = os.getenv("PDF_EXTRACT_ENABLED", "1").strip() == "1"
PDF_EXTRACT_MAX_CHARS = _env_int("PDF_EXTRACT_MAX_CHARS", 6000)
RAG_COMBINE_MAX_CHARS = _env_int("RAG_COMBINE_MAX_CHARS", 6000)
TOPIC_EXTRACT_MAX_CHARS = _env_int("TOPIC_EXTRACT_MAX_CHARS", 4000)
TOPIC_EXTRACT_LLM_ENABLED = os.getenv("TOPIC_EXTRACT_LLM_ENABLED", "1").strip() == "1"
TOPIC_EXTRACT_KEYWORD_LIMIT = _env_int("TOPIC_EXTRACT_KEYWORD_LIMIT", 12)
TOPIC_GRAPH_DEBUG = os.getenv("TOPIC_GRAPH_DEBUG", "0").strip() == "1"
TOPIC_GRAPH_MIN_EDGE_WEIGHT = _env_float("TOPIC_GRAPH_MIN_EDGE_WEIGHT", 0.65)
TOPIC_GRAPH_MAX_RELATED = _env_int("TOPIC_GRAPH_MAX_RELATED", 6)
TOPIC_GRAPH_MIN_TOPIC_FREQUENCY = _env_int("TOPIC_GRAPH_MIN_TOPIC_FREQUENCY", 2)

SCOPES = [
    "https://www.googleapis.com/auth/classroom.courses.readonly",
    "https://www.googleapis.com/auth/classroom.coursework.me.readonly",
    "https://www.googleapis.com/auth/classroom.courseworkmaterials.readonly",
    "https://www.googleapis.com/auth/classroom.student-submissions.me.readonly",
    "https://www.googleapis.com/auth/classroom.announcements.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/forms.body.readonly",
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
    QUIZ_STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUIZ_BROWSER_PROFILE_DIR.mkdir(parents=True, exist_ok=True)


def validate_config() -> None:
    """Validate required configuration values."""
    missing = []
    if not GOOGLE_APPLICATION_CREDENTIALS:
        missing.append("GOOGLE_APPLICATION_CREDENTIALS")
    if missing:
        raise ValueError("Missing required environment variables: " + ", ".join(missing))
