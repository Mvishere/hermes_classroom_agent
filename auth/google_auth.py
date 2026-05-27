"""OAuth 2.0 authentication for Classroom and Drive services.

Persists OAuth tokens locally and reuses them across runs.
"""

from pathlib import Path
import logging
import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from config import (
    GOOGLE_APPLICATION_CREDENTIALS,
    GOOGLE_TOKEN_PATH,
    SCOPES,
    OAUTH_LOCAL_SERVER_PORT,
)

_CACHED_CREDS = None


def get_credentials() -> Credentials:
    """Load or refresh OAuth credentials for Classroom and Drive APIs."""
    global _CACHED_CREDS

    # Google may return a narrower scope set; relax strict matching to avoid failures.
    os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")

    if _CACHED_CREDS and _CACHED_CREDS.valid:
        return _CACHED_CREDS

    if not GOOGLE_APPLICATION_CREDENTIALS:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS is not set.")

    token_path = Path(GOOGLE_TOKEN_PATH)
    creds = None

    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        except Exception as exc:
            logging.warning("Failed to load token file: %s", exc)
            creds = None

    if creds and creds.scopes and set(creds.scopes) != set(SCOPES):
        logging.warning("OAuth scopes changed, re-authentication required.")
        creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logging.info("Refreshing OAuth token.")
            creds.refresh(Request())
        else:
            logging.info("Starting OAuth flow.")
            flow = InstalledAppFlow.from_client_secrets_file(
                GOOGLE_APPLICATION_CREDENTIALS, SCOPES
            )
            creds = flow.run_local_server(port=OAUTH_LOCAL_SERVER_PORT)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")

    _CACHED_CREDS = creds
    return creds


def get_classroom_service():
    """Build a Google Classroom API service."""
    creds = get_credentials()
    return build("classroom", "v1", credentials=creds, cache_discovery=False)


def get_drive_service():
    """Build a Google Drive API service."""
    creds = get_credentials()
    return build("drive", "v3", credentials=creds, cache_discovery=False)
