"""Playwright client for authenticated Google Form browsing."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any


class PlaywrightBrowserClient:
    """Persistent Chromium context used to keep a signed-in Google session."""

    def __init__(
        self,
        user_data_dir: Path,
        headless: bool = False,
        timeout_ms: int = 45000,
        max_retries: int = 2,
        retry_delay_ms: int = 1000,
    ) -> None:
        self.user_data_dir = Path(user_data_dir)
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.max_retries = max_retries
        self.retry_delay_ms = retry_delay_ms
        self._playwright = None
        self._context = None

    def _ensure_context(self) -> Any:
        if self._context is not None:
            return self._context

        from playwright.sync_api import sync_playwright

        self.user_data_dir.mkdir(parents=True, exist_ok=True)
        self._playwright = sync_playwright().start()
        browser_executable = self._resolve_browser_executable()
        launch_kwargs = {
            "user_data_dir": str(self.user_data_dir),
            "headless": self.headless,
            "viewport": {"width": 1440, "height": 1800},
            "args": ["--disable-blink-features=AutomationControlled"],
        }
        if browser_executable is not None:
            launch_kwargs["executable_path"] = str(browser_executable)

        try:
            self._context = self._playwright.chromium.launch_persistent_context(**launch_kwargs)
        except Exception as exc:
            self._playwright.stop()
            self._playwright = None
            raise RuntimeError(self._browser_install_message()) from exc
        self._context.set_default_timeout(self.timeout_ms)
        self._context.set_default_navigation_timeout(self.timeout_ms)
        logging.info("Started persistent Chromium profile at %s", self.user_data_dir)
        return self._context

    def _resolve_browser_executable(self) -> Path | None:
        override = os.getenv("QUIZ_BROWSER_EXECUTABLE_PATH", "").strip()
        if override:
            candidate = Path(override).expanduser()
            if candidate.exists():
                return candidate
            raise FileNotFoundError(f"QUIZ_BROWSER_EXECUTABLE_PATH does not exist: {candidate}")

        candidates = [
            *sorted(
                Path.home().glob("AppData/Local/ms-playwright/chromium-*/chrome-win*/chrome.exe"),
                reverse=True,
            ),
            Path.home() / "AppData" / "Local" / "ms-playwright" / "chromium" / "chrome-win" / "chrome.exe",
            Path(os.getenv("PROGRAMFILES", r"C:\Program Files")) / "Google" / "Chrome" / "Application" / "chrome.exe",
            Path(os.getenv("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "Google" / "Chrome" / "Application" / "chrome.exe",
            Path(os.getenv("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "Google" / "Chrome" / "Application" / "chrome.exe",
            Path(os.getenv("PROGRAMFILES", r"C:\Program Files")) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
            Path(os.getenv("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
            Path(os.getenv("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _browser_install_message(self) -> str:
        return (
            "Playwright Chromium is not installed. Run `python -m playwright install chromium` "
            "inside the active virtual environment, or set QUIZ_BROWSER_EXECUTABLE_PATH to an "
            "existing Chrome/Edge executable."
        )

    def new_page(self):
        """Create a new page from the persistent browser context."""
        context = self._ensure_context()
        page = context.new_page()
        page.set_default_timeout(self.timeout_ms)
        page.set_default_navigation_timeout(self.timeout_ms)
        return page

    def open_page(self, url: str):
        """Open a URL in a fresh page and wait for the content to settle."""
        page = self.new_page()
        self._goto_with_retries(page, url)
        return page

    def extract_form(self, form_url: str, form_id: str = "", fallback_title: str = "") -> dict:
        """Open a form URL and return the parsed quiz payload."""
        from quiz_extractor.parser.form_parser import GoogleFormParser

        page = self.open_page(form_url)
        try:
            parser = GoogleFormParser(timeout_ms=self.timeout_ms)
            return parser.extract(page, form_url=form_url, quiz_id=form_id, fallback_title=fallback_title)
        finally:
            try:
                page.close()
            except Exception:
                logging.debug("Failed to close Playwright page for %s", form_url, exc_info=True)

    def _goto_with_retries(self, page, url: str) -> None:
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                logging.info("Opening %s (attempt %s/%s)", url, attempt + 1, self.max_retries + 1)
                page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)
                try:
                    page.wait_for_load_state("networkidle", timeout=self.timeout_ms)
                except Exception:
                    logging.debug("Network idle wait timed out for %s", url, exc_info=True)
                return
            except Exception as exc:
                last_error = exc
                logging.warning(
                    "Failed to open %s on attempt %s/%s: %s",
                    url,
                    attempt + 1,
                    self.max_retries + 1,
                    exc,
                )
                if attempt < self.max_retries:
                    page.wait_for_timeout(self.retry_delay_ms)
        if last_error is not None:
            raise last_error

    def close(self) -> None:
        """Close the browser context and stop Playwright."""
        if self._context is not None:
            try:
                self._context.close()
            finally:
                self._context = None
        if self._playwright is not None:
            try:
                self._playwright.stop()
            finally:
                self._playwright = None

    def __enter__(self):
        self._ensure_context()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
