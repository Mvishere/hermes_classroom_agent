"""CLI entrypoint for persistent Playwright-based Google Form extraction."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import config
from quiz_extractor.browser.playwright_client import PlaywrightBrowserClient
from quiz_extractor.storage.quiz_storage import QuizStorage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Google Form quiz extractor")
    parser.add_argument(
        "--profile-dir",
        default=str(config.QUIZ_BROWSER_PROFILE_DIR),
        help="Persistent Chromium profile directory.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Chromium headless instead of showing the browser.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=config.QUIZ_BROWSER_TIMEOUT_MS,
        help="Page and selector timeout in milliseconds.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    login_parser = subparsers.add_parser(
        "login", help="Open Chromium with a persistent profile so you can sign in once."
    )
    login_parser.add_argument(
        "--url",
        default="https://docs.google.com/forms/",
        help="URL to open while signing in.",
    )

    extract_parser = subparsers.add_parser(
        "extract", help="Extract a rendered Google Form into structured JSON."
    )
    extract_parser.add_argument("--url", required=True, help="Google Form URL to extract.")
    extract_parser.add_argument("--quiz-id", default="", help="Optional quiz identifier.")
    extract_parser.add_argument(
        "--course-id",
        default="",
        help="Optional course identifier for structured storage.",
    )
    extract_parser.add_argument(
        "--course-name",
        default="",
        help="Optional course name for structured storage.",
    )
    extract_parser.add_argument(
        "--assignment-id",
        default="",
        help="Optional assignment identifier for structured storage.",
    )
    extract_parser.add_argument(
        "--title",
        default="",
        help="Optional fallback title.",
    )
    extract_parser.add_argument(
        "--output",
        default="",
        help="Optional output path for the JSON payload.",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=config.LOG_LEVEL)

    browser = PlaywrightBrowserClient(
        Path(args.profile_dir),
        headless=args.headless,
        timeout_ms=args.timeout_ms,
    )
    storage = QuizStorage(config.QUIZ_STORAGE_PATH)

    try:
        if args.command == "login":
            page = browser.open_page(args.url)
            try:
                logging.info("Browser opened. Sign in manually if needed, then press Enter here.")
                input("Press Enter after you have completed Google sign-in in the browser... ")
            finally:
                page.close()
            return 0

        if args.command == "extract":
            quiz = browser.extract_form(
                args.url,
                form_id=args.quiz_id,
                fallback_title=args.title,
            )
            if args.course_id:
                quiz["course_id"] = args.course_id
            if args.course_name:
                quiz["course_name"] = args.course_name
            if args.assignment_id:
                quiz["assignment_id"] = args.assignment_id

            storage.upsert_quiz(quiz)

            if args.output:
                Path(args.output).write_text(
                    json.dumps(quiz, ensure_ascii=True, indent=2), encoding="utf-8"
                )
            print(json.dumps(quiz, ensure_ascii=True, indent=2))
            return 0
    finally:
        browser.close()

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
