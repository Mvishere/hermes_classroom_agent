"""Parse rendered Google Forms DOM into structured quiz JSON."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any


class GoogleFormParser:
    """Convert a rendered Google Form page into structured quiz data."""

    def __init__(self, timeout_ms: int = 45000) -> None:
        self.timeout_ms = timeout_ms

    def extract(
        self,
        page,
        form_url: str,
        quiz_id: str = "",
        fallback_title: str = "",
    ) -> dict:
        """Wait for the rendered form and extract its visible DOM content."""
        self._wait_for_form(page)
        snapshot = self._build_snapshot(page)
        return self.parse_snapshot(
            snapshot,
            form_url=form_url,
            quiz_id=quiz_id,
            fallback_title=fallback_title,
        )

    def _wait_for_form(self, page) -> None:
        """Wait for the page to render a form-like DOM."""
        selectors = [
            "div[role='listitem']",
            "fieldset",
            "div[role='group']",
            "input",
            "textarea",
            "[role='radio']",
            "[role='checkbox']",
        ]
        try:
            page.wait_for_load_state("domcontentloaded", timeout=self.timeout_ms)
        except Exception:
            logging.debug("DOM content load wait timed out", exc_info=True)

        for selector in selectors:
            try:
                page.wait_for_selector(selector, timeout=self.timeout_ms)
                return
            except Exception:
                continue

    def _build_snapshot(self, page) -> dict[str, Any]:
        """Use browser DOM evaluation to build a serializable form snapshot."""
        try:
            return page.evaluate(_SNAPSHOT_SCRIPT)
        except Exception:
            logging.exception("Failed to snapshot Google Form DOM")
            return {
                "title": page.title() if hasattr(page, "title") else "",
                "questions": [],
                "page_url": getattr(page, "url", ""),
                "fetch_status": "snapshot_error",
            }

    @staticmethod
    def parse_snapshot(
        snapshot: dict[str, Any],
        form_url: str,
        quiz_id: str = "",
        fallback_title: str = "",
    ) -> dict:
        """Convert a DOM snapshot into the storage-friendly quiz structure."""
        questions_payload: list[dict[str, Any]] = []
        section_titles: list[str] = []
        current_section = ""

        for raw_item in snapshot.get("questions", []) if isinstance(snapshot, dict) else []:
            if not isinstance(raw_item, dict):
                continue

            kind = str(raw_item.get("kind") or "question").strip() or "question"
            title = str(raw_item.get("title") or "").strip()
            if not title:
                continue

            if kind == "section_title":
                section_titles.append(title)
                current_section = title
                continue

            question = {
                "question": title,
                "options": _dedupe_list(raw_item.get("options", [])),
                "type": _normalize_question_type(kind),
            }
            if raw_item.get("required") is not None:
                question["required"] = bool(raw_item.get("required"))
            if current_section:
                question["section_title"] = current_section
            if raw_item.get("rows"):
                question["rows"] = _dedupe_list(raw_item.get("rows", []))
            if raw_item.get("columns"):
                question["columns"] = _dedupe_list(raw_item.get("columns", []))

            questions_payload.append(question)

        form_title = _coalesce(
            snapshot.get("title") if isinstance(snapshot, dict) else "",
            fallback_title,
        )
        resolved_quiz_id = _coalesce(
            quiz_id,
            snapshot.get("quiz_id") if isinstance(snapshot, dict) else "",
            snapshot.get("form_id") if isinstance(snapshot, dict) else "",
            form_url,
        )

        quiz_data = {
            "quiz_id": resolved_quiz_id,
            "form_id": snapshot.get("form_id") if isinstance(snapshot, dict) else "",
            "source_url": form_url,
            "title": form_title,
            "questions": questions_payload,
            "section_titles": _dedupe_list(section_titles),
            "quiz_text": render_quiz_text(form_title, questions_payload, section_titles),
            "fetch_status": snapshot.get("fetch_status", "rendered_dom")
            if isinstance(snapshot, dict)
            else "rendered_dom",
            "source": snapshot.get("source", "playwright_dom") if isinstance(snapshot, dict) else "playwright_dom",
            "page_url": snapshot.get("page_url", form_url) if isinstance(snapshot, dict) else form_url,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        }
        return quiz_data


def render_quiz_text(
    title: str,
    questions: list[dict[str, Any]],
    section_titles: list[str] | None = None,
) -> str:
    """Render quiz data into plain text for RAG and topic extraction."""
    lines: list[str] = []
    if title:
        lines.append(f"Quiz Title: {title}")
    for section in section_titles or []:
        if section:
            lines.append(f"Section: {section}")
    for index, question in enumerate(questions, start=1):
        question_text = str(question.get("question", "")).strip()
        if not question_text:
            continue
        lines.append(f"Question {index}: {question_text}")
        question_type = str(question.get("type", "question")).strip()
        if question_type:
            lines.append(f"Type: {question_type}")
        options = question.get("options") or []
        if options:
            lines.append("Options: " + ", ".join(str(option) for option in options))
        if question.get("section_title"):
            lines.append(f"Section Title: {question.get('section_title')}")
    return "\n".join(lines).strip()


def _normalize_question_type(kind: str) -> str:
    mapping = {
        "multiple_choice": "multiple_choice",
        "checkbox": "checkbox",
        "dropdown": "dropdown",
        "short_answer": "short_answer",
        "long_answer": "paragraph",
        "date": "date",
        "time": "time",
        "file_upload": "file_upload",
        "grid": "grid",
        "section_title": "section_title",
        "question": "question",
    }
    return mapping.get(kind, kind or "question")


def _dedupe_list(values: list[Any]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _coalesce(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


_SNAPSHOT_SCRIPT = r"""
() => {
  const normalize = (value) => (value || '').replace(/\s+/g, ' ').trim();
  const isVisible = (element) => {
    if (!element) return false;
    const style = window.getComputedStyle(element);
    return style && style.display !== 'none' && style.visibility !== 'hidden' && style.opacity !== '0';
  };
  const textOf = (element) => normalize(element?.innerText || element?.textContent || '');
  const controlSelector = 'input, textarea, select, [role="radio"], [role="checkbox"], [role="option"], [role="combobox"]';
  const blockSelectors = [
    'div[role="listitem"]',
    'fieldset',
    'div[role="group"]',
  ];

  let blocks = Array.from(document.querySelectorAll(blockSelectors.join(','))).filter(isVisible);
  if (!blocks.length) {
    blocks = Array.from(document.querySelectorAll('div, section')).filter((element) => {
      return isVisible(element) && element.querySelector(controlSelector);
    });
  }

  const titleCandidates = Array.from(document.querySelectorAll('h1, [role="heading"], title'))
    .map((element) => textOf(element))
    .filter(Boolean);
  const pageTitle = titleCandidates[0] || normalize(document.title || '');
  const questions = [];
  let currentSection = '';

  for (const block of blocks) {
    const controls = Array.from(block.querySelectorAll(controlSelector)).filter(isVisible);
    const blockText = textOf(block);
    const lines = blockText.split('\n').map((line) => normalize(line)).filter(Boolean);
    const heading = textOf(block.querySelector('[role="heading"], h1, h2, h3, h4, .freebirdFormviewerComponentsQuestionBaseTitle'));

    const optionTexts = [];
    const rowTexts = [];
    const columnTexts = [];
    const seenOptions = new Set();

    for (const control of controls) {
      const role = normalize(control.getAttribute('role')).toLowerCase();
      const type = normalize(control.getAttribute('type')).toLowerCase();
      const tagName = normalize(control.tagName).toLowerCase();
      const label = normalize(
        control.getAttribute('aria-label') ||
        control.getAttribute('data-value') ||
        control.title ||
        control.closest('label')?.innerText ||
        control.innerText ||
        control.textContent ||
        ''
      );
      if (label && !seenOptions.has(label)) {
        seenOptions.add(label);
        optionTexts.push(label);
      }
      if (tagName === 'input' && type === 'radio') {
        // multiple_choice
      } else if (tagName === 'input' && type === 'checkbox') {
        // checkbox
      } else if (tagName === 'textarea') {
        // long_answer
      }
    }

    const tableHeaders = Array.from(block.querySelectorAll('th, [role="columnheader"], [role="rowheader"]'))
      .map((element) => textOf(element))
      .filter(Boolean);
    if (tableHeaders.length) {
      for (const header of tableHeaders) {
        if (/row/i.test(header)) rowTexts.push(header);
        else columnTexts.push(header);
      }
    }

    let kind = 'question';
    if (!controls.length) {
      kind = 'section_title';
    } else if (controls.some((control) => normalize(control.tagName).toLowerCase() === 'textarea')) {
      kind = 'long_answer';
    } else if (controls.some((control) => normalize(control.getAttribute('type')).toLowerCase() === 'radio' || normalize(control.getAttribute('role')).toLowerCase() === 'radio')) {
      kind = 'multiple_choice';
    } else if (controls.some((control) => normalize(control.getAttribute('type')).toLowerCase() === 'checkbox' || normalize(control.getAttribute('role')).toLowerCase() === 'checkbox')) {
      kind = 'checkbox';
    } else if (controls.some((control) => normalize(control.tagName).toLowerCase() === 'select' || normalize(control.getAttribute('role')).toLowerCase() === 'combobox')) {
      kind = 'dropdown';
    } else if (controls.some((control) => normalize(control.getAttribute('type')).toLowerCase() === 'date')) {
      kind = 'date';
    } else if (controls.some((control) => normalize(control.getAttribute('type')).toLowerCase() === 'time')) {
      kind = 'time';
    } else if (controls.some((control) => normalize(control.getAttribute('type')).toLowerCase() === 'file')) {
      kind = 'file_upload';
    } else if (tableHeaders.length) {
      kind = 'grid';
    }

    let questionText = heading;
    if (!questionText) {
      const ignored = new Set(optionTexts.map((option) => option.toLowerCase()));
      for (const line of lines) {
        const lower = line.toLowerCase();
        if (!line) continue;
        if (ignored.has(lower)) continue;
        if (/^(short answer|paragraph|multiple choice|checkboxes|drop-down|dropdown|linear scale|date|time|file upload)$/i.test(line)) continue;
        if (/^required$/i.test(line)) continue;
        if (/^(answer|your answer|optional)$/i.test(line)) continue;
        questionText = line;
        break;
      }
    }

    if (!questionText) {
      continue;
    }

    if (kind === 'section_title' || (!controls.length && questionText.length <= 160)) {
      currentSection = questionText;
      questions.push({
        title: questionText,
        kind: 'section_title',
      });
      continue;
    }

    questions.push({
      title: questionText,
      kind,
      options: optionTexts,
      rows: rowTexts,
      columns: columnTexts,
      section: currentSection,
      required: Boolean(block.querySelector('[aria-label*="Required"], [title*="Required"], [data-required="true"]')),
    });
  }

  return {
    source: 'playwright_dom',
    fetch_status: 'rendered_dom',
    page_url: window.location.href,
    title: pageTitle,
    questions,
  };
}
"""
