"""Summarization tool: returns stored summaries or generates light summaries from item text."""
from typing import Optional

from storage.json_store import JsonStore
from storage.summary_store import SummaryStore


def handle(question: str, data_dir: str) -> Optional[str]:
    q = question.lower()
    if "summarize" not in q and "summary" not in q:
        return None

    js = JsonStore(data_dir)
    ss = SummaryStore(js.data_dir / "rag" / "summaries.json")

    # If user asks to summarize assignments, return summaries of all assignments
    if "assignment" in q:
        assigns = js.get_all_items("assignments")
        if not assigns:
            return "No assignments to summarize."
        lines = []
        for a in assigns:
            s = ss.get_summary(a.get("course_id", ""), a.get("id", ""))
            title = a.get("title", "Untitled")
            if s:
                lines.append(f"{title}: {s}")
            else:
                desc = a.get("description", "")
                lines.append(f"{title}: {desc or 'No summary available.'}")
        return "\n".join(lines)

    # Default: return most recent summary across materials
    materials = js.get_all_items("materials")
    if not materials:
        return "No materials to summarize."
    latest = sorted(materials, key=lambda m: m.get("updated_at") or m.get("created_at") or "")[-1]
    s = ss.get_summary(latest.get("course_id", ""), latest.get("id", ""))
    if s:
        return f"Summary for {latest.get('title','Untitled')}: {s}"
    return "No stored summary for the most recent material."
