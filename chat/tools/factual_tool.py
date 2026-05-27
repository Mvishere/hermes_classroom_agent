"""Factual question tool: answers direct queries from stored JSON data."""
from typing import Optional

from storage.json_store import JsonStore
from storage.user_status import UserStatusManager


def handle(question: str, data_dir: str) -> Optional[str]:
    js = JsonStore(data_dir)
    us = UserStatusManager(js.data_dir / "state" / "user_status.json")
    q = question.lower()

    if "what courses" in q or ("course" in q and ("am i" in q or "which" in q)):
        courses = js._load(js.data_dir / "courses" / "courses.json", {"courses": {}})
        names = [c.get("name") for c in (courses.get("courses") or {}).values() if c.get("name")]
        if not names:
            return "I couldn't find any courses in the local data."
        if len(names) == 1:
            return f"You are enrolled in {names[0]}."
        return "You are enrolled in: " + ", ".join(names) + "."

    # handle various spellings/typos of "announcement" by matching on common stem
    if "recent announcement" in q or ("recent" in q and ("announcement" in q or "announ" in q or "annou" in q)):
        anns = js.get_all_items("announcements")
        if not anns:
            return "You have no announcements in the local data."
        latest = sorted(anns, key=lambda a: a.get("updated_at") or a.get("created_at") or "")[-1]
        title = latest.get("title", "Untitled")
        desc = latest.get("description", latest.get("text", ""))
        return f"Most recent announcement: {title}. {desc}"

    # titles for assignments
    if ("title" in q and "assignment" in q) or ("what is the title" in q and "assignment" in q):
        assigns = js.get_all_items("assignments")
        if not assigns:
            return "No assignments found."
        titles = [a.get("title") for a in assigns if a.get("title")]
        if not titles:
            return "No assignment titles found."
        if len(titles) == 1:
            return f"The assignment title is: {titles[0]}"
        return "Assignment titles: " + ", ".join(titles) + "."

    if ("how many" in q and "assignment" in q) or ("pending" in q and "assignment" in q):
        assigns = js.get_all_items("assignments")
        pending = [a for a in assigns if a.get("submission_state", "").upper() not in {"TURNED_IN", "RETURNED"}]
        return f"You have {len(pending)} pending assignment(s)."

    # Titles for announcements (including plural, list requests, and common misspellings)
    if ("title" in q or "tell" in q or "list" in q or "all" in q) and (
        "announcement" in q or "announ" in q or "annou" in q
    ):
        anns = js.get_all_items("announcements")
        if not anns:
            return "You have no announcements in the local data."
        titles = [a.get("title") for a in anns if a.get("title")]
        if not titles:
            return "No announcement titles found."
        return "Announcement titles: " + ", ".join(titles) + "."

    # Quiz / assignment specific title requests
    if ("quiz" in q or "quiz" in question.lower()) and ("title" in q or "what is the title" in q or "recent" in q):
        assigns = js.get_all_items("assignments")
        if not assigns:
            return "No assignments found."
        # Prefer assignments with 'quiz' in the title
        quiz_assigns = [a for a in assigns if "quiz" in (a.get("title", "").lower())]
        target = None
        if quiz_assigns:
            # pick most recent
            target = sorted(quiz_assigns, key=lambda a: a.get("updated_at") or a.get("created_at") or "")[-1]
        else:
            # fallback: most recent assignment
            target = sorted(assigns, key=lambda a: a.get("updated_at") or a.get("created_at") or "")[-1]
        return f"The most recent quiz/assignment title is: {target.get('title','Untitled')}"

    return None
