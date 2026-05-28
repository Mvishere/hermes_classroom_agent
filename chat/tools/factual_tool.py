"""Factual question tool: answers direct queries from stored JSON data."""
from typing import Optional
import re
from pathlib import Path

from storage.json_store import JsonStore
from student.knowledge_store import KnowledgeStore
from student.topic_graph import TopicGraph
from storage.user_status import UserStatusManager


def handle(question: str, data_dir: str) -> Optional[str]:
    js = JsonStore(data_dir)
    us = UserStatusManager(js.data_dir / "state" / "user_status.json")
    q = question.lower()

    topic_response = _handle_topic_question(q, data_dir)
    if topic_response:
        return topic_response

    multi_count_response = _handle_multi_count_question(q, js)
    if multi_count_response:
        return multi_count_response

    if "how many" in q and "announcement" in q and "title" not in q and "mention" not in q:
        anns = js.get_all_items("announcements")
        return f"You have {len(anns)} announcement(s) in your Classroom data."

    if "how many" in q and "material" in q and "title" not in q and "mention" not in q:
        materials = js.get_all_items("materials")
        return f"You have {len(materials)} material(s) in your Classroom data."

    if "how many" in q and "assignment" in q and "title" not in q and "pending" not in q and "returned" not in q and "mention" not in q:
        assigns = js.get_all_items("assignments")
        return f"You have {len(assigns)} assignment(s) in your Classroom data."

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


def _handle_multi_count_question(question: str, js: JsonStore) -> Optional[str]:
    mention_response = _handle_mention_count_question(question, js)
    if mention_response:
        return mention_response

    if "pending" in question and "returned" in question and "assignment" in question:
        assignments = js.get_all_items("assignments")
        pending = [
            assignment
            for assignment in assignments
            if assignment.get("submission_state", "").upper() not in {"TURNED_IN", "RETURNED"}
        ]
        returned = [
            assignment
            for assignment in assignments
            if assignment.get("submission_state", "").upper() == "RETURNED"
        ]
        return (
            f"You have {len(pending)} pending assignment(s) and {len(returned)} returned assignment(s)."
        )

    if "stored locally" in question and {"announcement", "material", "assignment"}.issubset(set(_tokenize(question))):
        return (
            f"Stored locally: {len(js.get_all_items('announcements'))} announcement(s), "
            f"{len(js.get_all_items('materials'))} material(s), and {len(js.get_all_items('assignments'))} assignment(s)."
        )

    if "more recent" in question and "latest material" in question and "latest announcement" in question:
        materials = js.get_all_items("materials")
        announcements = js.get_all_items("announcements")
        if not materials and not announcements:
            return "I could not find any materials or announcements in the local data."
        latest_material = _latest_item(materials)
        latest_announcement = _latest_item(announcements)
        if latest_material and latest_announcement:
            if _item_timestamp(latest_material) >= _item_timestamp(latest_announcement):
                return f"The latest material is more recent: {latest_material.get('title', 'Untitled')}."
            return f"The latest announcement is more recent: {latest_announcement.get('title', 'Untitled')}."
        if latest_material:
            return f"The latest material is: {latest_material.get('title', 'Untitled')}."
        return f"The latest announcement is: {latest_announcement.get('title', 'Untitled')}."

    return None


def _handle_mention_count_question(question: str, js: JsonStore) -> Optional[str]:
    if "mention" not in question or "how many" not in question:
        return None

    clauses = [part.strip() for part in re.split(r",| and ", question) if "mention" in part]
    if not clauses:
        clauses = [question]

    results = []
    inherited_item_type = ""
    for clause in clauses:
        item_type = ""
        if "material" in clause:
            item_type = "materials"
        elif "assignment" in clause:
            item_type = "assignments"
        elif "announcement" in clause:
            item_type = "announcements"
        elif inherited_item_type:
            item_type = inherited_item_type

        if not item_type:
            continue
        inherited_item_type = item_type

        match = re.search(r"mention(?:s)?(?:\s+\w+)*\s+(.+)$", clause)
        if not match:
            continue
        term_text = match.group(1)
        terms = [term for term in _tokenize(term_text) if term not in {"how", "many", "and", "or", "the", "a", "an", "of", "for", "to", "with", "in"}]
        if not terms:
            continue

        items = js.get_all_items(item_type)
        count = _count_items_matching_terms(items, terms)
        label = item_type[:-1] if item_type.endswith("s") else item_type
        results.append(f"{count} {label}(s) mention {', '.join(terms)}")

    if not results:
        return None

    return "; ".join(results).capitalize() + "."


def _count_items_matching_terms(items: list[dict], terms: list[str]) -> int:
    count = 0
    for item in items:
        haystack = " ".join(
            str(item.get(field, "")) for field in ("title", "description", "text")
        ).lower()
        if any(term in haystack for term in terms):
            count += 1
    return count


def _handle_topic_question(question: str, data_dir: str) -> Optional[str]:
    if not any(marker in question for marker in ("prerequisite", "prerequisites", "related", "ready", "review", "prepare", "before learning", "before", "seem ready", "likely requires", "most likely requires")):
        return None

    graph = TopicGraph(Path(data_dir) / "topic_graph.json")
    topic = _best_topic_match(question, graph.all_topics())
    if not topic:
        return None

    node = graph.get(topic)
    if not node:
        return None

    if "related" in question:
        related = _format_related(node.get("related_topics", []))
        if related:
            return f"Related to {topic}: {related}."
        return f"I could not find related topics for {topic}."

    if any(marker in question for marker in ("prerequisite", "prerequisites", "before learning", "before", "prepare", "review")):
        prerequisites = node.get("prerequisites", []) or []
        if prerequisites:
            return f"Before {topic}, review: {', '.join(prerequisites)}."
        return f"I could not find prerequisites for {topic}."

    if any(marker in question for marker in ("ready", "seem ready", "likely requires", "most likely requires")):
        prerequisites = node.get("prerequisites", []) or []
        if not prerequisites:
            return f"I do not have enough prerequisite data to judge readiness for {topic}."
        knowledge = KnowledgeStore(Path(data_dir) / "knowledge_state.json")
        missing = []
        weak = []
        known = 0
        for prereq in prerequisites:
            status = (knowledge.get_topic(prereq) or {}).get("status", "unknown")
            if status == "weak":
                weak.append(prereq)
            elif status in {"known", "learning"}:
                known += 1
            else:
                missing.append(prereq)
        if weak or len(missing) > max(1, len(prerequisites) // 2):
            focus = weak or missing
            return f"You are not fully ready for {topic}. Review: {', '.join(focus)}."
        if missing:
            return f"You are partially ready for {topic}. Review: {', '.join(missing)}."
        return f"You look ready for {topic}; you know most prerequisites."

    return None


def _best_topic_match(question: str, topics: list[str]) -> str:
    question_tokens = set(_tokenize(question))
    best_topic = ""
    best_score = 0
    for topic in topics:
        topic_tokens = set(_tokenize(topic))
        score = len(question_tokens & topic_tokens)
        topic_lower = topic.lower()
        if topic_lower in question:
            score += 3
        elif all(token in question for token in topic_lower.split() if len(token) > 2):
            score += 2
        if score > best_score:
            best_score = score
            best_topic = topic
    return best_topic


def _format_related(values: list) -> str:
    related = []
    for value in values or []:
        if isinstance(value, dict):
            topic = value.get("topic")
            weight = value.get("weight")
            if topic:
                related.append(f"{topic} ({weight:.2f})" if isinstance(weight, (int, float)) else str(topic))
        elif value:
            related.append(str(value))
    return ", ".join(related)


def _latest_item(items: list[dict]) -> Optional[dict]:
    if not items:
        return None
    return sorted(items, key=_item_timestamp)[-1]


def _item_timestamp(item: dict) -> str:
    return item.get("updated_at") or item.get("created_at") or ""


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
