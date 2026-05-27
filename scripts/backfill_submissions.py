"""Backfill student submission states for all stored assignments.

Runs as a one-off script. Updates `data/assignments/assignments.json` and
`data/state/user_status.json` if submission states change.
"""
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config
from auth.google_auth import get_classroom_service
from classroom.classroom_client import ClassroomClient
from storage.json_store import JsonStore
from storage.user_status import UserStatusManager


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
    config.ensure_directories()

    classroom_service = get_classroom_service()
    client = ClassroomClient(classroom_service)
    json_store = JsonStore(config.DATA_DIRECTORY)
    user_status = UserStatusManager(config.USER_STATUS_PATH)

    assignments = json_store.get_all_items("assignments")
    materials = json_store.get_all_items("materials")
    changed = []

    for a in assignments:
        course_id = a.get("course_id")
        item_id = a.get("id")
        if not course_id or not item_id:
            continue
        try:
            subs = client.list_student_submissions(course_id, item_id, user_id="me")
            state = subs[0].get("state", "") if subs else ""
            if state and state != a.get("submission_state", ""):
                a["submission_state"] = state
                json_store.upsert_item("assignments", a)
                changed.append((course_id, item_id, a.get("title", ""), state))
        except Exception:
            logging.exception("Failed to fetch submissions for %s in course %s", item_id, course_id)

    # Re-run status update to mark known/completed items
    updates = user_status.update_from_assignments(json_store.get_all_items("assignments"), materials)

    print("Backfill complete. Changed assignments: %d. User status updates: %d" % (len(changed), updates))
    if changed:
        print("Changed items:")
        for c in changed:
            print(f" - course={c[0]} id={c[1]} title={c[2]!r} state={c[3]}")


if __name__ == "__main__":
    main()
