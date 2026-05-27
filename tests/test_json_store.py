from pathlib import Path

from storage.json_store import JsonStore


def test_json_store_upsert_and_query(tmp_path: Path) -> None:
    store = JsonStore(tmp_path)
    store.upsert_courses([
        {"id": "c1", "name": "Course One", "updateTime": "2025-01-01"},
        {"id": "c2", "name": "Course Two", "updateTime": "2025-01-02"},
    ])

    record = {
        "id": "m1",
        "course_id": "c1",
        "course_name": "Course One",
        "title": "Material",
        "description": "Details",
    }
    store.upsert_item("materials", record)

    items = store.get_all_items("materials")
    assert len(items) == 1
    assert items[0]["id"] == "m1"

    record["title"] = "Updated"
    store.upsert_item("materials", record)
    items = store.get_all_items("materials")
    assert items[0]["title"] == "Updated"

    subset = store.get_items_by_ids("materials", ["m1", "missing"])
    assert len(subset) == 1
    assert subset[0]["id"] == "m1"
