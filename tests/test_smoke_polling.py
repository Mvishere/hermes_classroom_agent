from pathlib import Path

import config
from main import run_polling_cycle
from storage.file_storage import FileStorage
from storage.json_store import JsonStore
from storage.state_manager import StateManager
from storage.user_status import UserStatusManager


class DummyClient:
    def list_courses(self):
        return [{"id": "c1", "name": "Course One"}]

    def list_coursework(self, course_id: str):
        return [
            {
                "id": "a1",
                "title": "Assignment",
                "description": "Do it",
                "creationTime": "2025-01-01",
                "updateTime": "2025-01-02",
            }
        ]

    def list_coursework_materials(self, course_id: str):
        return [
            {
                "id": "m1",
                "title": "Material",
                "description": "Read it",
                "creationTime": "2025-01-01",
                "updateTime": "2025-01-02",
            }
        ]

    def list_announcements(self, course_id: str):
        return [
            {
                "id": "n1",
                "text": "Announcement",
                "creationTime": "2025-01-01",
                "updateTime": "2025-01-02",
            }
        ]


class DummyRagPipeline:
    def process_new_materials(self, json_store, material_ids):
        return 0


def test_run_polling_cycle_smoke(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "FETCH_SUBMISSIONS", False)

    data_dir = tmp_path / "data"
    download_dir = tmp_path / "downloads"
    json_store = JsonStore(data_dir)
    state_manager = StateManager(tmp_path / "state.json")
    user_status = UserStatusManager(tmp_path / "user_status.json")
    file_storage = FileStorage(tmp_path, download_dir, data_dir)

    run_polling_cycle(
        DummyClient(),
        json_store,
        state_manager,
        file_storage,
        drive_service=None,
        rag_pipeline=DummyRagPipeline(),
        user_status=user_status,
    )

    assert len(json_store.get_all_items("assignments")) == 1
    assert len(json_store.get_all_items("materials")) == 1
    assert len(json_store.get_all_items("announcements")) == 1
    assert state_manager.is_seen("assignment", "a1", "c1") is True
    assert state_manager.is_seen("material", "m1", "c1") is True
    assert state_manager.is_seen("announcement", "n1", "c1") is True
