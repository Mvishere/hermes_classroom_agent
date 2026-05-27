from pathlib import Path

from storage.state_manager import StateManager


def test_state_manager_marks_and_tracks_items(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    manager = StateManager(state_path)

    assert manager.is_seen("assignment", "a1", "c1") is False
    assert manager.mark_seen("assignment", "a1", "c1") is True
    assert manager.is_seen("assignment", "a1", "c1") is True
    assert manager.mark_seen("assignment", "a1", "c1") is False

    manager.mark_hermes_processed("material", "m1", "c1")
    assert manager.is_seen("material", "m1", "c1") is True
    assert state_path.exists()
