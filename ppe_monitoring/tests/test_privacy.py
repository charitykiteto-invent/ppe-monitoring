from ppe_monitoring.privacy import public_camera_name
from ppe_monitoring.storage.event_repository import EventRecord, EventRepository


def test_camera_name_removes_credentials_query_and_fragment():
    source = "rtsp://admin:secret@192.168.1.50:554/Preview_01_main?token=hidden#value"
    assert public_camera_name(source) == "rtsp://192.168.1.50:554/Preview_01_main"


def test_existing_camera_credentials_are_scrubbed_from_database(tmp_path):
    path = tmp_path / "events.db"
    repository = EventRepository(path)
    repository.record(EventRecord("2026-08-31T10:00:00+00:00", "rtsp://admin:secret@camera.local/live", 1, True, True, "COMPLIANT", "OK", .9))
    repository = EventRepository(path)
    event = repository.list_events()[0]
    assert event["camera"] == "rtsp://camera.local/live"
    assert "admin" not in event["camera"]
    assert "secret" not in event["camera"]
