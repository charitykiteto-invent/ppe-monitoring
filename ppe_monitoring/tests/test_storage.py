from datetime import datetime, timezone

from ppe_monitoring.storage.event_repository import EventRecord, EventRepository


def event(status="NO PPE"):
    return EventRecord(datetime.now(timezone.utc).isoformat(), "0", 1, False, False, status, "Missing", 0.8)


def test_sqlite_event_cooldown_and_filters(tmp_path):
    repository = EventRepository(tmp_path / "events.db", cooldown_seconds=60)
    assert repository.record(event())
    assert not repository.record(event())
    assert len(repository.list_events(status="NO PPE")) == 1
    assert repository.analytics()["total_violations_today"] == 1

