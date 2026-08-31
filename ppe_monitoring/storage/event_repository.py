from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..privacy import public_camera_name


@dataclass(frozen=True, slots=True)
class EventRecord:
    timestamp: str
    camera: str
    tracking_id: int
    helmet_worn: bool
    vest_worn: bool
    status: str
    reason: str
    confidence: float
    helmet_confidence: float | None = None
    vest_confidence: float | None = None
    evidence_path: str | None = None
    role: str | None = None
    helmet_color: str | None = None


class EventRepository:
    """Thread-safe SQLite event store with per-track state/cooldown suppression."""

    def __init__(self, path: str | Path, cooldown_seconds: float = 10):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cooldown_seconds = cooldown_seconds
        self._lock = threading.RLock()
        self._last: dict[tuple[str, int], tuple[tuple[bool, bool], float]] = {}
        self._create_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _create_schema(self) -> None:
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL, camera TEXT NOT NULL,
                    tracking_id INTEGER NOT NULL, helmet_worn INTEGER NOT NULL,
                    vest_worn INTEGER NOT NULL, status TEXT NOT NULL,
                    reason TEXT NOT NULL, confidence REAL NOT NULL,
                    helmet_confidence REAL, vest_confidence REAL, evidence_path TEXT,
                    role TEXT, helmet_color TEXT
                )
            """)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(events)")}
            if "role" not in columns:
                connection.execute("ALTER TABLE events ADD COLUMN role TEXT")
            if "helmet_color" not in columns:
                connection.execute("ALTER TABLE events ADD COLUMN helmet_color TEXT")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_events_status ON events(status)")
            for row in connection.execute("SELECT DISTINCT camera FROM events"):
                safe = public_camera_name(row[0])
                if safe != row[0]:
                    connection.execute("UPDATE events SET camera = ? WHERE camera = ?", (safe, row[0]))

    def record(self, event: EventRecord, *, force: bool = False) -> bool:
        key = (event.camera, event.tracking_id)
        state = (event.helmet_worn, event.vest_worn)
        now = time.monotonic()
        with self._lock:
            previous = self._last.get(key)
            if not force and previous and previous[0] == state and now - previous[1] < self.cooldown_seconds:
                return False
            values = asdict(event)
            with self._connect() as connection:
                connection.execute("""
                    INSERT INTO events (
                        timestamp, camera, tracking_id, helmet_worn, vest_worn,
                        status, reason, confidence, helmet_confidence,
                        vest_confidence, evidence_path, role, helmet_color
                    ) VALUES (
                        :timestamp, :camera, :tracking_id, :helmet_worn,
                        :vest_worn, :status, :reason, :confidence,
                        :helmet_confidence, :vest_confidence, :evidence_path,
                        :role, :helmet_color
                    )
                """, values)
            self._last[key] = (state, now)
            return True

    def should_record(self, camera: str, tracking_id: int, helmet_worn: bool, vest_worn: bool) -> bool:
        with self._lock:
            previous = self._last.get((camera, tracking_id))
            return not previous or previous[0] != (helmet_worn, vest_worn) or time.monotonic() - previous[1] >= self.cooldown_seconds

    def list_events(
        self, *, status: str | None = None, start: str | None = None,
        end: str | None = None, limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses, values = [], []
        if status:
            clauses.append("status = ?")
            values.append(status)
        if start:
            clauses.append("timestamp >= ?")
            values.append(start)
        if end:
            clauses.append("timestamp <= ?")
            values.append(end)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        query = f"SELECT * FROM events{where} ORDER BY timestamp DESC LIMIT ?"
        values.append(max(1, min(int(limit), 1000)))
        with self._lock, self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return [self._row(row) for row in rows]

    def analytics(self) -> dict[str, Any]:
        today = datetime.now(timezone.utc).date().isoformat()
        with self._lock, self._connect() as connection:
            violations = connection.execute(
                "SELECT reason, COUNT(*) count FROM events WHERE timestamp >= ? AND status != 'COMPLIANT' GROUP BY reason",
                (today,),
            ).fetchall()
            hourly = connection.execute("""
                SELECT substr(timestamp, 12, 2) hour,
                    SUM(CASE WHEN status = 'COMPLIANT' THEN 1 ELSE 0 END) compliant,
                    SUM(CASE WHEN status != 'COMPLIANT' THEN 1 ELSE 0 END) non_compliant
                FROM events WHERE timestamp >= ? GROUP BY hour ORDER BY hour
            """, (today,)).fetchall()
            timeline = connection.execute("""
                SELECT substr(timestamp, 1, 16) minute,
                    ROUND(100.0 * SUM(CASE WHEN status = 'COMPLIANT' THEN 1 ELSE 0 END) / COUNT(*), 1) rate
                FROM events WHERE timestamp >= ? GROUP BY minute ORDER BY minute DESC LIMIT 60
            """, (today,)).fetchall()
            total_violations = connection.execute(
                "SELECT COUNT(*) FROM events WHERE timestamp >= ? AND status != 'COMPLIANT'", (today,)
            ).fetchone()[0]
            roles = connection.execute("""
                SELECT COALESCE(role, 'Unassigned') role, COUNT(*) count
                FROM events WHERE timestamp >= ? AND helmet_worn = 1
                GROUP BY COALESCE(role, 'Unassigned') ORDER BY count DESC
            """, (today,)).fetchall()
        return {
            "violations_by_type": [dict(row) for row in violations],
            "hourly": [dict(row) for row in hourly],
            "compliance_timeline": [dict(row) for row in reversed(timeline)],
            "total_violations_today": total_violations,
            "roles": [dict(row) for row in roles],
        }

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["helmet_worn"] = bool(item["helmet_worn"])
        item["vest_worn"] = bool(item["vest_worn"])
        return item
