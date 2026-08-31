from __future__ import annotations

import csv
import time
from datetime import datetime, timezone
from pathlib import Path

from .compliance import ComplianceResult


class EventLogger:
    HEADER = ("timestamp", "tracking_id", "helmet_worn", "vest_worn", "status", "detection_confidence", "camera_source")

    def __init__(self, path: str | Path, camera_source: str, periodic_seconds: float = 60.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.camera_source = camera_source
        self.periodic_seconds = periodic_seconds
        self.last: dict[int, tuple[tuple[bool, bool], float]] = {}
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(self.HEADER)

    def maybe_log(self, track_id: int, result: ComplianceResult, confidence: float) -> bool:
        now = time.monotonic()
        state = (result.helmet_worn, result.vest_worn)
        previous = self.last.get(track_id)
        if previous and previous[0] == state and now - previous[1] < self.periodic_seconds:
            return False
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow((
                datetime.now(timezone.utc).isoformat(), track_id,
                result.helmet_worn, result.vest_worn, result.status,
                f"{confidence:.3f}", self.camera_source,
            ))
        self.last[track_id] = (state, now)
        return True

