from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Iterable

import cv2
import numpy as np

from .association import Box


@dataclass(frozen=True, slots=True)
class HelmetRole:
    role: str | None
    color: str | None
    confidence: float


def classify_helmet_role(frame: Any, box: Box, config: dict[str, Any]) -> HelmetRole:
    if not config.get("enabled", True):
        return HelmetRole(None, None, 0.0)
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in box)
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return HelmetRole(None, None, 0.0)
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    scores: list[tuple[float, str, str]] = []
    for name in ("supervisor", "worker"):
        rule = config.get(name, {})
        lower = np.array(rule.get("hsv_lower", [0, 0, 0]), dtype=np.uint8)
        upper = np.array(rule.get("hsv_upper", [179, 255, 255]), dtype=np.uint8)
        fraction = float(np.count_nonzero(cv2.inRange(hsv, lower, upper))) / max(crop.shape[0] * crop.shape[1], 1)
        scores.append((fraction, str(rule.get("label", name.title())), str(rule.get("helmet_color", name))))
    fraction, role, color = max(scores)
    if fraction < float(config.get("min_color_fraction", 0.08)):
        return HelmetRole(None, None, round(fraction, 3))
    return HelmetRole(role, color, round(fraction, 3))


class HelmetRoleSmoother:
    def __init__(self, window: int = 8, min_frames: int = 3, max_missing: int = 30):
        self.window = window
        self.min_frames = min_frames
        self.max_missing = max_missing
        self.history: dict[int, deque[HelmetRole]] = {}
        self.missed: dict[int, int] = {}

    def update(self, observations: Iterable[tuple[int, HelmetRole]]) -> dict[int, HelmetRole]:
        seen: set[int] = set()
        result: dict[int, HelmetRole] = {}
        for track_id, observation in observations:
            seen.add(track_id)
            history = self.history.setdefault(track_id, deque(maxlen=self.window))
            history.append(observation)
            self.missed[track_id] = 0
            known = [item for item in history if item.role]
            if not known:
                result[track_id] = HelmetRole(None, None, observation.confidence)
                continue
            role, count = Counter(item.role for item in known).most_common(1)[0]
            winner = [item for item in known if item.role == role]
            if count < self.min_frames:
                result[track_id] = HelmetRole(None, None, max(item.confidence for item in winner))
            else:
                result[track_id] = HelmetRole(role, winner[-1].color, sum(item.confidence for item in winner) / len(winner))
        for track_id in list(self.history):
            if track_id not in seen:
                self.missed[track_id] = self.missed.get(track_id, 0) + 1
                if self.missed[track_id] > self.max_missing:
                    del self.history[track_id]
                    del self.missed[track_id]
        return result
