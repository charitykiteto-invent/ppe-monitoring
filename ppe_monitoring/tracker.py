from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Iterable

from .association import Box, PersonDetection, area, intersection


def iou(a: Box, b: Box) -> float:
    overlap = intersection(a, b)
    return overlap / max(area(a) + area(b) - overlap, 1e-9)


class IoUTracker:
    """Fallback ID tracker used when the inference backend provides no ByteTrack ID."""

    def __init__(self, threshold: float = 0.3, max_age: int = 30):
        self.threshold = threshold
        self.max_age = max_age
        self.next_id = 1
        self.tracks: dict[int, tuple[Box, int]] = {}

    def update(self, people: list[PersonDetection]) -> list[PersonDetection]:
        candidates = sorted(
            ((iou(p.box, box), pi, tid) for pi, p in enumerate(people) for tid, (box, _) in self.tracks.items()),
            reverse=True,
        )
        assigned_people: set[int] = set()
        assigned_tracks: set[int] = set()
        for score, pi, tid in candidates:
            if score < self.threshold or pi in assigned_people or tid in assigned_tracks:
                continue
            people[pi].track_id = tid
            assigned_people.add(pi)
            assigned_tracks.add(tid)
        for person in people:
            if person.track_id is None:
                person.track_id = self.next_id
                self.next_id += 1
            self.tracks[person.track_id] = (person.box, 0)
        visible = {p.track_id for p in people}
        for tid, (box, age) in list(self.tracks.items()):
            if tid not in visible:
                if age + 1 > self.max_age:
                    del self.tracks[tid]
                else:
                    self.tracks[tid] = (box, age + 1)
        return people


@dataclass(slots=True)
class TrackState:
    history: deque[tuple[bool, bool]]
    visible_frames: int = 0
    missed_frames: int = 0
    last_emitted: tuple[bool, bool] | None = None
    last_event_time: float = 0.0


@dataclass(frozen=True, slots=True)
class SmoothedState:
    helmet_worn: bool
    vest_worn: bool
    confirmed: bool


class TemporalSmoother:
    def __init__(self, window: int = 7, min_frames: int = 4, max_missing: int = 30):
        self.window = window
        self.min_frames = min_frames
        self.max_missing = max_missing
        self.states: dict[int, TrackState] = {}

    def update(self, observations: Iterable[tuple[int, bool, bool]]) -> dict[int, SmoothedState]:
        seen: set[int] = set()
        output: dict[int, SmoothedState] = {}
        for track_id, helmet, vest in observations:
            seen.add(track_id)
            state = self.states.setdefault(track_id, TrackState(deque(maxlen=self.window)))
            state.history.append((helmet, vest))
            state.visible_frames += 1
            state.missed_frames = 0
            # Vote on the pair, preserving a real observed state instead of mixing
            # independent votes into a combination that never occurred.
            voted = Counter(state.history).most_common(1)[0][0]
            output[track_id] = SmoothedState(*voted, state.visible_frames >= self.min_frames)
        for track_id, state in list(self.states.items()):
            if track_id not in seen:
                state.missed_frames += 1
                if state.missed_frames > self.max_missing:
                    del self.states[track_id]
        return output

