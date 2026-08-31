from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


Box = tuple[float, float, float, float]


@dataclass(slots=True)
class Detection:
    box: Box
    confidence: float
    class_name: str


@dataclass(slots=True)
class PersonDetection:
    box: Box
    confidence: float
    track_id: int | None = None
    keypoints: Mapping[str, tuple[float, float, float]] = field(default_factory=dict)


@dataclass(slots=True)
class BodyRegions:
    head: Box
    torso: Box


@dataclass(slots=True)
class PersonPPE:
    person: PersonDetection
    regions: BodyRegions
    helmet: Detection | None = None
    vest: Detection | None = None
    no_helmet: Detection | None = None
    no_vest: Detection | None = None

    @property
    def helmet_worn(self) -> bool:
        return self.helmet is not None and not (
            self.no_helmet is not None and self.no_helmet.confidence > self.helmet.confidence
        )

    @property
    def vest_worn(self) -> bool:
        return self.vest is not None and not (
            self.no_vest is not None and self.no_vest.confidence > self.vest.confidence
        )

    @property
    def confidence(self) -> float:
        values = [self.person.confidence]
        values.extend(d.confidence for d in (self.helmet, self.vest, self.no_helmet, self.no_vest) if d)
        return sum(values) / len(values)


def area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection(a: Box, b: Box) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )


def overlap_fraction(item: Box, region: Box) -> float:
    """Fraction of the PPE item covered by a target body region."""
    return intersection(item, region) / max(area(item), 1e-9)


def center(box: Box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def point_in_box(point: tuple[float, float], box: Box) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def _clamp(region: Box, person: Box, allow_above: float = 0.0) -> Box:
    height = person[3] - person[1]
    return (
        max(person[0], region[0]),
        max(person[1] - allow_above * height, region[1]),
        min(person[2], region[2]),
        min(person[3], region[3]),
    )


def body_regions(
    person: PersonDetection,
    head_range: tuple[float, float] = (0.0, 0.32),
    torso_range: tuple[float, float] = (0.25, 0.75),
    keypoint_confidence: float = 0.35,
) -> BodyRegions:
    """Estimate head/torso from COCO pose points, with proportional fallbacks."""
    x1, y1, x2, y2 = person.box
    width, height = x2 - x1, y2 - y1
    fallback_head = (x1, y1 + head_range[0] * height, x2, y1 + head_range[1] * height)
    fallback_torso = (x1, y1 + torso_range[0] * height, x2, y1 + torso_range[1] * height)
    kp = person.keypoints

    face = [kp[name] for name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear") if name in kp and kp[name][2] >= keypoint_confidence]
    shoulders = [kp[name] for name in ("left_shoulder", "right_shoulder") if name in kp and kp[name][2] >= keypoint_confidence]
    hips = [kp[name] for name in ("left_hip", "right_hip") if name in kp and kp[name][2] >= keypoint_confidence]

    head = fallback_head
    if face and len(shoulders) == 2:
        shoulder_y = sum(p[1] for p in shoulders) / 2
        face_x = [p[0] for p in face]
        top = max(y1 - 0.08 * height, min(p[1] for p in face) - 0.10 * height)
        pad_x = max(0.08 * width, (max(face_x) - min(face_x)) * 0.45)
        head = _clamp((min(face_x) - pad_x, top, max(face_x) + pad_x, shoulder_y), person.box, 0.08)

    torso = fallback_torso
    if len(shoulders) == 2 and len(hips) == 2:
        points = shoulders + hips
        pad = 0.08 * width
        torso = _clamp((min(p[0] for p in points) - pad, min(p[1] for p in shoulders), max(p[0] for p in points) + pad, max(p[1] for p in hips)), person.box)
    return BodyRegions(head=head, torso=torso)


def _expand(box: Box, x_fraction: float, y_fraction: float) -> Box:
    w, h = box[2] - box[0], box[3] - box[1]
    return (box[0] - x_fraction * w, box[1] - y_fraction * h, box[2] + x_fraction * w, box[3] + y_fraction * h)


def _candidate_score(item: Detection, region: Box, kind: str, required_overlap: float) -> float | None:
    expanded = _expand(region, 0.12 if kind == "helmet" else 0.04, 0.25 if kind == "helmet" else 0.03)
    item_center = center(item.box)
    # Center-location is deliberately mandatory: a large held/carried item must not
    # qualify merely because an edge overlaps the body region.
    if not point_in_box(item_center, expanded):
        return None
    overlap = overlap_fraction(item.box, expanded if kind == "helmet" else region)
    if overlap < required_overlap:
        return None
    region_center = center(region)
    rw, rh = max(region[2] - region[0], 1.0), max(region[3] - region[1], 1.0)
    distance = abs(item_center[0] - region_center[0]) / rw + abs(item_center[1] - region_center[1]) / rh
    return 2.0 * overlap + item.confidence - 0.25 * distance


def _assign_one_to_one(
    results: list[PersonPPE], items: Sequence[Detection], region_kind: str,
    target_attribute: str, required_overlap: float,
) -> None:
    candidates: list[tuple[float, int, int]] = []
    for person_index, result in enumerate(results):
        region = result.regions.head if region_kind == "helmet" else result.regions.torso
        for item_index, item in enumerate(items):
            score = _candidate_score(item, region, region_kind, required_overlap)
            if score is not None:
                candidates.append((score, person_index, item_index))
    used_people: set[int] = set()
    used_items: set[int] = set()
    for _, person_index, item_index in sorted(candidates, reverse=True):
        if person_index in used_people or item_index in used_items:
            continue
        setattr(results[person_index], target_attribute, items[item_index])
        used_people.add(person_index)
        used_items.add(item_index)


def associate_ppe(
    people: Sequence[PersonDetection],
    helmets: Sequence[Detection],
    vests: Sequence[Detection],
    *,
    head_range: tuple[float, float] = (0.0, 0.32),
    torso_range: tuple[float, float] = (0.25, 0.75),
    helmet_overlap: float = 0.35,
    vest_overlap: float = 0.45,
    keypoint_confidence: float = 0.35,
    no_helmets: Sequence[Detection] = (),
    no_vests: Sequence[Detection] = (),
) -> list[PersonPPE]:
    results = [PersonPPE(p, body_regions(p, head_range, torso_range, keypoint_confidence)) for p in people]
    _assign_one_to_one(results, helmets, "helmet", "helmet", helmet_overlap)
    _assign_one_to_one(results, vests, "vest", "vest", vest_overlap)
    # Explicit absence classes are secondary evidence only. They must pass the
    # same person-region checks, and suppress a positive only when more confident.
    _assign_one_to_one(results, no_helmets, "helmet", "no_helmet", helmet_overlap)
    _assign_one_to_one(results, no_vests, "vest", "no_vest", vest_overlap)
    return results
