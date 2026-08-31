from __future__ import annotations

from typing import Any

from ..association import Detection


def normalize_class_name(name: str) -> str:
    return "_".join(name.lower().strip().replace("-", " ").split())


ALIASES = {
    "helmet": {"helmet", "hardhat", "hard_hat", "safety_helmet"},
    "vest": {"vest", "safety_vest", "reflective_jacket", "safety_jacket", "hi_vis_vest"},
    "person": {"person", "worker", "human"},
    "no_helmet": {"no_hardhat", "no_helmet", "without_helmet", "no_helmet"},
    "no_vest": {"no_safety_vest", "no_vest", "without_vest"},
}


def canonical_class(name: str) -> str | None:
    normalized = normalize_class_name(name)
    return next((kind for kind, names in ALIASES.items() if normalized in names), None)


class PPEDetector:
    def __init__(self, model_path: str, confidence: float, *, device: Any = None, iou: float = 0.5):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed") from exc
        except Exception as exc:
            raise RuntimeError(f"Unable to load PPE weights {model_path}: {exc}") from exc
        self.confidence, self.device, self.iou = confidence, device, iou
        names = self.model.names.values() if isinstance(self.model.names, dict) else self.model.names
        canonical = {canonical_class(str(name)) for name in names}
        if not {"helmet", "vest"}.issubset(canonical):
            shown = list(self.model.names.values()) if isinstance(self.model.names, dict) else list(self.model.names)
            raise RuntimeError(f"PPE model lacks helmet/vest classes: {shown}")

    def predict(self, frame: Any) -> dict[str, list[Detection]]:
        try:
            result = self.model.predict(frame, conf=self.confidence, iou=self.iou, device=self.device, verbose=False)[0]
        except Exception as exc:
            raise RuntimeError(f"PPE inference failed: {exc}") from exc
        grouped: dict[str, list[Detection]] = {key: [] for key in ALIASES}
        if result.boxes is None:
            return grouped
        for box, confidence, class_id in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.int().cpu().tolist()):
            raw_name = str(result.names[class_id])
            kind = canonical_class(raw_name)
            if kind:
                grouped[kind].append(Detection(tuple(map(float, box)), float(confidence), raw_name))
        return grouped


class HelmetFallbackDetector:
    """Helmet-only secondary detector; primary PPE inference remains authoritative."""

    def __init__(self, model_path: str, confidence: float, *, device: Any = None, iou: float = 0.45):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed") from exc
        except Exception as exc:
            raise RuntimeError(f"Unable to load helmet fallback weights {model_path}: {exc}") from exc
        self.confidence, self.device, self.iou = confidence, device, iou
        names = self.model.names.values() if isinstance(self.model.names, dict) else self.model.names
        canonical = {canonical_class(str(name)) for name in names}
        if "helmet" not in canonical:
            shown = list(self.model.names.values()) if isinstance(self.model.names, dict) else list(self.model.names)
            raise RuntimeError(f"Helmet fallback model lacks a helmet/Hardhat class: {shown}")

    def predict(self, frame: Any) -> dict[str, list[Detection]]:
        try:
            result = self.model.predict(
                frame, conf=self.confidence, iou=self.iou,
                device=self.device, verbose=False,
            )[0]
        except Exception as exc:
            raise RuntimeError(f"Helmet fallback inference failed: {exc}") from exc
        grouped: dict[str, list[Detection]] = {"helmet": [], "no_helmet": []}
        if result.boxes is None:
            return grouped
        rows = zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
            result.boxes.cls.int().cpu().tolist(),
        )
        for box, confidence, class_id in rows:
            raw_name = str(result.names[class_id])
            kind = canonical_class(raw_name)
            if kind in grouped:
                grouped[kind].append(
                    Detection(tuple(map(float, box)), float(confidence), f"fallback:{raw_name}")
                )
        return grouped
