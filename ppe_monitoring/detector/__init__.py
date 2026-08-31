from __future__ import annotations

from pathlib import Path
from typing import Any

from .ppe_detector import ALIASES, PPEDetector, canonical_class, normalize_class_name
from .pose_detector import PoseDetector


class YoloDetector:
    """Facade combining independent PPE and pose inference adapters."""

    def __init__(self, config: dict[str, Any]):
        models = config["models"]
        detection = config["detection"]
        ppe_path = str(models.get("ppe_model", ""))
        pose_path = str(models.get("pose_model", ""))
        backend = str(models.get("inference_backend", "auto")).lower()
        if backend not in {"pytorch", "onnx", "tensorrt", "auto"}:
            raise ValueError(f"Unsupported models.inference_backend: {backend}")
        if not ppe_path or not pose_path:
            raise FileNotFoundError("Both models.ppe_model and models.pose_model must be configured")
        self._validate_model_reference(ppe_path, "PPE")
        self._validate_model_reference(pose_path, "person/pose")
        options = {"device": models.get("device") or None, "iou": float(detection.get("nms_iou", 0.5))}
        self.ppe = PPEDetector(ppe_path, float(detection["ppe_confidence"]), **options)
        self.pose = PoseDetector(pose_path, float(detection["person_confidence"]), models.get("tracker", "bytetrack.yaml"), **options)

    @staticmethod
    def _validate_model_reference(reference: str, label: str) -> None:
        if not Path(reference).is_file():
            raise FileNotFoundError(f"{label} model not found: {reference}. Run: python -m ppe_monitoring.scripts.download_models")

    def infer(self, frame: Any):
        return self.pose.track(frame), self.ppe.predict(frame)


__all__ = ["ALIASES", "PPEDetector", "PoseDetector", "YoloDetector", "canonical_class", "normalize_class_name"]
