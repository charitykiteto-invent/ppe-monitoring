from __future__ import annotations

from typing import Any

from ..association import PersonDetection


KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)


class PoseDetector:
    def __init__(self, model_path: str, confidence: float, tracker: str, *, device: Any = None, iou: float = 0.5):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed") from exc
        except Exception as exc:
            raise RuntimeError(f"Unable to load pose weights {model_path}: {exc}") from exc
        if getattr(self.model.model, "kpt_shape", None) is None:
            raise RuntimeError(f"Configured pose checkpoint has no keypoints: {model_path}")
        self.confidence, self.tracker_name, self.device, self.iou = confidence, tracker, device, iou

    def track(self, frame: Any) -> list[PersonDetection]:
        try:
            result = self.model.track(frame, persist=True, tracker=self.tracker_name, conf=self.confidence, iou=self.iou, classes=[0], device=self.device, verbose=False)[0]
        except Exception as exc:
            raise RuntimeError(f"Pose inference failed: {exc}") from exc
        people = []
        if result.boxes is None:
            return people
        boxes, confidences = result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()
        ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else [None] * len(boxes)
        xy = result.keypoints.xy.cpu().tolist() if result.keypoints is not None else [None] * len(boxes)
        cf = result.keypoints.conf.cpu().tolist() if result.keypoints is not None and result.keypoints.conf is not None else [None] * len(boxes)
        for index, (box, confidence, track_id) in enumerate(zip(boxes, confidences, ids)):
            keypoints = {}
            if xy[index] is not None:
                scores = cf[index] if cf[index] is not None else [1.0] * len(xy[index])
                keypoints = {name: (float(point[0]), float(point[1]), float(score)) for name, point, score in zip(KEYPOINT_NAMES, xy[index], scores)}
            people.append(PersonDetection(tuple(map(float, box)), float(confidence), track_id, keypoints))
        return people

