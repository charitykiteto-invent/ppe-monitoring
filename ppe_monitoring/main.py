from __future__ import annotations

import argparse
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import yaml

from .association import Detection, PersonPPE, associate_ppe, center, overlap_fraction
from .compliance import PENDING, ComplianceResult, evaluate, led_state
from .cloud.supabase_publisher import SupabasePublisher
from .dashboard.server import DashboardState, create_app
from .detector import YoloDetector
from .hardware.arduino_controller import ArduinoController, MockArduinoController
from .storage.event_repository import EventRecord, EventRepository
from .tracker import IoUTracker, TemporalSmoother


PROJECT_DIR = Path(__file__).resolve().parent


def parse_source(value: Any) -> int | str:
    text = str(value)
    return int(text) if text.isdigit() else text


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    try:
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Cannot read configuration {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise RuntimeError(f"Configuration must be a YAML mapping: {config_path}")
    base = config_path.parent
    paths = (
        ("models", "ppe_model"), ("models", "pose_model"),
        ("models", "helmet_fallback_model"),
        ("dashboard", "video_path"), ("events", "sqlite_path"),
        ("events", "evidence_dir"),
    )
    for section, key in paths:
        value = config.get(section, {}).get(key)
        if value and not Path(str(value)).is_absolute():
            config[section][key] = str((base / str(value)).resolve())
    return config


def open_capture(source: int | str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open camera/video source: {source}")
    return capture


def _box(box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    return tuple(map(int, box))  # type: ignore[return-value]


def draw_detection(frame: Any, detection: Detection, color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = _box(detection.box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    cv2.putText(frame, f"{label} {detection.confidence:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def draw_label(frame: Any, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x, y = origin
    cv2.rectangle(frame, (x, y - height - baseline - 5), (x + width + 7, y + 2), color, -1)
    cv2.putText(frame, text, (x + 3, y - baseline - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def _person_label(result: ComplianceResult) -> str:
    if result.category == "helmet_missing":
        return "HELMET MISSING"
    if result.category == "vest_missing":
        return "VEST MISSING"
    if result.category == "both_missing":
        return "HELMET + VEST MISSING"
    return "COMPLIANT"


def annotate(
    frame: Any, associations: list[PersonPPE], smoothed: dict[int, Any],
    *, fps: float, show_ppe: bool, debug: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[tuple[PersonPPE, ComplianceResult]]]:
    counts = {"compliant": 0, "helmet_missing": 0, "vest_missing": 0, "both_missing": 0}
    people_data: list[dict[str, Any]] = []
    confirmed: list[tuple[PersonPPE, ComplianceResult]] = []
    for item in associations:
        track_id = int(item.person.track_id or -1)
        state = smoothed[track_id]
        if state.confirmed:
            result = evaluate(state.helmet_worn, state.vest_worn)
            color, label, reason = result.color, _person_label(result), result.reason
            counts[result.category] += 1
            confirmed.append((item, result))
            color_name = "green" if result.category == "compliant" else "red" if result.category == "both_missing" else "blue"
        else:
            color, label, reason, color_name = PENDING, "ANALYZING", "Collecting observations", "pending"
        x1, y1, x2, y2 = _box(item.person.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_label(frame, f"Person {track_id} - {label}", (x1, max(20, y1)), color)
        people_data.append({"tracking_id": track_id, "label": label, "reason": reason, "color": color_name, "confidence": round(item.confidence, 3)})
        if show_ppe:
            if item.helmet:
                draw_detection(frame, item.helmet, (255, 190, 0), "helmet")
            if item.vest:
                draw_detection(frame, item.vest, (190, 80, 255), "vest")
        if debug:
            hx1, hy1, hx2, hy2 = _box(item.regions.head)
            tx1, ty1, tx2, ty2 = _box(item.regions.torso)
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 255, 0), 1)
            cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (255, 0, 255), 1)
            for name, (px, py, confidence) in item.person.keypoints.items():
                if confidence >= 0.4:
                    cv2.circle(frame, (int(px), int(py)), 2, (0, 220, 255), -1)
            for detection, region, kind in ((item.helmet, item.regions.head, "H"), (item.vest, item.regions.torso, "V")):
                if detection:
                    dc, rc = center(detection.box), center(region)
                    cv2.line(frame, tuple(map(int, dc)), tuple(map(int, rc)), (255, 255, 255), 1)
                    cv2.putText(frame, f"{kind} overlap {overlap_fraction(detection.box, region):.2f} conf {detection.confidence:.2f}", (x1, min(y2 - 6, int(dc[1]) + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)
    confirmed_count = sum(counts.values())
    summary = {
        "total": len(associations), **counts,
        "compliance_rate": round(100 * counts["compliant"] / confirmed_count, 1) if confirmed_count else 0.0,
    }
    lines = [f"FPS {fps:.1f}", f"People {len(associations)}", f"Compliant {counts['compliant']}", f"Helmet missing {counts['helmet_missing']}", f"Vest missing {counts['vest_missing']}", f"Both missing {counts['both_missing']}"]
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (205, 20 + len(lines) * 20), (10, 17, 25), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (12, 24 + index * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return summary, people_data, confirmed


class MonitoringEngine:
    def __init__(self, config: dict[str, Any], state: DashboardState, repository: EventRepository, arduino: Any, publisher: SupabasePublisher, source_override: str | None = None):
        self.config, self.state, self.repository, self.arduino, self.publisher = config, state, repository, arduino, publisher
        self.source = parse_source(source_override if source_override is not None else config["camera"]["source"])
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="ppe-inference", daemon=True)
            self._thread.start()
            return True

    def stop(self) -> bool:
        if not self._thread or not self._thread.is_alive():
            return False
        self._stop.set()
        return True

    def join(self, timeout: float = 5) -> None:
        if self._thread:
            self._thread.join(timeout)

    def _writer(self, capture: cv2.VideoCapture) -> cv2.VideoWriter | None:
        if not self.config["dashboard"]["save_video"]:
            return None
        path = Path(self.config["dashboard"]["video_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        fps = capture.get(cv2.CAP_PROP_FPS) or 20.0
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        if not writer.isOpened():
            raise RuntimeError(f"Unable to create output video: {path}")
        return writer

    def _overall(self, results: list[ComplianceResult], people_count: int) -> dict[str, str]:
        state = led_state(results)
        if not people_count:
            return {"state": "OFF", "message": "No person detected"}
        if not results:
            return {"state": "OFF", "message": "Analyzing visible people"}
        if state == "RED":
            return {"state": state, "message": "At least one person is missing helmet and safety vest"}
        if state == "BLUE":
            return {"state": state, "message": "At least one person is missing one PPE item"}
        return {"state": state, "message": "All visible confirmed people are compliant"}

    def _record(self, frame: Any, item: PersonPPE, result: ComplianceResult) -> None:
        camera = str(self.source)
        track_id = int(item.person.track_id)
        if not self.repository.should_record(camera, track_id, result.helmet_worn, result.vest_worn):
            return
        evidence_path = None
        if self.config["events"]["save_evidence"]:
            directory = Path(self.config["events"]["evidence_dir"])
            directory.mkdir(parents=True, exist_ok=True)
            filename = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')}_person_{track_id}.jpg"
            target = directory / filename
            if cv2.imwrite(str(target), frame):
                evidence_path = str(target)
        event = EventRecord(
            datetime.now(timezone.utc).isoformat(), camera, track_id,
            result.helmet_worn, result.vest_worn, result.status, result.reason,
            item.confidence, item.helmet.confidence if item.helmet else None,
            item.vest.confidence if item.vest else None, evidence_path,
        )
        self.repository.record(event, force=True)
        self.publisher.publish_event(event)

    def _run(self) -> None:
        capture = None
        writer = None
        self.state.update(running=True, model_status="loading", error=None)
        try:
            detector = YoloDetector(self.config)
            self.state.update(model_status="ready")
            capture = open_capture(self.source)
            self.state.update(camera_connected=True)
            writer = self._writer(capture)
            detection, tracking = self.config["detection"], self.config["tracking"]
            fallback = IoUTracker(tracking["iou_threshold"], tracking["lost_track_timeout"])
            smoother = TemporalSmoother(tracking["history_frames"], tracking["confirmation_frames"], tracking["lost_track_timeout"])
            previous, fps = time.perf_counter(), 0.0
            failed = 0
            while not self._stop.is_set():
                ok, frame = capture.read()
                if not ok:
                    is_live = isinstance(self.source, int) or str(self.source).lower().startswith(("rtsp://", "http://", "https://"))
                    if not is_live:
                        break
                    failed += 1
                    self.state.update(camera_connected=False, error="Camera stream disconnected; reconnecting")
                    capture.release()
                    if failed > self.config["camera"]["reconnect_attempts"]:
                        raise RuntimeError(f"Camera remained unavailable after {failed} attempts")
                    if self._stop.wait(self.config["camera"]["reconnect_seconds"]):
                        break
                    capture = open_capture(self.source)
                    self.state.update(camera_connected=True, error=None)
                    continue
                failed = 0
                people, ppe = detector.infer(frame)
                if any(person.track_id is None for person in people):
                    people = fallback.update(people)
                associations = associate_ppe(
                    people, ppe["helmet"], ppe["vest"],
                    head_range=tuple(detection["head_region"]), torso_range=tuple(detection["torso_region"]),
                    helmet_overlap=detection["helmet_head_overlap"], vest_overlap=detection["vest_torso_overlap"],
                    keypoint_confidence=detection["keypoint_confidence"], no_helmets=ppe["no_helmet"], no_vests=ppe["no_vest"],
                )
                smoothed = smoother.update((int(item.person.track_id), item.helmet_worn, item.vest_worn) for item in associations)
                now = time.perf_counter()
                instant = 1 / max(now - previous, 1e-9)
                fps = instant if not fps else .9 * fps + .1 * instant
                previous = now
                summary, people_data, confirmed = annotate(frame, associations, smoothed, fps=fps, show_ppe=detection["show_ppe_boxes"], debug=self.config["dashboard"]["debug_regions"])
                results = [result for _, result in confirmed]
                overall = self._overall(results, len(associations))
                self.arduino.set_state(overall["state"])
                for item, result in confirmed:
                    self._record(frame, item, result)
                self.state.update(fps=fps, people=people_data, summary=summary, overall=overall, arduino=self.arduino.status, error=None)
                self.publisher.publish_status({
                    "camera": str(self.source), "camera_connected": True,
                    "model_status": "ready", "fps": round(fps, 3),
                    "people": people_data, "summary": summary, "overall": overall,
                    "arduino": self.arduino.status,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                })
                encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if encoded:
                    self.state.set_frame(buffer.tobytes())
                if writer:
                    writer.write(frame)
                if self.config["dashboard"].get("native_window"):
                    cv2.imshow("PPE Monitoring", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        self._stop.set()
        except Exception as exc:
            self.state.update(error=str(exc), camera_connected=False)
        finally:
            self.arduino.set_state("OFF")
            if capture:
                capture.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            self.state.update(running=False, camera_connected=False, fps=0.0, arduino=self.arduino.status)


class PPESystem:
    def __init__(self, config: dict[str, Any], source: str | None = None, mock_arduino: bool = False):
        self.config = config
        self.state = DashboardState()
        self.repository = EventRepository(config["events"]["sqlite_path"], config["events"]["cooldown_seconds"])
        self.arduino = MockArduinoController() if mock_arduino else ArduinoController(config["arduino"])
        self.publisher = SupabasePublisher(config.get("cloud", {}))
        self.engine = MonitoringEngine(config, self.state, self.repository, self.arduino, self.publisher, source)

    def start(self) -> bool:
        self.arduino.start()
        self.publisher.start()
        accepted = self.engine.start()
        self.state.update(arduino=self.arduino.status)
        return accepted

    def stop(self) -> bool:
        return self.engine.stop()

    def close(self) -> None:
        self.engine.stop()
        self.engine.join()
        self.arduino.stop()
        self.publisher.stop()
        self.state.update(arduino=self.arduino.status)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Edge AI PPE Monitoring System")
    parser.add_argument("--config", default=str(PROJECT_DIR / "config.yaml"))
    parser.add_argument("--source", help="Camera index, video path, or RTSP URL")
    parser.add_argument("--host", help="Dashboard bind host")
    parser.add_argument("--port", type=int, help="Dashboard bind port")
    parser.add_argument("--mock-arduino", action="store_true", help="Use the in-memory Arduino for testing")
    parser.add_argument("--no-auto-start", action="store_true", help="Open dashboard without starting inference")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        import uvicorn
        config = load_config(args.config)
        system = PPESystem(config, args.source, args.mock_arduino)
        app = create_app(system.state, system.repository, system.start, system.stop)
        if config["dashboard"].get("auto_start", True) and not args.no_auto_start:
            system.start()
        try:
            uvicorn.run(app, host=args.host or config["dashboard"]["host"], port=args.port or config["dashboard"]["port"], log_level="info")
        finally:
            system.close()
        return 0
    except ImportError as exc:
        print(f"PPE monitoring dependency missing: {exc}. Install requirements.txt", file=sys.stderr)
        return 2
    except (RuntimeError, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"PPE monitoring error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
