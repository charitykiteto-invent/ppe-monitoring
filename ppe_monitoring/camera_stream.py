from __future__ import annotations

import os
import threading
import time
from typing import Any

import cv2

from .privacy import public_camera_name


def is_network_stream(source: int | str) -> bool:
    return isinstance(source, str) and source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://"))


class LatestFrameCapture:
    """Continuously drains a live stream and exposes only its newest frame."""

    def __init__(self, capture: Any, read_timeout: float = 10.0):
        self._capture = capture
        self._read_timeout = max(0.25, float(read_timeout))
        self._condition = threading.Condition()
        self._frame: Any = None
        self._sequence = 0
        self._delivered = 0
        self._failed = False
        self._stopped = False
        self._thread = threading.Thread(target=self._drain, name="camera-latest-frame", daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        while not self._stopped:
            ok, frame = self._capture.read()
            with self._condition:
                if not ok:
                    self._failed = True
                    self._condition.notify_all()
                    return
                self._frame = frame
                self._sequence += 1
                self._condition.notify_all()

    def read(self) -> tuple[bool, Any]:
        deadline = time.monotonic() + self._read_timeout
        with self._condition:
            while self._sequence == self._delivered and not self._failed and not self._stopped:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, None
                self._condition.wait(remaining)
            if self._sequence == self._delivered:
                return False, None
            self._delivered = self._sequence
            return True, self._frame

    def get(self, prop: int) -> float:
        return self._capture.get(prop)

    def isOpened(self) -> bool:  # OpenCV-compatible spelling
        return not self._failed and not self._stopped and self._capture.isOpened()

    def release(self) -> None:
        self._stopped = True
        self._capture.release()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)


def open_capture(source: int | str, config: dict[str, Any] | None = None) -> Any:
    settings = config or {}
    network = is_network_stream(source)
    if network and settings.get("low_latency", True):
        transport = str(settings.get("rtsp_transport", "tcp"))
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            f"rtsp_transport;{transport}|fflags;nobuffer|flags;low_delay",
        )
    backend = cv2.CAP_FFMPEG if network else cv2.CAP_ANY
    capture = cv2.VideoCapture(source, backend)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open camera/video source: {public_camera_name(source)}")
    if network and settings.get("low_latency", True):
        return LatestFrameCapture(capture, settings.get("read_timeout_seconds", 10))
    return capture
