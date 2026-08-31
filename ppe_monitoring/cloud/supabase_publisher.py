from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from typing import Any

from ..storage.event_repository import EventRecord
from ..privacy import public_camera_name


class SupabasePublisher:
    """Non-blocking Edge-to-Supabase publisher using a server-side key."""

    def __init__(self, config: dict[str, Any]):
        self.enabled = bool(config.get("enabled", False))
        self.url = str(config.get("url", "")).rstrip("/")
        key_env = str(config.get("service_key_env", "SUPABASE_SERVICE_ROLE_KEY"))
        self.key = os.environ.get(key_env, "")
        self.status_interval = float(config.get("status_interval_seconds", 2))
        self.timeout = float(config.get("request_timeout_seconds", 5))
        self.error: str | None = None
        self._last_status = 0.0
        self._queue: queue.Queue[tuple[str, dict[str, Any], bool]] = queue.Queue(maxsize=250)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if self.enabled and (not self.url or not self.key):
            self.error = f"Cloud publishing requires {key_env} and cloud.url"
            self.enabled = False

    @property
    def status(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "connected": self.enabled and self.error is None, "error": self.error}

    def start(self) -> None:
        if not self.enabled or (self._thread and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="supabase-publisher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.timeout + 1)

    def publish_event(self, event: EventRecord) -> None:
        payload = asdict(event)
        payload.pop("evidence_path", None)
        # Keep compatibility with the deployed table until its optional role
        # columns are migrated. Live status still contains per-person roles.
        payload.pop("role", None)
        payload.pop("helmet_color", None)
        payload["camera"] = public_camera_name(payload["camera"])
        self._enqueue("ppe_events", payload, False)

    def publish_status(self, payload: dict[str, Any]) -> None:
        now = time.monotonic()
        if now - self._last_status < self.status_interval:
            return
        self._last_status = now
        payload = dict(payload)
        if "camera" in payload:
            payload["camera"] = public_camera_name(payload["camera"])
        self._enqueue("ppe_monitor_status?on_conflict=camera", payload, True)

    def _enqueue(self, resource: str, payload: dict[str, Any], upsert: bool) -> None:
        if not self.enabled:
            return
        try:
            self._queue.put_nowait((resource, payload, upsert))
        except queue.Full:
            self.error = "Cloud publisher queue is full; dropping telemetry"

    def _run(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                resource, payload, upsert = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self._post(resource, payload, upsert)
                self.error = None
            except Exception as exc:
                self.error = str(exc)
            finally:
                self._queue.task_done()

    def _post(self, resource: str, payload: dict[str, Any], upsert: bool) -> None:
        headers = {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates,return=minimal" if upsert else "return=minimal",
        }
        request = urllib.request.Request(
            f"{self.url}/rest/v1/{resource}", data=json.dumps(payload).encode("utf-8"),
            headers=headers, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if response.status not in (200, 201, 204):
                    raise RuntimeError(f"Supabase returned HTTP {response.status}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Supabase publish failed ({exc.code}): {detail}") from exc
