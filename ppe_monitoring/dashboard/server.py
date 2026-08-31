from __future__ import annotations

import asyncio
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..storage.event_repository import EventRepository


DASHBOARD_DIR = Path(__file__).resolve().parent


class DashboardState:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._frame: bytes | None = None
        self._frame_number = 0
        self._data: dict[str, Any] = {
            "running": False, "camera_connected": False,
            "model_status": "not loaded", "error": None, "fps": 0.0,
            "people": [], "summary": {
                "total": 0, "compliant": 0, "helmet_missing": 0,
                "vest_missing": 0, "both_missing": 0, "compliance_rate": 0.0,
            },
            "overall": {"state": "OFF", "message": "No person detected"},
            "arduino": {"enabled": False, "connected": False, "port": None, "led_state": "OFF", "error": None},
        }

    def update(self, **values: Any) -> None:
        with self._lock:
            self._data.update(values)

    def set_frame(self, jpeg: bytes) -> None:
        with self._condition:
            self._frame = jpeg
            self._frame_number += 1
            self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._data)

    def wait_for_frame(self, last_number: int, timeout: float = 2) -> tuple[int, bytes | None]:
        with self._condition:
            if self._frame_number == last_number:
                self._condition.wait(timeout)
            return self._frame_number, self._frame


def create_app(
    state: DashboardState, repository: EventRepository,
    start_monitoring: Callable[[], bool] | None = None,
    stop_monitoring: Callable[[], bool] | None = None,
) -> FastAPI:
    app = FastAPI(title="PPE Monitoring", version="1.0.0")
    app.mount("/static", StaticFiles(directory=DASHBOARD_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=DASHBOARD_DIR / "templates")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request=request, name="index.html", context={})

    @app.get("/analytics", response_class=HTMLResponse)
    async def analytics_page(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(request=request, name="analytics.html", context={})

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        data = state.snapshot()
        data["analytics"] = repository.analytics()
        return data

    @app.get("/api/events")
    async def events(
        status_filter: str | None = Query(None, alias="status"),
        start: str | None = None, end: str | None = None, limit: int = 100,
    ) -> dict[str, Any]:
        return {"events": repository.list_events(status=status_filter, start=start, end=end, limit=limit)}

    @app.get("/api/analytics")
    async def analytics() -> dict[str, Any]:
        return repository.analytics()

    @app.post("/api/monitoring/start")
    async def start() -> JSONResponse:
        accepted = start_monitoring() if start_monitoring else False
        return JSONResponse({"accepted": accepted, "running": state.snapshot()["running"]}, status_code=202 if accepted else 409)

    @app.post("/api/monitoring/stop")
    async def stop() -> JSONResponse:
        accepted = stop_monitoring() if stop_monitoring else False
        return JSONResponse({"accepted": accepted, "running": state.snapshot()["running"]}, status_code=202 if accepted else 409)

    def frames():
        number = -1
        while True:
            number, frame = state.wait_for_frame(number)
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

    @app.get("/video-feed")
    async def video_feed() -> StreamingResponse:
        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws")
    async def websocket(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                data = state.snapshot()
                data["analytics"] = repository.analytics()
                data["events"] = repository.list_events(limit=30)
                await websocket.send_json(data)
                await asyncio.sleep(0.75)
        except (WebSocketDisconnect, RuntimeError):
            return

    return app
