from __future__ import annotations

import threading
import time
from typing import Any, Callable


COMMANDS = {"RED", "BLUE", "GREEN", "OFF"}


class ArduinoController:
    """Background, reconnecting serial controller. Hardware failure is non-fatal."""

    def __init__(
        self, config: dict[str, Any], *, serial_factory: Callable[..., Any] | None = None,
        port_finder: Callable[[], list[str]] | None = None,
    ):
        self.enabled = bool(config.get("enabled", True))
        self.configured_port = str(config.get("port", "auto"))
        self.baud_rate = int(config.get("baud_rate", 115200))
        self.heartbeat = float(config.get("heartbeat_seconds", 2))
        self.reconnect = float(config.get("reconnect_seconds", 3))
        self.timeout = float(config.get("acknowledgement_timeout_seconds", 1))
        self._serial_factory = serial_factory
        self._port_finder = port_finder
        self._serial: Any = None
        self._desired = "OFF"
        self._sent: str | None = None
        self._connected = False
        self._port: str | None = None
        self._error: str | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled, "connected": self._connected,
            "port": self._port, "led_state": self._sent if self._connected else "OFF",
            "error": self._error,
        }

    def start(self) -> None:
        if not self.enabled or (self._thread and self._thread.is_alive()):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="arduino-controller", daemon=True)
        self._thread.start()

    def set_state(self, state: str) -> None:
        state = state.upper()
        if state not in COMMANDS:
            raise ValueError(f"Unsupported Arduino LED state: {state}")
        if state != self._desired:
            self._desired = state
            self._wake.set()

    def stop(self) -> None:
        self._desired = "OFF"
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=max(2.0, self.timeout + 0.5))
        if self._serial:
            try:
                self._exchange("OFF")
            except Exception:
                pass
        self._disconnect(None)

    def send_and_validate(self, command: str) -> str:
        """Public synchronous primitive used by the hardware test and unit tests."""
        command = command.upper()
        expected = "PONG" if command == "PING" else f"ACK:{command}"
        if command not in COMMANDS | {"PING"}:
            raise ValueError(f"Unsupported serial command: {command}")
        if not self._serial:
            self._connect()
        response = self._exchange(command)
        if response != expected:
            raise RuntimeError(f"Arduino returned {response!r}; expected {expected!r}")
        return response

    def _run(self) -> None:
        next_attempt = 0.0
        next_heartbeat = 0.0
        while not self._stop.is_set():
            now = time.monotonic()
            if not self._serial and now >= next_attempt:
                try:
                    self._connect()
                    self.send_and_validate("OFF")
                    self._sent = "OFF"
                    next_heartbeat = now + self.heartbeat
                except Exception as exc:
                    self._disconnect(str(exc))
                    next_attempt = now + self.reconnect
            if self._serial:
                try:
                    if self._desired != self._sent:
                        self.send_and_validate(self._desired)
                        self._sent = self._desired
                        next_heartbeat = now + self.heartbeat
                    elif now >= next_heartbeat:
                        self.send_and_validate("PING")
                        next_heartbeat = now + self.heartbeat
                except Exception as exc:
                    self._disconnect(str(exc))
                    next_attempt = now + self.reconnect
            self._wake.wait(0.1)
            self._wake.clear()
        if self._serial:
            try:
                self.send_and_validate("OFF")
            except Exception:
                pass

    def _connect(self) -> None:
        factory = self._serial_factory
        if factory is None:
            try:
                import serial
            except ImportError as exc:
                raise RuntimeError("pyserial is not installed") from exc
            factory = serial.Serial
        ports = [self.configured_port] if self.configured_port.lower() != "auto" else self._discover_ports()
        if not ports:
            raise RuntimeError("No Arduino serial port found")
        errors = []
        for port in ports:
            try:
                connection = factory(port=port, baudrate=self.baud_rate, timeout=self.timeout, write_timeout=self.timeout)
                self._serial = connection
                self._port = port
                # Most Arduino boards reset when the port opens.
                time.sleep(1.8 if self._serial_factory is None else 0)
                self._connected = True
                self._error = None
                return
            except Exception as exc:
                errors.append(f"{port}: {exc}")
        raise RuntimeError("; ".join(errors))

    def _discover_ports(self) -> list[str]:
        if self._port_finder:
            return self._port_finder()
        try:
            from serial.tools import list_ports
        except ImportError:
            return []
        entries = list(list_ports.comports())
        preferred = []
        fallback = []
        for item in entries:
            label = f"{item.description} {item.manufacturer or ''}".lower()
            target = preferred if any(word in label for word in ("arduino", "wch", "ch340", "cp210", "usb serial")) else fallback
            target.append(item.device)
        return preferred + fallback

    def _exchange(self, command: str) -> str:
        self._serial.reset_input_buffer()
        self._serial.write((command + "\n").encode("ascii"))
        self._serial.flush()
        raw = self._serial.readline()
        if not raw:
            raise TimeoutError(f"No acknowledgement for {command}")
        return raw.decode("ascii", errors="replace").strip()

    def _disconnect(self, error: str | None) -> None:
        if self._serial:
            try:
                self._serial.close()
            except Exception:
                pass
        self._serial = None
        self._connected = False
        self._port = None
        self._sent = None
        self._error = error


class MockArduinoController:
    def __init__(self) -> None:
        self.led_state = "OFF"
        self.enabled = True
        self.connected = False

    @property
    def status(self) -> dict[str, Any]:
        return {"enabled": True, "connected": self.connected, "port": "MOCK", "led_state": self.led_state, "error": None}

    def start(self) -> None:
        self.connected = True

    def set_state(self, state: str) -> None:
        if state not in COMMANDS:
            raise ValueError(state)
        self.led_state = state

    def stop(self) -> None:
        self.led_state = "OFF"
        self.connected = False

