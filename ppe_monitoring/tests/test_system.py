from pathlib import Path

import pytest

from ppe_monitoring.compliance import evaluate, led_state
from ppe_monitoring.detector import YoloDetector
from ppe_monitoring.hardware.arduino_controller import ArduinoController, MockArduinoController


def test_requested_three_state_mapping():
    assert (evaluate(True, True).status, evaluate(True, True).reason) == ("COMPLIANT", "Helmet and vest correctly worn")
    assert evaluate(True, False).status == "PARTIAL PPE"
    assert evaluate(False, True).status == "PARTIAL PPE"
    assert evaluate(False, False).status == "NO PPE"


def test_arduino_led_mapping_and_worst_status():
    assert led_state([]) == "OFF"
    assert led_state([evaluate(True, True)]) == "GREEN"
    assert led_state([evaluate(True, True), evaluate(False, True)]) == "BLUE"
    assert led_state([evaluate(True, True), evaluate(False, True), evaluate(False, False)]) == "RED"


class FakeSerial:
    def __init__(self, **kwargs):
        self.command = ""
        self.closed = False

    def reset_input_buffer(self):
        pass

    def write(self, payload):
        self.command = payload.decode().strip()

    def flush(self):
        pass

    def readline(self):
        return ("PONG" if self.command == "PING" else f"ACK:{self.command}").encode() + b"\n"

    def close(self):
        self.closed = True


def test_serial_command_and_acknowledgement_handling():
    controller = ArduinoController({"port": "COM3"}, serial_factory=FakeSerial)
    assert controller.send_and_validate("RED") == "ACK:RED"
    assert controller.send_and_validate("PING") == "PONG"
    controller.stop()


def test_invalid_serial_acknowledgement_is_rejected():
    class BadSerial(FakeSerial):
        def readline(self):
            return b"WRONG\n"

    controller = ArduinoController({"port": "/dev/ttyACM0"}, serial_factory=BadSerial)
    with pytest.raises(RuntimeError, match="expected"):
        controller.send_and_validate("GREEN")
    controller.stop()


def test_missing_arduino_mock_stays_non_fatal_and_turns_off():
    controller = MockArduinoController()
    controller.start()
    controller.set_state("RED")
    controller.stop()
    assert controller.status["led_state"] == "OFF"
    assert not controller.status["connected"]


def test_missing_model_error_is_clear(tmp_path: Path):
    missing = tmp_path / "models" / "ppe_model.pt"
    with pytest.raises(FileNotFoundError, match="PPE model not found"):
        YoloDetector._validate_model_reference(str(missing), "PPE")

