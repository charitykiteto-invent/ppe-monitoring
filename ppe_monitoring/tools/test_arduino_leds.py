from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ppe_monitoring.hardware.arduino_controller import ArduinoController


def main() -> int:
    parser = argparse.ArgumentParser(description="Safely test PPE Arduino LEDs")
    parser.add_argument("--port", default="auto", help="COM3, /dev/ttyACM0, /dev/ttyUSB0, or auto")
    parser.add_argument("--cycle", action="store_true", help="Cycle OFF, RED, BLUE, GREEN, OFF")
    parser.add_argument("--seconds", type=float, default=1.0, help="Seconds per cycle state")
    args = parser.parse_args()
    controller = ArduinoController({"enabled": True, "port": args.port, "baud_rate": 115200})
    try:
        states = ["OFF", "RED", "BLUE", "GREEN", "OFF"] if args.cycle else ["OFF"]
        for state in states:
            print(f"{state}: {controller.send_and_validate(state)}")
            time.sleep(max(0, args.seconds))
        return 0
    except Exception as exc:
        print(f"Arduino test failed: {exc}", file=sys.stderr)
        return 2
    finally:
        controller.stop()


if __name__ == "__main__":
    raise SystemExit(main())

