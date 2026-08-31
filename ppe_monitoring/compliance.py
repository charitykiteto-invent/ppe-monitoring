from __future__ import annotations

from dataclasses import dataclass


GREEN = (0, 180, 0)
BLUE = (255, 110, 20)
RED = (0, 0, 255)
PENDING = (170, 170, 170)


@dataclass(frozen=True, slots=True)
class ComplianceResult:
    helmet_worn: bool
    vest_worn: bool
    status: str
    reason: str
    color: tuple[int, int, int]
    category: str


def evaluate(helmet_worn: bool, vest_worn: bool) -> ComplianceResult:
    if helmet_worn and vest_worn:
        return ComplianceResult(True, True, "COMPLIANT", "Helmet and vest correctly worn", GREEN, "compliant")
    if helmet_worn:
        return ComplianceResult(True, False, "PARTIAL PPE", "Safety vest missing", BLUE, "vest_missing")
    if vest_worn:
        return ComplianceResult(False, True, "PARTIAL PPE", "Helmet missing", BLUE, "helmet_missing")
    return ComplianceResult(False, False, "NO PPE", "Helmet and safety vest missing", RED, "both_missing")


def led_state(results: list[ComplianceResult]) -> str:
    """Map current people to the Arduino state; worst confirmed state wins."""
    if not results:
        return "OFF"
    if any(result.category == "both_missing" for result in results):
        return "RED"
    if any(result.category in {"helmet_missing", "vest_missing"} for result in results):
        return "BLUE"
    return "GREEN"
