import cv2
import numpy as np

from ppe_monitoring.helmet_roles import HelmetRoleSmoother, classify_helmet_role


CONFIG = {
    "enabled": True,
    "min_color_fraction": 0.08,
    "supervisor": {"label": "Supervisor", "helmet_color": "yellow", "hsv_lower": [15, 80, 80], "hsv_upper": [40, 255, 255]},
    "worker": {"label": "Worker", "helmet_color": "blue", "hsv_lower": [90, 70, 50], "hsv_upper": [135, 255, 255]},
}


def hsv_frame(hue):
    hsv = np.full((40, 60, 3), (hue, 220, 220), dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_yellow_helmet_is_supervisor_and_blue_is_worker():
    yellow = classify_helmet_role(hsv_frame(28), (0, 0, 60, 40), CONFIG)
    blue = classify_helmet_role(hsv_frame(110), (0, 0, 60, 40), CONFIG)
    assert (yellow.role, yellow.color) == ("Supervisor", "yellow")
    assert (blue.role, blue.color) == ("Worker", "blue")


def test_role_requires_multiple_frames_before_confirmation():
    observation = classify_helmet_role(hsv_frame(28), (0, 0, 60, 40), CONFIG)
    smoother = HelmetRoleSmoother(window=5, min_frames=3)
    assert smoother.update([(1, observation)])[1].role is None
    assert smoother.update([(1, observation)])[1].role is None
    assert smoother.update([(1, observation)])[1].role == "Supervisor"
