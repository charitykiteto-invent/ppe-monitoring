import sys
from types import SimpleNamespace

from ppe_monitoring.detector import HelmetFallbackDetector, canonical_class


def test_common_class_aliases():
    assert canonical_class("Safety-Helmet") == "helmet"
    assert canonical_class("safety vest") == "vest"
    assert canonical_class("reflective-jacket") == "vest"
    assert canonical_class("worker") == "person"
    assert canonical_class("NO-Hardhat") == "no_helmet"
    assert canonical_class("NO-Safety Vest") == "no_vest"
    assert canonical_class("Vest") == "vest"
    assert canonical_class("helmet") == "helmet"


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def int(self):
        return self

    def tolist(self):
        return self.value


def test_helmet_fallback_keeps_only_hardhat_classes(monkeypatch):
    class FakeYOLO:
        names = {0: "Hardhat", 1: "NO-Hardhat", 2: "goggles"}

        def __init__(self, _path):
            pass

        def predict(self, *_args, **_kwargs):
            boxes = SimpleNamespace(
                xyxy=FakeTensor([[10, 10, 30, 30], [40, 10, 60, 30], [70, 10, 90, 30]]),
                conf=FakeTensor([0.91, 0.82, 0.99]),
                cls=FakeTensor([0, 1, 2]),
            )
            return [SimpleNamespace(boxes=boxes, names=self.names)]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    detector = HelmetFallbackDetector("fallback.pt", 0.25)
    result = detector.predict(object())

    assert [item.class_name for item in result["helmet"]] == ["fallback:Hardhat"]
    assert [item.class_name for item in result["no_helmet"]] == ["fallback:NO-Hardhat"]
