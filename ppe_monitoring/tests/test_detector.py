from ppe_monitoring.detector import canonical_class


def test_common_class_aliases():
    assert canonical_class("Safety-Helmet") == "helmet"
    assert canonical_class("safety vest") == "vest"
    assert canonical_class("reflective-jacket") == "vest"
    assert canonical_class("worker") == "person"
    assert canonical_class("NO-Hardhat") == "no_helmet"
    assert canonical_class("NO-Safety Vest") == "no_vest"
    assert canonical_class("Vest") == "vest"
    assert canonical_class("helmet") == "helmet"
