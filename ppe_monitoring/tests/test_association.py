from ppe_monitoring.association import Detection, PersonDetection, associate_ppe
from ppe_monitoring.compliance import evaluate


PERSON = PersonDetection((0, 0, 100, 200), 0.95, 1)
HELMET = Detection((25, 2, 75, 35), 0.90, "hardhat")
VEST = Detection((15, 55, 85, 140), 0.88, "safety_vest")


def associated(helmets=(), vests=(), people=(PERSON,)):
    return associate_ppe(people, helmets, vests, helmet_overlap=0.3, vest_overlap=0.4)


def test_helmet_and_vest_correctly_worn():
    result = associated([HELMET], [VEST])[0]
    assert evaluate(result.helmet_worn, result.vest_worn).category == "compliant"


def test_helmet_worn_but_vest_missing():
    result = associated([HELMET], [])[0]
    assert evaluate(result.helmet_worn, result.vest_worn).category == "vest_missing"


def test_vest_worn_but_helmet_missing():
    result = associated([], [VEST])[0]
    assert evaluate(result.helmet_worn, result.vest_worn).category == "helmet_missing"


def test_both_missing():
    result = associated([], [])[0]
    assert evaluate(result.helmet_worn, result.vest_worn).category == "both_missing"


def test_helmet_held_in_hand_is_rejected():
    held_at_hand = Detection((70, 80, 100, 115), 0.98, "helmet")
    result = associated([held_at_hand], [])[0]
    assert not result.helmet_worn


def test_vest_held_beside_body_is_rejected():
    beside_body = Detection((90, 55, 145, 135), 0.98, "vest")
    result = associated([], [beside_body])[0]
    assert not result.vest_worn


def test_helmet_on_floor_is_rejected():
    floor = Detection((20, 170, 75, 198), 0.99, "helmet")
    result = associated([floor], [])[0]
    assert not result.helmet_worn


def test_ppe_is_assigned_only_to_nearby_second_person():
    first = PersonDetection((0, 0, 100, 200), 0.9, 10)
    second = PersonDetection((80, 0, 180, 200), 0.9, 11)
    second_helmet = Detection((120, 3, 170, 34), 0.9, "helmet")
    results = associated([second_helmet], [], [first, second])
    assert not results[0].helmet_worn
    assert results[1].helmet_worn


def test_one_item_cannot_be_shared_by_two_people():
    first = PersonDetection((0, 0, 100, 200), 0.9, 10)
    second = PersonDetection((50, 0, 150, 200), 0.9, 11)
    one_helmet = Detection((55, 3, 95, 34), 0.9, "helmet")
    results = associated([one_helmet], [], [first, second])
    assert sum(result.helmet_worn for result in results) == 1


def test_multiple_people_have_independent_combinations():
    first = PersonDetection((0, 0, 100, 200), 0.9, 10)
    second = PersonDetection((150, 0, 250, 200), 0.9, 11)
    first_helmet = Detection((25, 2, 75, 35), 0.9, "helmet")
    second_vest = Detection((165, 55, 235, 140), 0.9, "vest")
    results = associated([first_helmet], [second_vest], [first, second])
    assert (results[0].helmet_worn, results[0].vest_worn) == (True, False)
    assert (results[1].helmet_worn, results[1].vest_worn) == (False, True)


def test_stronger_spatially_valid_negative_class_is_additional_evidence():
    no_helmet = Detection((25, 2, 75, 35), 0.95, "NO-Hardhat")
    result = associate_ppe([PERSON], [HELMET], [], no_helmets=[no_helmet])[0]
    assert not result.helmet_worn
