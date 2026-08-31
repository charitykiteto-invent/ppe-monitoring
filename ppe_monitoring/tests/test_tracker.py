from ppe_monitoring.tracker import TemporalSmoother


def test_temporary_detection_loss_does_not_flip_majority():
    smoother = TemporalSmoother(window=5, min_frames=2, max_missing=3)
    smoother.update([(7, True, True)])
    assert smoother.update([(7, True, True)])[7].confirmed
    state = smoother.update([(7, False, False)])[7]
    assert (state.helmet_worn, state.vest_worn) == (True, True)


def test_confirmation_is_delayed_and_stale_history_removed():
    smoother = TemporalSmoother(window=3, min_frames=2, max_missing=1)
    assert not smoother.update([(4, True, False)])[4].confirmed
    assert smoother.update([(4, True, False)])[4].confirmed
    smoother.update([])
    smoother.update([])
    assert 4 not in smoother.states

