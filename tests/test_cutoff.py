import numpy as np

from run_neighbor_check import _correct, _tune_cutoff


def _counter(frames):
    count = frames[:, 0].astype(int)
    return count > 0, count


def test_cutoff_separates_few_from_many():
    base = np.zeros((100, 1))
    groups = [("noise", np.full((100, 1), 12.0)), ("noise", np.full((100, 1), 9.0)), ("lie", np.full((100, 1), 2.0))]
    cutoff = _tune_cutoff(base, groups, _counter, 16)
    assert 3 <= cutoff <= 9
    eligible = ~_counter(base)[0]
    assert _correct(eligible, groups[0][1], _counter, cutoff, "noise") == 1.0
    assert _correct(eligible, groups[2][1], _counter, cutoff, "lie") == 1.0


def test_zero_count_alarm_is_never_a_correct_lie():
    base = np.zeros((10, 1))
    eligible = ~_counter(base)[0]
    assert _correct(eligible, np.zeros((10, 1)), _counter, 5, "lie") == 0.0
