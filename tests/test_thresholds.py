import numpy as np
import pytest

from thresholds import empirical_line, empirical_lines


def test_empirical_line_hits_target():
    scores = np.random.default_rng(1).exponential(size=5000)
    line = empirical_line(scores, 0.02)
    assert np.mean(scores > line) <= 0.02
    assert np.mean(scores > line) > 0.015


def test_empirical_lines_joint_rate_at_most_target():
    scores = np.random.default_rng(2).exponential(size=(4000, 16))
    lines = empirical_lines(scores, 0.05)
    assert lines.shape == (16,)
    rate = np.mean(np.any(scores > lines, axis=1))
    assert 0.04 < rate <= 0.05


@pytest.mark.parametrize("target", [0.0, 1.0, -0.1])
def test_target_must_be_inside_unit_interval(target):
    with pytest.raises(ValueError):
        empirical_line(np.ones(10), target)
