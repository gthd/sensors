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


def test_line_sits_between_tied_values():
    scores = np.repeat([0.0, 1.0, 4.0, 9.0], [700, 250, 40, 10]).astype(float)
    line = empirical_line(scores, 0.02)
    assert 4.0 < line < 9.0
    assert np.mean(scores > line) == 0.01
    assert np.mean(scores + 1e-6 > line) == 0.01  # a tiny nudge does not cross it


def test_line_above_the_largest_tied_value():
    scores = np.repeat([0.0, 1.0, 4.0], [900, 90, 10]).astype(float)
    line = empirical_line(scores, 0.005)
    assert 4.0 < line < 9.0
    assert np.mean(scores + 1e-6 > line) == 0.0
