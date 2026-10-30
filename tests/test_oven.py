import numpy as np
import pytest

from run_neighbor_check import _tune_cutoff
from timeseries import Surprise, changed, noise, obvious_lie, plausible_lie, stuck, window_starts, windows

N_THERMO = 15


@pytest.fixture(scope="module")
def ar_windows():
    """Windows of 8 minutes whose jumps follow j_t = 0.6 j_(t-1) + e, per thermometer."""
    rng = np.random.default_rng(0)
    n, length = 20000, 8
    jumps = np.zeros((n, length - 1, N_THERMO))
    jumps[:, 0] = rng.normal(size=(n, N_THERMO))
    for t in range(1, length - 1):
        jumps[:, t] = 0.6 * jumps[:, t - 1] + rng.normal(size=(n, N_THERMO))
    start = rng.uniform(100, 400, size=(n, 1, N_THERMO))
    return np.concatenate([start, start + np.cumsum(jumps, axis=1)], axis=1)


def test_window_starts_skip_gaps():
    minute = np.timedelta64(60, "s")
    stamps = np.datetime64("2018-01-01T00:00") + np.array([0, 1, 2, 3, 5, 6, 7, 8]) * minute
    starts = window_starts(stamps, 3, 0, len(stamps), 60.0)
    np.testing.assert_array_equal(starts, [0, 1, 4, 5])
    frames = windows(np.arange(8.0)[:, None].repeat(N_THERMO, axis=1), starts, 3)
    assert frames.shape == (4, 3, N_THERMO)


def test_trend_surprise_recovers_the_persistence(ar_windows):
    trend = Surprise(3).fit(ar_windows)
    assert abs(trend.coef_[:, 1].mean() - 0.6) < 0.02
    assert np.all(np.abs(trend.coef_[:, 2:]) < 0.05)
    jump = Surprise(0).fit(ar_windows)
    assert np.median(trend.residual_std(ar_windows) / jump.residual_std(ar_windows)) < 0.85


def test_surprise_layout_is_thermometer_by_thermometer(ar_windows):
    trend = Surprise(3).fit(ar_windows)
    features = trend(ar_windows[:10], 4)
    assert features.shape == (10, N_THERMO * 4)
    last = trend(ar_windows[:10], 1)
    np.testing.assert_allclose(features.reshape(10, N_THERMO, 4)[:, :, -1], last)


@pytest.mark.parametrize("minutes", [1, 3])
def test_corruptions_touch_only_the_last_minutes(ar_windows, minutes):
    rng = np.random.default_rng(1)
    base = ar_windows[:200]
    donors = ar_windows[1000:2000]
    for corrupted in (
        noise(base, np.ones(N_THERMO), 1.0, minutes, rng),
        obvious_lie(base, donors, 2, minutes, rng),
        plausible_lie(base, donors, 2, minutes, rng),
        stuck(base, 2, minutes, rng),
    ):
        np.testing.assert_array_equal(corrupted[:, :-minutes], base[:, :-minutes])


def test_lies_change_exactly_n_thermometers(ar_windows):
    rng = np.random.default_rng(2)
    base = ar_windows[:200]
    lied = obvious_lie(base, ar_windows[1000:2000], 4, 3, rng)
    assert np.all(np.any(lied != base, axis=1).sum(axis=1) == 4)


def test_plausible_lie_replays_donor_jumps(ar_windows):
    rng = np.random.default_rng(3)
    base = ar_windows[:50]
    lied = plausible_lie(base, ar_windows[1000:2000], N_THERMO, 3, rng)
    assert np.all(lied[:, -4] == base[:, -4])
    lied_jumps = np.diff(lied, axis=1)[:, -3:]
    donor_jumps = np.diff(ar_windows[1000:2000], axis=1)[:, -3:]
    for row in lied_jumps[:5]:
        assert np.any(np.all(np.isclose(donor_jumps, row), axis=(1, 2)))


def test_stuck_freezes_and_unchanged_rows_are_detected(ar_windows):
    base = ar_windows[:20].copy()
    base[:10, -3:, :] = base[:10, -4:-3, :]  # already flat at the end
    frozen = stuck(base, N_THERMO, 3, np.random.default_rng(4))
    assert np.all(frozen[:, -3:] == frozen[:, -4:-3])
    was_changed = changed(base, frozen)
    assert not was_changed[:10].any() and was_changed[10:].all()


def test_tuner_ignores_unchanged_rows():
    def counter(frames):
        count = frames[:, 0].astype(int)
        return count > 0, count

    base = np.zeros((100, 1))
    lie = np.zeros((100, 1))
    lie[:20] = 2.0  # only a fifth of the lie rows changed
    cutoff = _tune_cutoff(base, [("noise", np.full((100, 1), 10.0)), ("lie", lie)], counter, 15)
    assert 3 <= cutoff <= 10
