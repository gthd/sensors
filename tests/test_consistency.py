import numpy as np
import pytest

from consistency import SensorRemovalCheck
from corruptions import random_noise, sensor_lie


@pytest.fixture(scope="module")
def check(factor_data):
    return SensorRemovalCheck().fit(factor_data[0])


@pytest.mark.parametrize("m", [1, 3, 8, 9, 15])
def test_remainder_matches_leaving_sensors_out(check, factor_data, m):
    test = factor_data[1][:40]
    w, d = check._whiten(test)
    removed = np.random.default_rng(m).permuted(np.tile(np.arange(16), (40, 1)), axis=1)[:, :m]
    z = (test - check.mu_) / check.scale_
    expected = []
    for i in range(40):
        keep = np.setdiff1d(np.arange(16), removed[i])
        cols = (keep[:, None] * 8 + np.arange(8)).ravel()
        expected.append(z[i, cols] @ np.linalg.solve(check.cov_[np.ix_(cols, cols)], z[i, cols]))
    np.testing.assert_allclose(check._remainder(w, d, removed), expected, rtol=1e-8)


def test_conditional_score_is_what_leaving_one_out_removes(check, factor_data):
    test = factor_data[1][:40]
    w, d = check._whiten(test)
    scores = check.conditional_scores(test)
    for k in (0, 7, 15):
        removed = np.full((40, 1), k)
        np.testing.assert_allclose(d - scores[:, k], check._remainder(w, d, removed), rtol=1e-8)


def test_one_lying_sensor_is_found(check, factor_data):
    train, test, _ = factor_data
    clean = check.counts(test) == 0
    assert clean.mean() > 0.95
    rng = np.random.default_rng(6)
    lied = sensor_lie(test, train, 1, 8, rng)
    liar = np.argmax((lied != test).reshape(-1, 16, 8).any(axis=2), axis=1)
    counts, order = check.removal(lied)
    caught = clean & (counts > 0)
    assert caught.mean() > 0.8
    assert np.mean(counts[caught] == 1) > 0.95
    assert np.mean(order[caught, 0] == liar[caught]) > 0.95


def test_noise_needs_many_sensors_set_aside(check, factor_data):
    train, test, scales = factor_data
    noisy = random_noise(test, train.std(axis=0), 0.5, np.random.default_rng(7))
    counts = check.counts(noisy)
    assert np.median(counts) >= 12


def test_empirical_lines(check, factor_data):
    test = factor_data[1]
    lines = check.empirical_lines(test, 0.05)
    assert lines.shape == (17,)
    assert np.all(np.diff(lines[:-1]) <= 0)
    assert np.isinf(lines[-1])
    check.set_lines(lines)
    try:
        assert np.mean(check.counts(test) > 0) <= 0.05
    finally:
        check.set_lines(None)
    np.testing.assert_array_equal(check.lines, check.chi2_lines)
