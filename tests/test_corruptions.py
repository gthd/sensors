import numpy as np

from corruptions import random_poison, sensor_lie


def test_sensor_lie_replaces_whole_sensors_from_one_donor():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(50, 128))
    donors = rng.normal(size=(30, 128)) + 100.0
    out = sensor_lie(base, donors, 2, 8, rng)
    changed = (out != base).reshape(50, 16, 8)
    assert np.all(changed.all(axis=2) == changed.any(axis=2))  # whole sensors only
    assert np.all(changed.all(axis=2).sum(axis=1) == 2)
    for row in out:
        hits = [np.flatnonzero(np.isin(row, d)) for d in donors]
        assert max(len(h) for h in hits) == 16  # one donor supplies both sensors


def test_same_gas_donor():
    rng = np.random.default_rng(4)
    base = np.zeros((40, 128))
    donors = np.repeat(np.arange(1, 5, dtype=float)[:, None], 128, axis=1).repeat(10, axis=0)
    donor_gas = np.repeat(np.arange(1, 5), 10)
    base_gas = rng.integers(1, 5, size=40)
    out = sensor_lie(base, donors, 1, 8, rng, base_gas=base_gas, donor_gas=donor_gas)
    assert np.all(out.max(axis=1) == base_gas)


def test_random_poison_column_count():
    rng = np.random.default_rng(5)
    base = rng.normal(size=(20, 128))
    out = random_poison(base, rng.normal(size=(10, 128)), 0.10, rng)
    assert np.all((out != base).sum(axis=1) == 13)
