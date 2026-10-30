"""Corruptions shared by every experiment.

- ``random_noise``: gaussian jitter on every feature.
- ``random_poison``: a random share of single features copied from another
  clean row. These features are scattered across many sensors.
- ``sensor_lie``: whole sensors (all 8 features of a block) copied from
  another clean row. This is what "one sensor lies" actually looks like.
  With ``base_gas`` and ``donor_gas`` the donor row is the same gas, which
  makes the lie quieter.
- ``column_swap``: copy a fixed number of columns, used by the oven.
"""

from __future__ import annotations

import numpy as np

NOISE_SCALES = (0.10, 0.25, 0.50, 1.00)
POISON_FRACTIONS = (0.05, 0.10, 0.25)
SENSOR_COUNTS = (1, 2, 4, 6)


def random_noise(base: np.ndarray, std: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    shock = rng.normal(0.0, scale, size=base.shape) * std
    return base + shock


def column_swap(
    base: np.ndarray,
    donors: np.ndarray,
    n_columns: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Replace ``n_columns`` random columns of each row with a random donor row's values."""
    out = base.copy()
    n_rows, n_features = out.shape
    n_columns = int(min(max(1, n_columns), n_features))
    donor_ids = rng.integers(0, donors.shape[0], size=n_rows)
    for i in range(n_rows):
        columns = rng.choice(n_features, size=n_columns, replace=False)
        out[i, columns] = donors[donor_ids[i], columns]
    return out


def random_poison(
    base: np.ndarray,
    donors: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Replace a random share of single features with values from a random clean row."""
    n_swap = max(1, int(round(fraction * base.shape[1])))
    return column_swap(base, donors, n_swap, rng)


def sensor_lie(
    base: np.ndarray,
    donors: np.ndarray,
    n_sensors: int,
    block_size: int,
    rng: np.random.Generator,
    base_gas: np.ndarray | None = None,
    donor_gas: np.ndarray | None = None,
) -> np.ndarray:
    """Replace ``n_sensors`` whole sensors of each row with the same sensors of a donor row.

    If ``base_gas`` and ``donor_gas`` are given, each row's donor is drawn from
    rows of the same gas, so the fake values are typical for what the array is
    smelling right now. Rows whose gas has no donor fall back to any donor.
    """
    out = base.copy()
    n_rows, n_features = out.shape
    n_blocks = n_features // block_size
    if not 1 <= n_sensors <= n_blocks:
        raise ValueError(f"n_sensors must be in 1..{n_blocks}")
    offsets = np.arange(block_size)

    same_gas = base_gas is not None and donor_gas is not None
    pools: dict[int, np.ndarray] = {}
    if same_gas:
        for gas in np.unique(donor_gas):
            pools[int(gas)] = np.flatnonzero(donor_gas == gas)

    for i in range(n_rows):
        pool = pools.get(int(base_gas[i])) if same_gas else None
        if pool is not None and len(pool):
            donor = int(pool[rng.integers(0, len(pool))])
        else:
            donor = int(rng.integers(0, donors.shape[0]))
        sensors = rng.choice(n_blocks, size=n_sensors, replace=False)
        columns = (sensors[:, None] * block_size + offsets).ravel()
        out[i, columns] = donors[donor, columns]
    return out
