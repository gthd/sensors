"""Catch rate when the corruptions are random, not aimed at the alarm.

Also the shared corruption library used by every other script:

- ``_random_noise``: gaussian jitter on every feature.
- ``_random_poison``: a random share of single features copied from another
  clean row. These features are scattered across many sensors.
- ``_sensor_lie``: whole sensors (all 8 features of a block) copied from
  another clean row. This is what "one sensor lies" actually looks like.
  With ``same_gas`` the donor row is the same gas, which makes the lie quieter.
- ``_column_swap``: copy a fixed number of columns, used by the oven.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_loader import load_nominal_dataset, split_nominal
from monitor import BlockMahalanobisMonitor
from run_full_vs_blocks import FullMahalanobis

SEED = 42
N_SAMPLES = 500
NOISE_SCALES = (0.10, 0.25, 0.50, 1.00)
POISON_FRACTIONS = (0.05, 0.10, 0.25)
SENSOR_COUNTS = (1, 2, 4, 6)


def _random_noise(base: np.ndarray, std: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    shock = rng.normal(0.0, scale, size=base.shape) * std
    return base + shock


def _column_swap(
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


def _random_poison(
    base: np.ndarray,
    donors: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Replace a random share of single features with values from a random clean row."""
    n_swap = max(1, int(round(fraction * base.shape[1])))
    return _column_swap(base, donors, n_swap, rng)


def _sensor_lie(
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


def _catch(alarm_fn, base: np.ndarray, corrupted: np.ndarray) -> tuple[int, int, float]:
    clean = np.array([not alarm_fn(row) for row in base])
    hit = np.array([alarm_fn(row) for row in corrupted])
    eligible = int(clean.sum())
    caught = int(np.logical_and(hit, clean).sum())
    rate = float(caught / eligible) if eligible else float("nan")
    return caught, eligible, rate


def main() -> None:
    features, gas_ids = load_nominal_dataset()
    x_train, x_test = split_nominal(features, gas_ids, seed=SEED)
    blocks = BlockMahalanobisMonitor().fit(x_train)
    full = FullMahalanobis().fit(x_train)
    std = np.maximum(blocks.feature_std_, 1e-8)

    rng = np.random.default_rng(SEED)
    base = x_test[rng.choice(x_test.shape[0], size=N_SAMPLES, replace=False)]
    block_alarm = lambda frame: bool(np.any(blocks.block_distances(frame) > blocks.tau))
    full_alarm = full.alarms

    variants: list[tuple[str, str, np.ndarray]] = []
    for scale in NOISE_SCALES:
        variants.append(("random noise", f"{scale:.2f} std", _random_noise(base, std, scale, rng)))
    for fraction in POISON_FRACTIONS:
        variants.append(("feature lie", f"{int(fraction * 100)}% features", _random_poison(base, x_train, fraction, rng)))
    for count in SENSOR_COUNTS:
        variants.append(("sensor lie", f"{count} sensors", _sensor_lie(base, x_train, count, blocks.block_size, rng)))

    rows: list[dict[str, object]] = []
    for kind, size, corrupted in variants:
        b_caught, b_of, b_rate = _catch(block_alarm, base, corrupted)
        f_caught, f_of, f_rate = _catch(full_alarm, base, corrupted)
        rows.append(
            {
                "kind": kind,
                "size": size,
                "blocks": f"{b_caught}/{b_of}",
                "blocks rate": b_rate,
                "all-128": f"{f_caught}/{f_of}",
                "all-128 rate": f_rate,
            }
        )

    table = pd.DataFrame(rows)
    print("Random corruptions only. Nothing is pushed toward the alarm.")
    print(f"Rows: {N_SAMPLES}. Noise = gaussian shock. Lies = features or whole sensors copied from another clean row.")
    print()
    print(
        table.to_string(
            index=False,
            formatters={"blocks rate": "{:.4f}".format, "all-128 rate": "{:.4f}".format},
        )
    )


if __name__ == "__main__":
    main()
