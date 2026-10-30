"""Catch rate when the corruptions are random, not aimed at the alarm.

The corruptions themselves live in ``corruptions.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from corruptions import NOISE_SCALES, POISON_FRACTIONS, SENSOR_COUNTS, random_noise, random_poison, sensor_lie
from data_loader import load_nominal_dataset, split_nominal
from monitor import BlockMahalanobisMonitor, FullMahalanobis

SEED = 42
N_SAMPLES = 500


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
        variants.append(("random noise", f"{scale:.2f} std", random_noise(base, std, scale, rng)))
    for fraction in POISON_FRACTIONS:
        variants.append(("feature lie", f"{int(fraction * 100)}% features", random_poison(base, x_train, fraction, rng)))
    for count in SENSOR_COUNTS:
        variants.append(("sensor lie", f"{count} sensors", sensor_lie(base, x_train, count, blocks.block_size, rng)))

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
