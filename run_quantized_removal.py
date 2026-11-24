"""Q15 and Q7 removal on the gas-array evaluation split.

The float64 fit, the empirical lines, and the noise-versus-lie cutoff are
unchanged. Only the online kernel is quantized. The three reported rates
match the rest of the gas-array evaluation:

- clean alarm: share of clean evaluation rows the kernel alarms on
- lie caught: share of eligible rows whose one-sensor lie is called a lie
- top-1: among those caught lies, share whose first suspect is the copied sensor

Dot products of the 128 stored integers accumulate in int64.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from consistency import SensorRemovalCheck
from corruptions import NOISE_SCALES, POISON_FRACTIONS, SENSOR_COUNTS, random_noise, random_poison, sensor_lie
from data_loader import load_nominal_dataset, split_random_indices
from monitor import BLOCK_SIZE, BlockMahalanobisMonitor
from quantized import QuantizedRemoval
from run_neighbor_check import _tune_cutoff, removal_counter
from run_workshop_repeats import SEEDS, SPLIT_SEED
from thresholds import DEFAULT_TARGET

FORMATS = (16, 8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET)
    args = parser.parse_args()

    features, gas_ids = load_nominal_dataset()
    train_idx, test_idx = split_random_indices(gas_ids, seed=SPLIT_SEED)
    x_train, x_test = features[train_idx], features[test_idx]
    blocks = BlockMahalanobisMonitor().fit(x_train)
    check = SensorRemovalCheck().fit(x_train)
    std = np.maximum(blocks.feature_std_, 1e-8)
    z_train = (x_train - check.mu_) / check.scale_
    kernels = {bits: QuantizedRemoval(check, bits, float(np.max(np.abs(z_train)))) for bits in FORMATS}

    print(f"Train {len(train_idx)} rows, test pool {len(test_idx)} rows, split seed {SPLIT_SEED}.")
    print(f"Empirical lines at {args.target:.1%} clean alarms. Cutoff tuned as in the gas-array runs.")
    for bits, kernel in kernels.items():
        name = "Q15" if bits == 16 else "Q7"
        print(
            f"{name}: z scale {kernel.z_scale:g}, P scale {kernel.p_scale:g}, "
            f"inverse scale {kernel.inv_scale:g}, P stored as 0: {kernel.p_zero:.3f}, "
            f"inverse stored as 0: {kernel.inv_zero:.3f}"
        )

    rows = []
    for seed in args.seeds:
        print(f"\nSeed {seed}")
        metrics = _one_seed(seed, x_train, x_test, std, blocks, check, kernels, args.target)
        rows.extend(metrics)
        for item in metrics:
            print(
                f"  {item['format']:<8} clean {item['clean alarm']:.3f}  "
                f"lie {item['lie caught']:.3f}  top-1 {item['top-1']:.3f}  "
                f"count=1 {item['count is 1']:.3f}  "
                f"z clipped {item['z clipped']:.3f}  cutoff {item['cutoff']}"
            )

    table = pd.DataFrame(rows)
    summary = table.groupby("format", sort=False)[["clean alarm", "lie caught", "top-1", "count is 1"]].median()
    print("\nMedian over seeds")
    print(summary.to_string(float_format=lambda value: f"{value:.3f}"))
    path = "quantized_removal.csv"
    table.to_csv(path, index=False)
    print(f"\nPer-seed rows: {path}")


def _one_seed(seed, x_train, x_test, std, blocks, check, kernels, target) -> list[dict[str, float | str | int]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(x_test.shape[0])
    mid = x_test.shape[0] // 2
    cal, eva = x_test[order[:mid]], x_test[order[mid:]]
    check.set_lines(check.empirical_lines(cal, target))

    rng_tune = np.random.default_rng(seed + 7)
    groups = [("noise", random_noise(cal, std, scale, rng_tune)) for scale in NOISE_SCALES]
    groups += [("lie", sensor_lie(cal, x_train, n, BLOCK_SIZE, rng_tune)) for n in SENSOR_COUNTS]
    groups += [("lie", random_poison(cal, x_train, fraction, rng_tune)) for fraction in POISON_FRACTIONS]
    counter = removal_counter(check)
    cutoff = _tune_cutoff(cal, groups, counter, blocks.n_blocks)

    rng_eval = np.random.default_rng(seed + 1)
    for scale in NOISE_SCALES:
        random_noise(eva, std, scale, rng_eval)
    for fraction in POISON_FRACTIONS:
        random_poison(eva, x_train, fraction, rng_eval)
    lied, liar = _sensor_lie_tracked(eva, x_train, 1, BLOCK_SIZE, rng_eval)

    clean_counts, _ = check.removal(eva)
    lie_counts, lie_order = check.removal(lied)
    rows = [_score("float64", clean_counts, lie_counts, lie_order, liar, cutoff, z_clipped=0.0)]
    for bits, kernel in kernels.items():
        name = "Q15" if bits == 16 else "Q7"
        _z_q, clipped = kernel.quantize_z(eva)
        q_clean, _ = kernel.removal(eva)
        q_counts, q_order = kernel.removal(lied)
        rows.append(_score(name, q_clean, q_counts, q_order, liar, cutoff, z_clipped=clipped))
        del _z_q
    for row in rows:
        row["seed"] = seed
        row["cutoff"] = cutoff
    return rows


def _score(name, clean_counts, lie_counts, lie_order, liar, cutoff, z_clipped: float) -> dict[str, float | str]:
    eligible = clean_counts == 0
    caught = eligible & (lie_counts > 0) & (lie_counts < cutoff)
    n_eligible = int(eligible.sum())
    lie_rate = float(caught.sum() / n_eligible) if n_eligible else float("nan")
    if caught.any():
        top1 = float(np.mean(lie_order[caught, 0] == liar[caught]))
        exactly_one = float(np.mean(lie_counts[caught] == 1))
    else:
        top1 = float("nan")
        exactly_one = float("nan")
    return {
        "format": name,
        "clean alarm": float(np.mean(clean_counts > 0)),
        "lie caught": lie_rate,
        "top-1": top1,
        "count is 1": exactly_one,
        "z clipped": z_clipped,
    }


def _sensor_lie_tracked(base, donors, n_sensors, block_size, rng):
    """Same draws as ``sensor_lie`` for a random donor, plus the replaced sensor ids."""
    out = base.copy()
    n_rows, n_features = out.shape
    n_blocks = n_features // block_size
    offsets = np.arange(block_size)
    which = np.empty((n_rows, n_sensors), dtype=np.int64)
    for i in range(n_rows):
        donor = int(rng.integers(0, donors.shape[0]))
        sensors = rng.choice(n_blocks, size=n_sensors, replace=False)
        which[i] = sensors
        columns = (sensors[:, None] * block_size + offsets).ravel()
        out[i, columns] = donors[donor, columns]
    return out, which[:, 0]


if __name__ == "__main__":
    main()
