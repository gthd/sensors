"""Compare 16 blocks of 8 against one Mahalanobis test on all 128 features."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from corruptions import NOISE_SCALES, POISON_FRACTIONS, SENSOR_COUNTS, random_noise, random_poison, sensor_lie
from data_loader import load_nominal_dataset, split_nominal
from monitor import BlockMahalanobisMonitor, FullMahalanobis

SEED = 42
N_SAMPLES = 500


def _time_us(fn, frames: np.ndarray) -> np.ndarray:
    fn(frames[0])
    elapsed = np.empty(frames.shape[0], dtype=np.float64)
    for i, frame in enumerate(frames):
        started = time.perf_counter_ns()
        fn(frame)
        elapsed[i] = (time.perf_counter_ns() - started) / 1_000.0
    return elapsed


def _make_variants(base: np.ndarray, donors: np.ndarray, blocks: BlockMahalanobisMonitor, seed: int):
    """The corruption library from corruptions.py, as (name, frames) pairs."""
    rng = np.random.default_rng(seed)
    std = np.maximum(blocks.feature_std_, 1e-8)
    variants: list[tuple[str, np.ndarray]] = []
    for scale in NOISE_SCALES:
        variants.append((f"noise {scale:.2f} std", random_noise(base, std, scale, rng)))
    for fraction in POISON_FRACTIONS:
        variants.append((f"feature lie {int(fraction * 100)}%", random_poison(base, donors, fraction, rng)))
    for count in SENSOR_COUNTS:
        variants.append((f"sensor lie {count}", sensor_lie(base, donors, count, blocks.block_size, rng)))
    return variants


def main() -> None:
    features, gas_ids = load_nominal_dataset()
    x_train, x_test = split_nominal(features, gas_ids, seed=SEED)
    blocks = BlockMahalanobisMonitor().fit(x_train)
    full = FullMahalanobis().fit(x_train)

    rng = np.random.default_rng(SEED)
    base = x_test[rng.choice(x_test.shape[0], size=N_SAMPLES, replace=False)]
    block_clean = np.array([not np.any(blocks.block_distances(row) > blocks.tau) for row in base])
    full_clean = np.array([not full.alarms(row) for row in base])

    rows: list[dict[str, object]] = []
    for name, frames in _make_variants(base, x_train, blocks, seed=SEED):
        block_hit = np.array([np.any(blocks.block_distances(row) > blocks.tau) for row in frames])
        full_hit = np.array([full.alarms(row) for row in frames])
        block_eligible = int(block_clean.sum())
        full_eligible = int(full_clean.sum())
        rows.append(
            {
                "variant": name,
                "blocks caught": int(np.logical_and(block_hit, block_clean).sum()),
                "blocks of": block_eligible,
                "blocks rate": float(np.logical_and(block_hit, block_clean).sum() / block_eligible),
                "all-128 caught": int(np.logical_and(full_hit, full_clean).sum()),
                "all-128 of": full_eligible,
                "all-128 rate": float(np.logical_and(full_hit, full_clean).sum() / full_eligible),
            }
        )

    table = pd.DataFrame(rows)
    block_us = _time_us(lambda frame: np.any(blocks.block_distances(frame) > blocks.tau), base)
    full_us = _time_us(full.alarms, base)

    print("Same 500 held-out rows. Catch rate uses only rows that monitor called clean.")
    print(f"Block tau = {blocks.tau:.3f} (df=8, alpha={blocks.alpha_block:.7f})")
    print(f"All-128 tau = {full.tau:.3f} (df=128, alpha={full.system_fpr})")
    print(f"Clean rows left unflagged: blocks {int(block_clean.sum())} / {N_SAMPLES}, all-128 {int(full_clean.sum())} / {N_SAMPLES}")
    print(f"False alarms on these rows: blocks {1 - block_clean.mean():.4f}, all-128 {1 - full_clean.mean():.4f}")
    print()
    print(
        table.to_string(
            index=False,
            formatters={"blocks rate": "{:.4f}".format, "all-128 rate": "{:.4f}".format},
        )
    )
    print()
    print(f"Blocks mean {block_us.mean():.3f} us   p99 {np.percentile(block_us, 99):.3f} us")
    print(f"All-128 mean {full_us.mean():.3f} us   p99 {np.percentile(full_us, 99):.3f} us")


if __name__ == "__main__":
    main()
