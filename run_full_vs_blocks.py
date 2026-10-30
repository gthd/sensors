"""Compare 16 blocks of 8 against one Mahalanobis test on all 128 features."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import chi2

from data_loader import load_nominal_dataset, split_nominal
from monitor import SYSTEM_FPR, BlockMahalanobisMonitor

SEED = 42
N_SAMPLES = 500
RIDGE = 1e-6


class FullMahalanobis:
    """One squared Mahalanobis distance on the whole 128-vector."""

    def __init__(self, system_fpr: float = SYSTEM_FPR, ridge: float = RIDGE) -> None:
        self.system_fpr = system_fpr
        self.ridge = ridge
        self.mu: np.ndarray | None = None
        self.inv: np.ndarray | None = None
        self.tau = 0.0

    def fit(self, x_train: np.ndarray) -> "FullMahalanobis":
        x_train = np.asarray(x_train, dtype=np.float64)
        self.mu = x_train.mean(axis=0)
        cov = np.cov(x_train, rowvar=False)
        cov = cov + self.ridge * np.eye(x_train.shape[1])
        self.inv = np.linalg.inv(cov)
        self.chi2_tau = float(chi2.ppf(1.0 - self.system_fpr, df=x_train.shape[1]))
        self.tau = self.chi2_tau
        return self

    def distance(self, frame: np.ndarray) -> float:
        assert self.mu is not None and self.inv is not None
        residual = np.asarray(frame, dtype=np.float64) - self.mu
        return float(residual @ self.inv @ residual)

    def alarms(self, frame: np.ndarray) -> bool:
        return self.distance(frame) > self.tau


def _time_us(fn, frames: np.ndarray) -> np.ndarray:
    fn(frames[0])
    elapsed = np.empty(frames.shape[0], dtype=np.float64)
    for i, frame in enumerate(frames):
        started = time.perf_counter_ns()
        fn(frame)
        elapsed[i] = (time.perf_counter_ns() - started) / 1_000.0
    return elapsed


def _make_variants(base: np.ndarray, donors: np.ndarray, blocks: BlockMahalanobisMonitor, seed: int):
    """The corruption library from run_random_catch, as (name, frames) pairs.

    This replaces the missing ``library_variants.make_variants``.
    """
    # Imported here: run_random_catch imports FullMahalanobis from this module.
    from run_random_catch import NOISE_SCALES, POISON_FRACTIONS, SENSOR_COUNTS, _random_noise, _random_poison, _sensor_lie

    rng = np.random.default_rng(seed)
    std = np.maximum(blocks.feature_std_, 1e-8)
    variants: list[tuple[str, np.ndarray]] = []
    for scale in NOISE_SCALES:
        variants.append((f"noise {scale:.2f} std", _random_noise(base, std, scale, rng)))
    for fraction in POISON_FRACTIONS:
        variants.append((f"feature lie {int(fraction * 100)}%", _random_poison(base, donors, fraction, rng)))
    for count in SENSOR_COUNTS:
        variants.append((f"sensor lie {count}", _sensor_lie(base, donors, count, blocks.block_size, rng)))
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
