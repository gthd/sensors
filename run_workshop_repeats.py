"""Gas array: five reshuffles of the held-out pool, under several settings.

Settings (each can be one value or ``both``):

--split      random    shuffled 70/30 split, gas-stratified (the original)
             ordered   earliest 70% train, last 30% test, in recording order
             temporal  batches 1-7 train, batches 8-10 test (sensors drift)
--threshold  chi2      chi-square lines from the formula (the original)
             empirical lines from clean calibration rows, see thresholds.py
--stage2     hot       count sensors over their own line (the original)
             removal   count sensors set aside before the rest passes,
                       see consistency.py
             both      (default)
--tune-lie   sensors   tune on the shared lies: obvious copy, same-gas replay,
                       and stuck at the previous snapshot (default)
             features  tune on scattered single-feature lies only
             both      tune on the shared lies and on feature lies

Every run evaluates the same attacks as the time-series monitors: noise at
0.25x to 4x, an obvious whole-sensor copy, a same-gas replay, and a stuck
sensor, each on 1, 2 and 4 sensors. Scattered feature lies are scored too,
but they do not move the cutoff unless ``--tune-lie`` asks for them. The
training rows never change between the five seeds; the seeds only reshuffle
which held-out rows tune the cutoff and which score it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from consistency import SensorRemovalCheck
from corruptions import (
    LIE_COUNTS,
    NOISE_SCALES,
    POISON_FRACTIONS,
    random_noise,
    random_poison,
    sensor_lie,
    sensor_stuck,
)
from data_loader import (
    TEMPORAL_TRAIN_BATCHES,
    TRAIN_FRACTION,
    load_nominal_dataset,
    split_ordered_indices,
    split_random_indices,
    split_temporal_indices,
)
from monitor import BlockMahalanobisMonitor, FullMahalanobis
from run_neighbor_check import _correct, _count_when_alarmed, _still_alarming, _tune_cutoff, hot_counter, removal_counter
from thresholds import DEFAULT_TARGET, empirical_line, empirical_lines

SEEDS = (42, 43, 44, 45, 46)
FIGURE_SEED = 42
SPLIT_SEED = 42


def main() -> None:
    args = _parse_args()
    splits = ("random", "temporal") if args.split == "both" else (args.split,)
    thresholds = ("chi2", "empirical") if args.threshold == "both" else (args.threshold,)
    methods = ("hot", "removal") if args.stage2 == "both" else (args.stage2,)

    features, gas_ids, batch_ids = load_nominal_dataset(return_batches=True)
    summary: dict[str, pd.Series] = {}

    for split in splits:
        if split == "random":
            train_idx, test_idx = split_random_indices(gas_ids, seed=SPLIT_SEED)
        elif split == "ordered":
            train_idx, test_idx = split_ordered_indices(len(features), TRAIN_FRACTION)
        else:
            train_idx, test_idx = split_temporal_indices(batch_ids)
        x_train, gas_train = features[train_idx], gas_ids[train_idx]
        x_test, gas_test = features[test_idx], gas_ids[test_idx]
        blocks = BlockMahalanobisMonitor().fit(x_train)
        full = FullMahalanobis().fit(x_train)
        check = SensorRemovalCheck().fit(x_train)
        std = np.maximum(blocks.feature_std_, 1e-8)

        for threshold, method in [(t, m) for t in thresholds for m in methods]:
            label = f"{split} / {threshold} / {method}"
            print("=" * 78)
            print(f"Split: {split}   Threshold: {threshold}   Stage 2: {method}   Cutoff tuned on: {args.tune_lie} lies")
            if split == "temporal":
                print(f"Train batches 1-{TEMPORAL_TRAIN_BATCHES} ({len(train_idx)} rows), test batches "
                      f"{TEMPORAL_TRAIN_BATCHES + 1}-{int(batch_ids.max())} ({len(test_idx)} rows).")
            elif split == "ordered":
                print(f"Earliest {len(train_idx)} rows train ({len(train_idx) / len(features):.1%}), "
                      f"last {len(test_idx)} rows are the test pool.")
                print(f"Train batches {_batch_span(batch_ids[train_idx])}, "
                      f"test batches {_batch_span(batch_ids[test_idx])}.")
                train_dist = full.distances(x_train)
                print(f"Train distances: median {np.median(train_dist):.1f}, "
                      f"99% {np.quantile(train_dist, 0.99):.1f}.")
            else:
                print(f"Train {len(train_idx)} rows, test pool {len(test_idx)} rows (shuffled, seed {SPLIT_SEED}).")
            if threshold == "empirical":
                print(f"Lines set so about {args.target:.1%} of clean calibration rows alarm (per seed).")
            else:
                print(f"Chi-square lines: all-128 {full.chi2_tau:.1f}, per sensor {blocks.chi2_tau:.1f}.")
            if method == "removal":
                print("Stage 2 counts the sensors set aside before the rest passes (the suspects).")

            rows, counts, cutoff = [], None, None
            for seed in SEEDS:
                metrics, seed_counts = _one_split(
                    seed, features, test_idx, gas_ids, x_train, gas_train, std, blocks, full, check,
                    threshold, method, args.target, args.tune_lie,
                )
                rows.append(metrics)
                if seed == FIGURE_SEED:
                    counts, cutoff = seed_counts, metrics["cutoff"]

            table = pd.DataFrame(rows)
            print(f"Cutoff per seed: {', '.join(str(int(c)) for c in table['cutoff'])}")
            print()
            spread = _spread(table.drop(columns=["seed", "cutoff"]))
            print(spread.to_string(float_format="{:.3f}".format))
            summary[label] = spread["median"]

            assert counts is not None and cutoff is not None
            tag = f"{split}_{threshold}" if method == "hot" else f"{split}_{threshold}_{method}"
            path = _save_figure(counts, int(cutoff), tag, label, method)
            print(f"\nFigure: {path}\n")

    if len(summary) > 1:
        print("=" * 78)
        print("Medians across the five seeds, side by side:")
        print(pd.DataFrame(summary).to_string(float_format="{:.3f}".format))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("random", "ordered", "temporal", "both"), default="both")
    parser.add_argument("--threshold", choices=("chi2", "empirical", "both"), default="both")
    parser.add_argument("--stage2", choices=("hot", "removal", "both"), default="both")
    parser.add_argument("--tune-lie", choices=("both", "sensors", "features"), default="sensors")
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET, help="clean alarm rate for empirical lines")
    return parser.parse_args()


def _one_split(seed, features, test_idx, gas_ids, x_train, gas_train, std, blocks, full, check, threshold, method, target, tune_lie):
    x_test, gas_test = features[test_idx], gas_ids[test_idx]
    rng = np.random.default_rng(seed)
    order = rng.permutation(x_test.shape[0])
    mid = x_test.shape[0] // 2
    cal_pos, eva_pos = order[:mid], order[mid:]
    cal, eva = x_test[cal_pos], x_test[eva_pos]
    cal_gas, eva_gas = gas_test[cal_pos], gas_test[eva_pos]
    prev_cal = _previous_snapshot(features, test_idx[cal_pos])
    prev_eva = _previous_snapshot(features, test_idx[eva_pos])

    if threshold == "empirical":
        full.tau = empirical_line(full.distances(cal), target)
        blocks.set_block_lines(empirical_lines(blocks.block_distances_batch(cal), target))
        check.set_lines(check.empirical_lines(cal, target))
    else:
        full.tau = full.chi2_tau
        blocks.set_block_lines(None)
        check.set_lines(None)
    counter = hot_counter(blocks, full) if method == "hot" else removal_counter(check)

    rng_tune = np.random.default_rng(seed + 7)
    groups = [("noise", random_noise(cal, std, scale, rng_tune)) for scale in NOISE_SCALES]
    if tune_lie in ("sensors", "both"):
        groups += [("lie", sensor_lie(cal, x_train, n, blocks.block_size, rng_tune)) for n in LIE_COUNTS]
        groups += [
            ("lie", sensor_lie(cal, x_train, n, blocks.block_size, rng_tune, base_gas=cal_gas, donor_gas=gas_train))
            for n in LIE_COUNTS
        ]
        groups += [("lie", sensor_stuck(cal, prev_cal, n, blocks.block_size, rng_tune)) for n in LIE_COUNTS]
    if tune_lie in ("features", "both"):
        groups += [("lie", random_poison(cal, x_train, f, rng_tune)) for f in POISON_FRACTIONS]
    cutoff = _tune_cutoff(cal, groups, counter, blocks.n_blocks)

    rng_eval = np.random.default_rng(seed + 1)
    eligible = ~counter(eva)[0]
    metrics: dict[str, float] = {"seed": seed, "cutoff": cutoff, "clean still alarming": _still_alarming(eva, counter)}
    noise_50 = None
    for scale in NOISE_SCALES:
        corrupted = random_noise(eva, std, scale, rng_eval)
        metrics[f"noise {scale:.2f}"] = _rate(eligible, eva, corrupted, counter, cutoff, "noise")
        if scale == 0.50:
            noise_50 = corrupted
    one_sensor = None
    for n in LIE_COUNTS:
        if n == 1 and method == "removal":
            corrupted, lied = sensor_lie(
                eva, x_train, n, blocks.block_size, rng_eval, return_blocks=True,
            )
            changed = eligible & np.any(corrupted != eva, axis=1)
            metrics["lie 1 count is 1"] = _lie_exact(changed, corrupted, lied[:, 0], check, cutoff)
            metrics["lie 1 first sensor"] = _lie_first(changed, corrupted, lied[:, 0], check, cutoff)
        else:
            corrupted = sensor_lie(eva, x_train, n, blocks.block_size, rng_eval)
        metrics[f"obvious lie {n}"] = _rate(eligible, eva, corrupted, counter, cutoff, "lie")
        if n == 1:
            one_sensor = corrupted
            changed = eligible & np.any(corrupted != eva, axis=1)
            alarm, _ = counter(corrupted)
            metrics["obvious lie 1 alarm"] = float(np.mean(alarm[changed])) if changed.any() else float("nan")
    for n in LIE_COUNTS:
        corrupted = sensor_lie(eva, x_train, n, blocks.block_size, rng_eval, base_gas=eva_gas, donor_gas=gas_train)
        metrics[f"plausible lie {n}"] = _rate(eligible, eva, corrupted, counter, cutoff, "lie")
    for n in LIE_COUNTS:
        corrupted = sensor_stuck(eva, prev_eva, n, blocks.block_size, rng_eval)
        metrics[f"stuck {n}"] = _rate(eligible, eva, corrupted, counter, cutoff, "lie")
        metrics[f"stuck {n} changed"] = float(np.mean(np.any(corrupted != eva, axis=1)))
    for fraction in POISON_FRACTIONS:
        corrupted = random_poison(eva, x_train, fraction, rng_eval)
        metrics[f"feature lie {int(fraction * 100)}%"] = _rate(eligible, eva, corrupted, counter, cutoff, "lie")
    if threshold == "empirical":
        metrics["threshold"] = float(full.tau)
        metrics["eval median distance"] = float(np.median(full.distances(eva)))

    counts = {
        "Clean": _count_when_alarmed(eva, counter),
        "Noise 0.50": _count_when_alarmed(noise_50, counter),
        "1-sensor lie": _count_when_alarmed(one_sensor, counter),
    }
    return metrics, counts


def _previous_snapshot(features: np.ndarray, row_idx: np.ndarray) -> np.ndarray:
    """The recording just before each row. Row 0 has no predecessor, so it stays put."""
    return features[np.maximum(row_idx - 1, 0)]


def _rate(eligible, base, corrupted, counter, cutoff, expect) -> float:
    """Same fair scoring as the time-series monitors: unchanged rows are left out."""
    changed = eligible & np.any(corrupted != base, axis=1)
    return _correct(changed, corrupted, counter, cutoff, expect)


def _batch_span(batch_ids: np.ndarray) -> str:
    counts = pd.Series(batch_ids).value_counts().sort_index()
    return ", ".join(f"{int(batch)} ({int(n)})" for batch, n in counts.items())


def _caught_mask(eligible, corrupted, check, cutoff):
    counts, order = check.removal(corrupted)
    caught = eligible & (counts > 0) & (counts < cutoff)
    return caught, counts, order


def _lie_exact(eligible, corrupted, lied, check, cutoff) -> float:
    """Share of caught one-sensor lies whose removal count is exactly one."""
    caught, counts, _ = _caught_mask(eligible, corrupted, check, cutoff)
    if not caught.any():
        return float("nan")
    return float(np.mean(counts[caught] == 1))


def _lie_first(eligible, corrupted, lied, check, cutoff) -> float:
    """Share of caught one-sensor lies whose first suspect is the copied sensor."""
    caught, _, order = _caught_mask(eligible, corrupted, check, cutoff)
    if not caught.any():
        return float("nan")
    return float(np.mean(order[caught, 0] == lied[caught]))


def _spread(table: pd.DataFrame) -> pd.DataFrame:
    values = table.to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "min": np.nanmin(values, axis=0),
            "median": np.nanmedian(values, axis=0),
            "max": np.nanmax(values, axis=0),
        },
        index=table.columns,
    )


def _save_figure(counts: dict[str, np.ndarray], cutoff: int, tag: str, title: str, method: str) -> Path:
    path = Path(__file__).resolve().parent / "figures" / f"hot_sensors_{tag}.png"
    path.parent.mkdir(exist_ok=True)
    bins = np.arange(-0.5, 17.5, 1.0)
    colors = {"Clean": "#7a7a7a", "Noise 0.50": "#1f4e79", "1-sensor lie": "#c47b2b"}
    plt.figure(figsize=(7.2, 4.2))
    for name, values in counts.items():
        hist, edges = np.histogram(values, bins=bins)
        share = hist / max(len(values), 1)
        centers = (edges[:-1] + edges[1:]) / 2
        plt.plot(centers, share, marker="o", label=f"{name} (n={len(values)})", color=colors[name])
    plt.axvline(cutoff - 0.5, color="black", linestyle="--", linewidth=1, label=f"noise if at least {cutoff}")
    plt.xticks(range(0, 17))
    plt.xlabel("Sensors that also alarm" if method == "hot" else "Sensors set aside before the rest passes")
    plt.ylabel("Share of alarmed rows")
    plt.title(title)
    plt.ylim(0, 1)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


if __name__ == "__main__":
    main()
