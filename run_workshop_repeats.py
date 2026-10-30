"""Gas array: five reshuffles of the held-out pool, under several settings.

Settings (each can be one value or ``both``):

--split      random    shuffled 70/30 split, gas-stratified (the original)
             temporal  batches 1-7 train, batches 8-10 test (sensors drift)
--threshold  chi2      chi-square lines from the formula (the original)
             empirical lines from clean calibration rows, see thresholds.py
--stage2     hot       count sensors over their own line (the original)
             removal   count sensors set aside before the rest passes,
                       see consistency.py
             both      (default)
--tune-lie   both      tune the cutoff on both lie types below (default)
             sensors   tune on whole-sensor lies only
             features  tune on scattered single-feature lies only (the original)

Every run evaluates every corruption: four noise sizes, the original feature
lies, whole-sensor lies with a random donor, and quiet whole-sensor lies with
a same-gas donor. The training rows never change between the five seeds; the
seeds only reshuffle which held-out rows tune the cutoff and which score it.
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
from corruptions import NOISE_SCALES, POISON_FRACTIONS, SENSOR_COUNTS, random_noise, random_poison, sensor_lie
from data_loader import TEMPORAL_TRAIN_BATCHES, load_nominal_dataset, split_random_indices, split_temporal_indices
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
                    seed, x_test, gas_test, x_train, gas_train, std, blocks, full, check,
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
    parser.add_argument("--split", choices=("random", "temporal", "both"), default="both")
    parser.add_argument("--threshold", choices=("chi2", "empirical", "both"), default="both")
    parser.add_argument("--stage2", choices=("hot", "removal", "both"), default="both")
    parser.add_argument("--tune-lie", choices=("both", "sensors", "features"), default="both")
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET, help="clean alarm rate for empirical lines")
    return parser.parse_args()


def _one_split(seed, x_test, gas_test, x_train, gas_train, std, blocks, full, check, threshold, method, target, tune_lie):
    rng = np.random.default_rng(seed)
    order = rng.permutation(x_test.shape[0])
    mid = x_test.shape[0] // 2
    cal, eva = x_test[order[:mid]], x_test[order[mid:]]
    eva_gas = gas_test[order[mid:]]

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
        groups += [("lie", sensor_lie(cal, x_train, n, blocks.block_size, rng_tune)) for n in SENSOR_COUNTS]
    if tune_lie in ("features", "both"):
        groups += [("lie", random_poison(cal, x_train, f, rng_tune)) for f in POISON_FRACTIONS]
    cutoff = _tune_cutoff(cal, groups, counter, blocks.n_blocks)

    rng_eval = np.random.default_rng(seed + 1)
    eligible = ~counter(eva)[0]
    metrics: dict[str, float] = {"seed": seed, "cutoff": cutoff, "clean still alarming": _still_alarming(eva, counter)}
    noise_50 = None
    for scale in NOISE_SCALES:
        corrupted = random_noise(eva, std, scale, rng_eval)
        metrics[f"noise {scale:.2f}"] = _correct(eligible, corrupted, counter, cutoff, "noise")
        if scale == 0.50:
            noise_50 = corrupted
    for fraction in POISON_FRACTIONS:
        corrupted = random_poison(eva, x_train, fraction, rng_eval)
        metrics[f"feature lie {int(fraction * 100)}%"] = _correct(eligible, corrupted, counter, cutoff, "lie")
    one_sensor = None
    for n in SENSOR_COUNTS:
        corrupted = sensor_lie(eva, x_train, n, blocks.block_size, rng_eval)
        metrics[f"sensor lie {n}"] = _correct(eligible, corrupted, counter, cutoff, "lie")
        if n == 1:
            one_sensor = corrupted
    for n in SENSOR_COUNTS:
        corrupted = sensor_lie(eva, x_train, n, blocks.block_size, rng_eval, base_gas=eva_gas, donor_gas=gas_train)
        metrics[f"same-gas lie {n}"] = _correct(eligible, corrupted, counter, cutoff, "lie")

    counts = {
        "Clean": _count_when_alarmed(eva, counter),
        "Noise 0.50": _count_when_alarmed(noise_50, counter),
        "1-sensor lie": _count_when_alarmed(one_sensor, counter),
    }
    return metrics, counts


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
