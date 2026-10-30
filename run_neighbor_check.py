"""Second check: did many sensors move, or only a few?

Two ways to count, chosen with ``--stage2``:

hot      Stage 1 is the 128-dimensional alarm. On the rows it flags, count
         how many of the 16 sensor blocks also cross their own line (the
         original). A sensor that lies with a real reading from another
         moment looks normal on its own, so this count is often zero.
removal  Count how many sensors must be set aside before the rest passes,
         see ``consistency.py``. This measures disagreement between sensors.

Many sensors means noise, few means a lie. The count cutoff is chosen on one
half of the held-out rows and scored on the other half.

The tuner uses exactly the scoring rule: only rows that were clean before
corruption count, a lie is right only with 1 to cutoff-1 counted sensors, and
an alarm with zero counted sensors is never right.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from consistency import SensorRemovalCheck
from corruptions import NOISE_SCALES, POISON_FRACTIONS, SENSOR_COUNTS, random_noise, random_poison, sensor_lie
from data_loader import load_nominal_dataset, split_nominal
from monitor import BlockMahalanobisMonitor, FullMahalanobis

SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage2", choices=("hot", "removal", "both"), default="both")
    args = parser.parse_args()
    methods = ("hot", "removal") if args.stage2 == "both" else (args.stage2,)

    features, gas_ids = load_nominal_dataset()
    x_train, x_test = split_nominal(features, gas_ids, seed=SEED)
    blocks = BlockMahalanobisMonitor().fit(x_train)
    full = FullMahalanobis().fit(x_train)
    check = SensorRemovalCheck().fit(x_train)
    std = np.maximum(blocks.feature_std_, 1e-8)

    rng = np.random.default_rng(SEED)
    order = rng.permutation(x_test.shape[0])
    mid = x_test.shape[0] // 2
    cal, eva = x_test[order[:mid]], x_test[order[mid:]]

    for method in methods:
        counter = hot_counter(blocks, full) if method == "hot" else removal_counter(check)
        rng_tune = np.random.default_rng(SEED + 7)
        groups = [("noise", random_noise(cal, std, scale, rng_tune)) for scale in NOISE_SCALES]
        groups += [("lie", random_poison(cal, x_train, fraction, rng_tune)) for fraction in POISON_FRACTIONS]
        groups += [("lie", sensor_lie(cal, x_train, n, blocks.block_size, rng_tune)) for n in SENSOR_COUNTS]
        cutoff = _tune_cutoff(cal, groups, counter, blocks.n_blocks)

        print("=" * 78)
        if method == "hot":
            print("Stage 1: all-128 alarm. Stage 2: how many of the 16 sensors also alarm on their own.")
        else:
            print("Stage 1: all-128 alarm. Stage 2: how many sensors must be set aside before the rest passes.")
        print(f"Cutoff fit on {len(cal)} rows, scored on {len(eva)} other rows.")
        print(f"Call it noise if the count is at least {cutoff}, otherwise a lie.")
        print("An alarm with a count of zero is dropped as a false alarm.")
        print()

        rows = [_score_clean(eva, counter, cutoff)]
        rng_eval = np.random.default_rng(SEED + 1)
        for scale in NOISE_SCALES:
            corrupted = random_noise(eva, std, scale, rng_eval)
            rows.append(_score_bad("random noise", f"{scale:.2f} std", eva, corrupted, counter, cutoff, "noise"))
        for fraction in POISON_FRACTIONS:
            corrupted = random_poison(eva, x_train, fraction, rng_eval)
            rows.append(_score_bad("feature lie", f"{int(fraction * 100)}% features", eva, corrupted, counter, cutoff, "lie"))
        for n in SENSOR_COUNTS:
            corrupted = sensor_lie(eva, x_train, n, blocks.block_size, rng_eval)
            rows.append(_score_bad("sensor lie", f"{n} sensors", eva, corrupted, counter, cutoff, "lie"))

        table = pd.DataFrame(rows)
        print(
            table.to_string(
                index=False,
                formatters={
                    "stage1 alarm": "{:.3f}".format,
                    "after neighbor check": "{:.3f}".format,
                    "median count": "{:.1f}".format,
                    "called noise": "{:.3f}".format,
                    "called lie": "{:.3f}".format,
                    "correct type": "{:.3f}".format,
                },
            )
        )
        print()


def hot_counter(blocks: BlockMahalanobisMonitor, full: FullMahalanobis):
    """The original second check: the all-128 alarm, then sensors over their own line."""

    def count(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return full.distances(frames) > full.tau, blocks.hot_sensors(frames)

    return count


def removal_counter(check: SensorRemovalCheck):
    """Sensors set aside before the rest passes. Alarm and count come from the same check."""

    def count(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        counts = check.counts(frames)
        return counts > 0, counts

    return count


def _tune_cutoff(base, groups, counter, n_blocks: int) -> int:
    """Pick the count cutoff with the best balanced accuracy.

    ``groups`` is a list of (kind, corrupted) with kind "noise" or "lie",
    each ``corrupted`` built from ``base`` row for row. ``counter`` maps
    frames to (alarm, count). Rows the corruption left unchanged are skipped:
    nothing could tell them apart from clean rows.
    """
    eligible = ~counter(base)[0]
    if not eligible.any():
        raise ValueError("No calibration row is clean before corruption")

    prepared: list[tuple[str, np.ndarray, np.ndarray]] = []
    for kind, corrupted in groups:
        if kind not in ("noise", "lie"):
            raise ValueError(f"Unknown group kind: {kind}")
        rows = eligible & np.any(corrupted != base, axis=1)
        if not rows.any():
            continue
        alarm, count = counter(corrupted[rows])
        prepared.append((kind, alarm, count))
    if not any(kind == "noise" for kind, _, _ in prepared) or not any(kind == "lie" for kind, _, _ in prepared):
        raise ValueError("Tuning needs at least one changed noise row and one changed lie row")

    best_k, best_score, best_lie = 1, -np.inf, 0.0
    for k in range(1, n_blocks + 1):
        noise_hits, lie_hits = [], []
        for kind, alarm, count in prepared:
            if kind == "noise":
                noise_hits.append(float(np.mean(alarm & (count >= k))))
            else:
                lie_hits.append(float(np.mean(alarm & (count > 0) & (count < k))))
        score = 0.5 * (float(np.mean(noise_hits)) + float(np.mean(lie_hits)))
        best_lie = max(best_lie, float(np.mean(lie_hits)))
        if score > best_score:
            best_score, best_k = score, k
    if best_lie == 0.0:
        print("WARNING: no cutoff calls any tuning lie correctly (lies alarm with a count of zero). "
              "The cutoff was chosen from noise alone.")
    return best_k


def _correct(eligible, corrupted, counter, cutoff, expect) -> float:
    """Share of rows clean before corruption that are called the right kind.

    ``eligible`` marks the rows that did not alarm before corruption,
    ``~counter(base)[0]``. It is passed in so it is computed once per base.
    """
    alarm, count = counter(corrupted)
    alarm = alarm & eligible
    if expect == "noise":
        correct = alarm & (count >= cutoff)
    else:
        correct = alarm & (count > 0) & (count < cutoff)
    n_eligible = int(eligible.sum())
    return float(correct.sum() / n_eligible) if n_eligible else float("nan")


def _still_alarming(frames, counter) -> float:
    """Clean rows that survive both stages (and are therefore called something)."""
    alarm, count = counter(frames)
    return float(np.mean(alarm & (count > 0)))


def _count_when_alarmed(frames, counter) -> np.ndarray:
    alarm, count = counter(frames)
    return count[alarm]


def _score_clean(frames, counter, cutoff) -> dict[str, object]:
    alarm, count = counter(frames)
    return {
        "kind": "clean",
        "size": "-",
        "stage1 alarm": float(alarm.mean()),
        "after neighbor check": float((alarm & (count > 0)).mean()),
        "median count": float(np.median(count[alarm])) if alarm.any() else 0.0,
        "called noise": float((alarm & (count >= cutoff)).mean()),
        "called lie": float((alarm & (count > 0) & (count < cutoff)).mean()),
        "correct type": float("nan"),
    }


def _score_bad(kind, size, base, corrupted, counter, cutoff, expect) -> dict[str, object]:
    eligible = ~counter(base)[0]
    alarm, count = counter(corrupted)
    alarm = alarm & eligible
    called_noise = alarm & (count >= cutoff)
    called_lie = alarm & (count > 0) & (count < cutoff)
    correct = called_noise if expect == "noise" else called_lie
    n = int(eligible.sum())
    rate = lambda mask: float(mask.sum() / n) if n else float("nan")
    return {
        "kind": kind,
        "size": size,
        "stage1 alarm": rate(alarm),
        "after neighbor check": rate(alarm & (count > 0)),
        "median count": float(np.median(count[alarm])) if alarm.any() else 0.0,
        "called noise": rate(called_noise),
        "called lie": rate(called_lie),
        "correct type": rate(correct),
    }


if __name__ == "__main__":
    main()
