"""Second check: did many sensors move, or only a few?

Stage 1 is the 128-dimensional alarm. On the rows it flags, count how many of
the 16 sensor blocks also cross their own line. The count cutoff is chosen on
one half of the held-out rows and scored on the other half.

The tuner uses exactly the scoring rule: only rows that were clean before
corruption count, a lie is right only with 1 to cutoff-1 hot sensors, and an
alarm with zero hot sensors is never right.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_loader import load_nominal_dataset, split_nominal
from monitor import BlockMahalanobisMonitor
from run_full_vs_blocks import FullMahalanobis
from run_random_catch import NOISE_SCALES, POISON_FRACTIONS, _random_noise, _random_poison

SEED = 42


def main() -> None:
    features, gas_ids = load_nominal_dataset()
    x_train, x_test = split_nominal(features, gas_ids, seed=SEED)
    blocks = BlockMahalanobisMonitor().fit(x_train)
    full = FullMahalanobis().fit(x_train)
    std = np.maximum(blocks.feature_std_, 1e-8)

    rng = np.random.default_rng(SEED)
    order = rng.permutation(x_test.shape[0])
    mid = x_test.shape[0] // 2
    cal, eva = x_test[order[:mid]], x_test[order[mid:]]

    rng_tune = np.random.default_rng(SEED + 7)
    groups = [("noise", _random_noise(cal, std, scale, rng_tune)) for scale in NOISE_SCALES]
    groups += [("lie", _random_poison(cal, x_train, fraction, rng_tune)) for fraction in POISON_FRACTIONS]
    cutoff = _tune_cutoff(cal, groups, blocks, full)

    print("Stage 1: all-128 alarm. Stage 2: how many of the 16 sensors also alarm.")
    print(f"Cutoff fit on {len(cal)} rows, scored on {len(eva)} other rows.")
    print(f"Call it noise if at least {cutoff} sensors alarm, otherwise poisoning.")
    print("An alarm with zero hot sensors is dropped as a false alarm.")
    print()

    rows = [_score_clean(eva, blocks, full, cutoff)]
    rng_eval = np.random.default_rng(SEED + 1)
    for scale in NOISE_SCALES:
        corrupted = _random_noise(eva, std, scale, rng_eval)
        rows.append(_score_bad("random noise", f"{scale:.2f} std", eva, corrupted, blocks, full, cutoff, "noise"))
    for fraction in POISON_FRACTIONS:
        corrupted = _random_poison(eva, x_train, fraction, rng_eval)
        rows.append(_score_bad("feature lie", f"{int(fraction * 100)}% features", eva, corrupted, blocks, full, cutoff, "lie"))

    table = pd.DataFrame(rows)
    print(
        table.to_string(
            index=False,
            formatters={
                "stage1 alarm": "{:.3f}".format,
                "after neighbor check": "{:.3f}".format,
                "median hot sensors": "{:.1f}".format,
                "called noise": "{:.3f}".format,
                "called lie": "{:.3f}".format,
                "correct type": "{:.3f}".format,
            },
        )
    )


def _tune_cutoff(base, groups, blocks, full) -> int:
    """Pick the hot-sensor cutoff with the best balanced accuracy.

    ``groups`` is a list of (kind, corrupted) with kind "noise" or "lie",
    each ``corrupted`` built from ``base`` row for row.
    """
    eligible = ~_alarm(base, full)
    n_eligible = int(eligible.sum())
    if n_eligible == 0:
        raise ValueError("No calibration row is clean before corruption")

    prepared: list[tuple[str, np.ndarray, np.ndarray]] = []
    for kind, corrupted in groups:
        if kind not in ("noise", "lie"):
            raise ValueError(f"Unknown group kind: {kind}")
        alarm = _alarm(corrupted, full)[eligible]
        hot = _block_hot(corrupted[eligible], blocks)
        prepared.append((kind, alarm, hot))

    best_k, best_score, best_lie = 1, -np.inf, 0.0
    for k in range(1, blocks.n_blocks + 1):
        noise_hits, lie_hits = [], []
        for kind, alarm, hot in prepared:
            if kind == "noise":
                noise_hits.append(float(np.mean(alarm & (hot >= k))))
            else:
                lie_hits.append(float(np.mean(alarm & (hot > 0) & (hot < k))))
        score = 0.5 * (float(np.mean(noise_hits)) + float(np.mean(lie_hits)))
        best_lie = max(best_lie, float(np.mean(lie_hits)))
        if score > best_score:
            best_score, best_k = score, k
    if best_lie == 0.0:
        print("WARNING: no cutoff calls any tuning lie correctly (lies alarm with zero hot sensors). "
              "The cutoff was chosen from noise alone.")
    return best_k


def _correct(base, corrupted, blocks, full, cutoff, expect) -> float:
    """Share of rows clean before corruption that are called the right kind."""
    eligible = ~_alarm(base, full)
    alarm = _alarm(corrupted, full) & eligible
    hot = _block_hot(corrupted, blocks)
    if expect == "noise":
        correct = alarm & (hot >= cutoff)
    else:
        correct = alarm & (hot > 0) & (hot < cutoff)
    n_eligible = int(eligible.sum())
    return float(correct.sum() / n_eligible) if n_eligible else float("nan")


def _still_alarming(frames, blocks, full) -> float:
    """Clean rows that survive both stages (and are therefore called something)."""
    return float(np.mean(_alarm(frames, full) & (_block_hot(frames, blocks) > 0)))


def _hot_when_alarmed(frames, blocks, full) -> np.ndarray:
    return _block_hot(frames, blocks)[_alarm(frames, full)]


def _score_clean(frames, blocks, full, cutoff) -> dict[str, object]:
    alarm = _alarm(frames, full)
    hot = _block_hot(frames, blocks)
    return {
        "kind": "clean",
        "size": "-",
        "stage1 alarm": float(alarm.mean()),
        "after neighbor check": float((alarm & (hot > 0)).mean()),
        "median hot sensors": float(np.median(hot[alarm])) if alarm.any() else 0.0,
        "called noise": float((alarm & (hot >= cutoff)).mean()),
        "called lie": float((alarm & (hot > 0) & (hot < cutoff)).mean()),
        "correct type": float("nan"),
    }


def _score_bad(kind, size, base, corrupted, blocks, full, cutoff, expect) -> dict[str, object]:
    eligible = ~_alarm(base, full)
    alarm = _alarm(corrupted, full) & eligible
    hot = _block_hot(corrupted, blocks)
    called_noise = alarm & (hot >= cutoff)
    called_lie = alarm & (hot > 0) & (hot < cutoff)
    correct = called_noise if expect == "noise" else called_lie
    n = int(eligible.sum())
    rate = lambda mask: float(mask.sum() / n) if n else float("nan")
    return {
        "kind": kind,
        "size": size,
        "stage1 alarm": rate(alarm),
        "after neighbor check": rate(alarm & (hot > 0)),
        "median hot sensors": float(np.median(hot[alarm])) if alarm.any() else 0.0,
        "called noise": rate(called_noise),
        "called lie": rate(called_lie),
        "correct type": rate(correct),
    }


def _alarm(frames: np.ndarray, full: FullMahalanobis) -> np.ndarray:
    return _full_distance(frames, full) > full.tau


def _full_distance(frames: np.ndarray, full: FullMahalanobis) -> np.ndarray:
    assert full.mu is not None and full.inv is not None
    residual = frames - full.mu
    return np.einsum("ij,jk,ik->i", residual, full.inv, residual)


def _block_distances(frames: np.ndarray, blocks: BlockMahalanobisMonitor) -> np.ndarray:
    """Squared Mahalanobis distance per row and block, shape (n_rows, n_blocks)."""
    assert blocks.means_ is not None and blocks.inv_covs_ is not None
    residual = frames.reshape(-1, blocks.n_blocks, blocks.block_size) - blocks.means_
    whitened = np.einsum("bij,nbj->nbi", blocks.inv_covs_, residual)
    return np.einsum("nbi,nbi->nb", residual, whitened)


def _block_hot(frames: np.ndarray, blocks: BlockMahalanobisMonitor) -> np.ndarray:
    """Number of hot sensors per row. ``blocks.tau`` may be one line or one per block."""
    return np.sum(_block_distances(frames, blocks) > blocks.tau, axis=1)


if __name__ == "__main__":
    main()
