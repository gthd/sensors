"""Spike check on the roast oven: the jump from the previous minute.

The level of a thermometer wanders across years. The usual minute-to-minute
change does not. A thermometer spikes when its jump crosses a line fit on
clean minutes. Training is chronological: the first 70% of minutes.

The corruptions are sized against what the detector watches, the jump:

- noise:         gaussian jitter of ``scale x usual jump std`` on every
                 thermometer. Sizes 0.10-1.00 match the gas array; 2 and 4
                 are added to show where noise becomes visible at all.
- obvious lie:   n thermometers show a reading copied from a random training
                 minute, usually tens of degrees off (the original poison).
- plausible lie: n thermometers show the previous reading plus a jump
                 borrowed from a random training minute. Each lie looks like
                 an ordinary minute.

--threshold  chi2       chi-square line from the formula (the original)
             empirical  lines from clean calibration minutes, see thresholds.py
             both       (default)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from run_random_catch import _column_swap, _random_noise
from thresholds import DEFAULT_TARGET, empirical_lines

SEEDS = (42, 43, 44, 45, 46)
SYSTEM_FPR = 1e-3
TEMP_LOW = 50.0
TEMP_HIGH = 800.0
COLUMNS = tuple(f"T_data_{chamber}_{sensor}" for chamber in range(1, 6) for sensor in range(1, 4))
OVEN_NOISE_SCALES = (0.10, 0.25, 0.50, 1.00, 2.00, 4.00)
THERMO_COUNTS = (1, 2, 4, 6)


def main() -> None:
    args = _parse_args()
    thresholds = ("chi2", "empirical") if args.threshold == "both" else (args.threshold,)

    previous, current, stamps = _consecutive_minutes()
    n_train = int(round(0.70 * len(current)))
    prev_train, cur_train = previous[:n_train], current[:n_train]
    prev_test, cur_test = previous[n_train:], current[n_train:]
    jumps_train = cur_train - prev_train
    center = jumps_train.mean(axis=0)
    scale = np.maximum(jumps_train.std(axis=0), 1e-8)
    chi2_tau = float(chi2.ppf(1.0 - SYSTEM_FPR / len(COLUMNS), df=1))

    print(f"Consecutive minutes: {len(current)}. Train {stamps[0]} -> {stamps[n_train - 1]}.")
    print(f"Test {stamps[n_train]} -> {stamps[-1]}.")
    print(f"Usual jump: median std {np.median(scale):.3f} degrees. "
          f"Chi-square spike line about {np.median(np.sqrt(chi2_tau) * scale):.2f} degrees.")
    print("Noise is sized in units of the usual jump. Cutoff tuned on noise and obvious lies.")
    print()

    summary: dict[str, pd.Series] = {}
    for threshold in thresholds:
        print("=" * 78)
        print(f"Threshold: {threshold}" + (f" (about {args.target:.1%} of clean calibration minutes spike)" if threshold == "empirical" else ""))
        rows = [
            _one_seed(seed, prev_test, cur_test, cur_train, jumps_train, center, scale, chi2_tau, threshold, args.target)
            for seed in SEEDS
        ]
        table = pd.DataFrame(rows)
        print(f"Cutoff per seed: {', '.join(str(int(c)) for c in table['cutoff'])}")
        print(f"Spike line per seed (median, degrees): {', '.join(f'{v:.2f}' for v in table['line degrees'])}")
        print()
        values = table.drop(columns=["seed", "cutoff", "line degrees"])
        spread = pd.DataFrame(
            {
                "min": values.min(axis=0),
                "median": values.median(axis=0),
                "max": values.max(axis=0),
            }
        )
        print(spread.to_string(float_format="{:.3f}".format))
        print()
        summary[threshold] = spread["median"]

    if len(summary) > 1:
        print("=" * 78)
        print("Medians across the five seeds, side by side:")
        print(pd.DataFrame(summary).to_string(float_format="{:.3f}".format))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", choices=("chi2", "empirical", "both"), default="both")
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET, help="clean spike rate for empirical lines")
    return parser.parse_args()


def _consecutive_minutes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(
        Path(__file__).resolve().parent / "Dataset3" / "data_X.csv",
        usecols=["date_time", *COLUMNS],
        parse_dates=["date_time"],
    )
    values = frame[list(COLUMNS)].to_numpy(dtype=np.float64)
    keep = np.all((values >= TEMP_LOW) & (values <= TEMP_HIGH), axis=1)
    frame = frame.loc[keep]
    stamps = frame["date_time"].to_numpy()
    values = frame[list(COLUMNS)].to_numpy(dtype=np.float64)
    step = (stamps[1:] - stamps[:-1]) / np.timedelta64(1, "s")
    follow = step == 60
    return values[:-1][follow], values[1:][follow], stamps[1:][follow]


def _take(previous: np.ndarray, current: np.ndarray, limit: int = 8000) -> tuple[np.ndarray, np.ndarray]:
    if len(current) <= limit:
        return previous, current
    stride = len(current) // limit
    return previous[::stride][:limit], current[::stride][:limit]


def _z2(current, previous, center, scale) -> np.ndarray:
    jump = (current - previous - center) / scale
    return jump * jump


def _hot(current, previous, center, scale, tau) -> np.ndarray:
    """Spiking thermometers per minute. ``tau`` may be one line or one per thermometer."""
    return np.sum(_z2(current, previous, center, scale) > tau, axis=1)


def _plausible_lie(current, previous, donor_jumps, n_thermo, rng) -> np.ndarray:
    """n thermometers show previous reading + a jump borrowed from a random training minute."""
    believable = previous + donor_jumps[rng.integers(0, len(donor_jumps), size=len(current))]
    return _column_swap_rows(current, believable, n_thermo, rng)


def _column_swap_rows(base, replacement, n_columns, rng) -> np.ndarray:
    """Like ``_column_swap``, but row i takes its values from ``replacement[i]``."""
    out = base.copy()
    n_rows, n_features = out.shape
    for i in range(n_rows):
        columns = rng.choice(n_features, size=n_columns, replace=False)
        out[i, columns] = replacement[i, columns]
    return out


def _one_seed(seed, prev_test, cur_test, levels_train, jumps_train, center, scale, chi2_tau, threshold, target):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(cur_test))
    mid = len(order) // 2
    cal_prev, cal_cur = _take(prev_test[order[:mid]], cur_test[order[:mid]])
    eva_prev, eva_cur = _take(prev_test[order[mid:]], cur_test[order[mid:]])

    if threshold == "empirical":
        tau = empirical_lines(_z2(cal_cur, cal_prev, center, scale), target)
    else:
        tau = chi2_tau

    rng_tune = np.random.default_rng(seed + 7)
    groups = [("noise", _random_noise(cal_cur, scale, s, rng_tune)) for s in OVEN_NOISE_SCALES]
    groups += [("lie", _column_swap(cal_cur, levels_train, n, rng_tune)) for n in THERMO_COUNTS]
    cutoff = _tune(cal_cur, cal_prev, groups, center, scale, tau)

    rng_eval = np.random.default_rng(seed + 1)
    metrics: dict[str, float] = {
        "seed": seed,
        "cutoff": cutoff,
        "line degrees": float(np.median(np.sqrt(tau) * scale)),
        "clean spikes": float(np.mean(_hot(eva_cur, eva_prev, center, scale, tau) > 0)),
    }
    for s in OVEN_NOISE_SCALES:
        corrupted = _random_noise(eva_cur, scale, s, rng_eval)
        metrics[f"noise {s:.2f} jump-std"] = _correct(eva_cur, corrupted, eva_prev, center, scale, tau, cutoff, "noise")
    for n in THERMO_COUNTS:
        corrupted = _column_swap(eva_cur, levels_train, n, rng_eval)
        metrics[f"obvious lie {n}"] = _correct(eva_cur, corrupted, eva_prev, center, scale, tau, cutoff, "lie")
    for n in THERMO_COUNTS:
        corrupted = _plausible_lie(eva_cur, eva_prev, jumps_train, n, rng_eval)
        metrics[f"plausible lie {n}"] = _correct(eva_cur, corrupted, eva_prev, center, scale, tau, cutoff, "lie")
    return metrics


def _tune(base, previous, groups, center, scale, tau) -> int:
    """Balanced-accuracy cutoff, using exactly the scoring rule in ``_correct``."""
    eligible = _hot(base, previous, center, scale, tau) == 0
    prepared = [(kind, _hot(c[eligible], previous[eligible], center, scale, tau)) for kind, c in groups]
    best_k, best_score = 1, -np.inf
    for k in range(1, len(COLUMNS) + 1):
        noise_hits = [float(np.mean(h >= k)) for kind, h in prepared if kind == "noise"]
        lie_hits = [float(np.mean((h > 0) & (h < k))) for kind, h in prepared if kind == "lie"]
        score = 0.5 * (float(np.mean(noise_hits)) + float(np.mean(lie_hits)))
        if score > best_score:
            best_score, best_k = score, k
    return best_k


def _correct(base, current, previous, center, scale, tau, cutoff, expect) -> float:
    """Share of minutes clean before corruption that are called the right kind."""
    eligible = _hot(base, previous, center, scale, tau) == 0
    hot = _hot(current, previous, center, scale, tau)
    if expect == "noise":
        correct = eligible & (hot >= cutoff)
    else:
        correct = eligible & (hot > 0) & (hot < cutoff)
    n = int(eligible.sum())
    return float(correct.sum() / n) if n else float("nan")


if __name__ == "__main__":
    main()
