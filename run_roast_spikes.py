"""Spike check on the roast oven: is a thermometer's last move a surprise?

Fifteen thermometers, one reading a minute. The level of a thermometer
wanders across years. How it moves from minute to minute does not. Each
reading is turned into a *surprise*: the jump from the previous minute minus
the jump that was expected.

--detector  jump    expected jump = the usual average jump (the original)
            trend   expected jump = a linear prediction from the thermometer's
                    own last 3 jumps. Jumps persist (a thermometer that is
                    rising keeps rising), so this removes about two thirds
                    of the ordinary movement.
            both    (default)

The surprises of the last W minutes of one thermometer play the part of one
gas sensor's 8 numbers: a block of W. The same monitors as the gas array
then apply, ``BlockMahalanobisMonitor`` and ``SensorRemovalCheck``.

--minutes   1       corruptions that last one minute (the original)
            10      corruptions that last ten minutes
            both    (default)
--stage2    hot     count thermometers over their own line (the original)
            removal count thermometers set aside before the rest passes
            both    (default)
--threshold chi2 | empirical | both (default)

Corruptions, always on the last W minutes only, sized against what the
detector watches:

- noise:         gaussian jitter of ``scale x usual jump std`` on every
                 thermometer, independently every minute.
- obvious lie:   n thermometers show readings from a random stretch of
                 training minutes, usually tens of degrees off.
- plausible lie: n thermometers replay: their last true reading plus the
                 jumps of a random stretch of training minutes. With W = 1
                 this is the last reading plus one ordinary jump.
- stuck:         n thermometers freeze at their last true reading.

Fair scoring: readings are whole degrees and most minutes the jump is 0, so
a plausible lie or a stuck thermometer often shows exactly the true value.
Such a corruption changed nothing and no detector can see it. Only minutes
that were clean before the corruption *and* were changed by it are scored,
for the cutoff and for the results alike. The share that changed is printed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from consistency import SensorRemovalCheck
from monitor import BlockMahalanobisMonitor
from run_neighbor_check import _correct, _tune_cutoff, removal_counter
from thresholds import DEFAULT_TARGET, empirical_lines

SEEDS = (42, 43, 44, 45, 46)
TEMP_LOW = 50.0
TEMP_HIGH = 800.0
COLUMNS = tuple(f"T_data_{chamber}_{sensor}" for chamber in range(1, 6) for sensor in range(1, 4))
N_THERMO = len(COLUMNS)
TRAIN_FRACTION = 0.70
TREND_LAGS = 3
FIT_WINDOWS = 40_000
EVAL_LIMIT = 4_000
OVEN_NOISE_SCALES = (0.10, 0.25, 0.50, 1.00, 2.00, 4.00)
THERMO_COUNTS = (1, 2, 4, 6)
LIE_KINDS = ("obvious lie", "plausible lie", "stuck")


def main() -> None:
    args = _parse_args()
    minute_options = (1, 10) if args.minutes == "both" else (int(args.minutes),)
    detectors = ("jump", "trend") if args.detector == "both" else (args.detector,)
    thresholds = ("chi2", "empirical") if args.threshold == "both" else (args.threshold,)
    methods = ("hot", "removal") if args.stage2 == "both" else (args.stage2,)

    values, stamps = _load_oven()
    n_train = int(round(TRAIN_FRACTION * len(values)))
    print(f"Minutes kept: {len(values)}. Train {stamps[0]} -> {stamps[n_train - 1]}.")
    print(f"Test {stamps[n_train]} -> {stamps[-1]}.")

    for minutes in minute_options:
        length = TREND_LAGS + minutes + 1
        train_starts = _window_starts(stamps, length, 0, n_train)
        test_starts = _window_starts(stamps, length, n_train, len(values))
        train_win = _windows(values, _spread_out(train_starts, FIT_WINDOWS), length)
        jump_std = np.maximum(np.diff(train_win, axis=1)[:, -1, :].std(axis=0), 1e-8)

        print()
        print("#" * 78)
        print(f"Corruptions last {minutes} minute(s). Windows of {length} consecutive minutes: "
              f"{len(train_starts)} train, {len(test_starts)} test.")
        print(f"Usual jump: median std {np.median(jump_std):.3f} degrees.")

        surprises = {d: Surprise(0 if d == "jump" else TREND_LAGS).fit(train_win) for d in detectors}
        for d, s in surprises.items():
            print(f"{d:5s} detector: surprise std / jump std, median over thermometers "
                  f"{np.median(s.residual_std(train_win) / jump_std):.2f}")

        configs = [(d, t, m) for d in detectors for t in thresholds for m in methods]
        rows: dict[tuple[str, str, str], list[dict[str, float]]] = {c: [] for c in configs}
        changed_rows: list[dict[str, float]] = []
        for seed in SEEDS:
            cal, eva, tune_sets, eval_sets = _corrupted_sets(seed, values, test_starts, length, minutes, train_win, jump_std)
            changed_rows.append({name: float(np.mean(_changed(eva, w))) for name, _, w in eval_sets})
            for detector in detectors:
                surprise = surprises[detector]
                monitors = _fit_monitors(surprise(train_win, minutes), minutes)
                features = _features(surprise, minutes, cal, eva, tune_sets, eval_sets)
                for threshold in thresholds:
                    for method in methods:
                        metrics = _score(seed, features, monitors, threshold, method, args.target)
                        rows[(detector, threshold, method)].append(metrics)

        changed = pd.DataFrame(changed_rows).median(axis=0)
        print("\nShare of corrupted minutes whose reading actually changed (median over seeds):")
        print(changed[[c for c in changed.index if not c.startswith("noise")]].to_string(float_format="{:.3f}".format))

        summary: dict[str, pd.Series] = {}
        for config in configs:
            detector, threshold, method = config
            table = pd.DataFrame(rows[config])
            label = f"{detector} / {threshold} / {method}"
            print()
            print("=" * 78)
            print(f"{minutes} minute(s)   Detector: {detector}   Threshold: {threshold}   Stage 2: {method}")
            print(f"Cutoff per seed: {', '.join(str(int(c)) for c in table['cutoff'])}")
            values_only = table.drop(columns=["seed", "cutoff"])
            spread = pd.DataFrame(
                {"min": values_only.min(axis=0), "median": values_only.median(axis=0), "max": values_only.max(axis=0)}
            )
            print(spread.to_string(float_format="{:.3f}".format))
            summary[label] = spread["median"]

        if len(summary) > 1:
            print()
            print("=" * 78)
            print(f"{minutes} minute(s): medians across the five seeds, side by side:")
            print(pd.DataFrame(summary).to_string(float_format="{:.3f}".format))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minutes", choices=("1", "10", "both"), default="both")
    parser.add_argument("--detector", choices=("jump", "trend", "both"), default="both")
    parser.add_argument("--threshold", choices=("chi2", "empirical", "both"), default="both")
    parser.add_argument("--stage2", choices=("hot", "removal", "both"), default="both")
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET, help="clean alarm rate for empirical lines")
    return parser.parse_args()


def _load_oven() -> tuple[np.ndarray, np.ndarray]:
    """Readings with every thermometer in a plausible range, and their time stamps."""
    frame = pd.read_csv(
        Path(__file__).resolve().parent / "Dataset3" / "data_X.csv",
        usecols=["date_time", *COLUMNS],
        parse_dates=["date_time"],
    )
    values = frame[list(COLUMNS)].to_numpy(dtype=np.float64)
    keep = np.all((values >= TEMP_LOW) & (values <= TEMP_HIGH), axis=1)
    return values[keep], frame["date_time"].to_numpy()[keep]


def _window_starts(stamps: np.ndarray, length: int, lo: int, hi: int) -> np.ndarray:
    """Every start in ``[lo, hi - length]`` whose ``length`` minutes are consecutive."""
    gap = (np.diff(stamps) / np.timedelta64(1, "s")) != 60
    gaps_before = np.concatenate([[0], np.cumsum(gap)])
    starts = np.arange(lo, hi - length + 1)
    return starts[gaps_before[starts + length - 1] == gaps_before[starts]]


def _windows(values: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    """Readings of each window, shape ``(n, length, N_THERMO)``."""
    return values[starts[:, None] + np.arange(length)]


def _spread_out(starts: np.ndarray, limit: int) -> np.ndarray:
    """At most ``limit`` starts, evenly spaced, so they cover the whole period."""
    if len(starts) <= limit:
        return starts
    return starts[:: len(starts) // limit][:limit]


class Surprise:
    """Jump minus expected jump, per thermometer.

    With ``lags=0`` the expected jump is the average jump. Otherwise it is a
    least-squares prediction from the thermometer's own previous ``lags``
    jumps, fit on clean training windows.
    """

    def __init__(self, lags: int) -> None:
        self.lags = lags
        self.coef_: np.ndarray | None = None  # (N_THERMO, 1 + lags): intercept, then lag 1..lags

    def fit(self, windows: np.ndarray) -> "Surprise":
        jumps = np.diff(windows, axis=1)
        if jumps.shape[1] < self.lags + 1:
            raise ValueError("Windows are too short for the number of lags")
        coef = np.empty((jumps.shape[2], 1 + self.lags))
        for th in range(jumps.shape[2]):
            design = self._design(jumps, jumps.shape[1] - 1, th)
            coef[th], *_ = np.linalg.lstsq(design, jumps[:, -1, th], rcond=None)
        self.coef_ = coef
        return self

    def __call__(self, windows: np.ndarray, minutes: int) -> np.ndarray:
        """Surprises of the last ``minutes`` jumps, shape ``(n, N_THERMO * minutes)``.

        Laid out thermometer by thermometer, so each thermometer is one block.
        """
        assert self.coef_ is not None
        jumps = np.diff(windows, axis=1)
        n_jumps = jumps.shape[1]
        if n_jumps - minutes < self.lags:
            raise ValueError("Windows are too short for these minutes and lags")
        out = np.empty((jumps.shape[0], jumps.shape[2], minutes))
        for i, t in enumerate(range(n_jumps - minutes, n_jumps)):
            expected = np.full((jumps.shape[0], jumps.shape[2]), self.coef_[:, 0])
            for k in range(1, self.lags + 1):
                expected += self.coef_[:, k] * jumps[:, t - k, :]
            out[:, :, i] = jumps[:, t, :] - expected
        return out.reshape(jumps.shape[0], -1)

    def residual_std(self, windows: np.ndarray) -> np.ndarray:
        return self(windows, 1).std(axis=0)

    def _design(self, jumps: np.ndarray, t: int, th: int) -> np.ndarray:
        columns = [np.ones(jumps.shape[0])] + [jumps[:, t - k, th] for k in range(1, self.lags + 1)]
        return np.column_stack(columns)


def _pick(n_rows: int, n_thermo: int, rng: np.random.Generator) -> np.ndarray:
    """``n_thermo`` distinct thermometers per row, shape ``(n_rows, n_thermo)``."""
    return np.argsort(rng.random((n_rows, N_THERMO)), axis=1)[:, :n_thermo]


def _noise(windows, jump_std, scale, minutes, rng) -> np.ndarray:
    out = windows.copy()
    out[:, -minutes:, :] += rng.normal(0.0, scale, size=(len(out), minutes, N_THERMO)) * jump_std
    return out


def _replace(windows, replacement, n_thermo, minutes, rng) -> np.ndarray:
    """``n_thermo`` thermometers per row take ``replacement`` (shape ``(n, minutes, N_THERMO)``)."""
    out = windows.copy()
    rows = np.arange(len(out))[:, None]
    cols = _pick(len(out), n_thermo, rng)
    tail = out[:, -minutes:, :]
    tail[rows, :, cols] = replacement[rows, :, cols]
    return out


def _obvious_lie(windows, donors, n_thermo, minutes, rng) -> np.ndarray:
    donor = donors[rng.integers(0, len(donors), size=len(windows))]
    return _replace(windows, donor[:, -minutes:, :], n_thermo, minutes, rng)


def _plausible_lie(windows, donors, n_thermo, minutes, rng) -> np.ndarray:
    donor = donors[rng.integers(0, len(donors), size=len(windows))]
    borrowed = np.diff(donor, axis=1)[:, -minutes:, :]
    replay = windows[:, -minutes - 1, :][:, None, :] + np.cumsum(borrowed, axis=1)
    return _replace(windows, replay, n_thermo, minutes, rng)


def _stuck(windows, n_thermo, minutes, rng) -> np.ndarray:
    frozen = np.repeat(windows[:, -minutes - 1, :][:, None, :], minutes, axis=1)
    return _replace(windows, frozen, n_thermo, minutes, rng)


def _changed(base: np.ndarray, corrupted: np.ndarray) -> np.ndarray:
    return np.any(corrupted != base, axis=(1, 2))


def _corruptions(base, donors, jump_std, minutes, rng) -> list[tuple[str, str, np.ndarray]]:
    """Every corruption of ``base`` as (name, kind, windows), kind "noise" or "lie"."""
    out = [(f"noise {s:.2f} jump-std", "noise", _noise(base, jump_std, s, minutes, rng)) for s in OVEN_NOISE_SCALES]
    for n in THERMO_COUNTS:
        out.append((f"obvious lie {n}", "lie", _obvious_lie(base, donors, n, minutes, rng)))
    for n in THERMO_COUNTS:
        out.append((f"plausible lie {n}", "lie", _plausible_lie(base, donors, n, minutes, rng)))
    for n in THERMO_COUNTS:
        out.append((f"stuck {n}", "lie", _stuck(base, n, minutes, rng)))
    return out


def _corrupted_sets(seed, values, test_starts, length, minutes, donors, jump_std):
    """Calibration and evaluation windows for one seed, with their corruptions."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(test_starts)
    mid = len(order) // 2
    cal = _windows(values, _spread_out(order[:mid], EVAL_LIMIT), length)
    eva = _windows(values, _spread_out(order[mid:], EVAL_LIMIT), length)
    tune_sets = _corruptions(cal, donors, jump_std, minutes, np.random.default_rng(seed + 7))
    eval_sets = _corruptions(eva, donors, jump_std, minutes, np.random.default_rng(seed + 1))
    return cal, eva, tune_sets, eval_sets


def _fit_monitors(train_features: np.ndarray, minutes: int):
    blocks = BlockMahalanobisMonitor(n_blocks=N_THERMO, block_size=minutes).fit(train_features)
    check = SensorRemovalCheck(n_blocks=N_THERMO, block_size=minutes).fit(train_features)
    return blocks, check


def _features(surprise, minutes, cal, eva, tune_sets, eval_sets) -> dict[str, object]:
    def to_features(sets, base):
        return [(name, kind, surprise(w, minutes), _changed(base, w)) for name, kind, w in sets]

    return {
        "cal": surprise(cal, minutes),
        "eva": surprise(eva, minutes),
        "tune": to_features(tune_sets, cal),
        "eval": to_features(eval_sets, eva),
    }


def _score(seed, features, monitors, threshold, method, target) -> dict[str, float]:
    blocks, check = monitors
    cal, eva = features["cal"], features["eva"]
    if threshold == "empirical":
        blocks.set_block_lines(empirical_lines(blocks.block_distances_batch(cal), target))
        check.set_lines(check.empirical_lines(cal, target))
    else:
        blocks.set_block_lines(None)
        check.set_lines(None)
    counter = _hot_counter(blocks) if method == "hot" else removal_counter(check)

    groups = [(kind, f) for _, kind, f, changed in features["tune"]]
    cutoff = _tune_cutoff(cal, groups, counter, blocks.n_blocks)

    clean = ~counter(eva)[0]
    metrics: dict[str, float] = {"seed": seed, "cutoff": cutoff, "clean alarms": float(1.0 - clean.mean())}
    for name, kind, f, changed in features["eval"]:
        metrics[name] = _correct(clean & changed, f, counter, cutoff, kind)
    return metrics


def _hot_counter(blocks: BlockMahalanobisMonitor):
    """Thermometers over their own line. Any of them over the line is the alarm."""

    def count(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hot = blocks.hot_sensors(frames)
        return hot > 0, hot

    return count


if __name__ == "__main__":
    main()
