"""Surprise check for sensors read at a steady pace, shared by the oven and the home.

Each channel's reading is turned into a *surprise*: the step from the
previous reading minus the step that was expected.

- ``jump`` detector: the expected step is the average step.
- ``trend`` detector: the expected step is a least-squares prediction from
  the channel's own last 3 steps. Steps persist (a channel that is rising
  keeps rising), so this removes part of the ordinary movement.

The surprises of the last W steps of one channel play the part of one gas
sensor's 8 numbers: a block of W. The gas monitors then apply unchanged,
``BlockMahalanobisMonitor`` (``hot``: count channels over their own line) and
``SensorRemovalCheck`` (``removal``: count channels set aside before the rest
passes).

Corruptions, always on the last W steps only, sized against the step:

- noise:         gaussian jitter of ``scale x usual step std`` on every
                 channel, independently every step, rounded to the channel's
                 resolution as the sensor itself would.
- obvious lie:   n channels show readings from a random training stretch.
- plausible lie: n channels replay: their last true reading plus the steps
                 of a random training stretch. With W = 1 this is the last
                 reading plus one ordinary step.
- stuck:         n channels freeze at their last true reading.

Fair scoring: when readings are coarse and most steps are 0, a plausible lie
or a stuck channel often shows exactly the true value. Such a corruption
changed nothing and no detector can see it. Only rows that were clean before
the corruption *and* were changed by it are scored, for the cutoff and for
the results alike.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from consistency import SensorRemovalCheck
from monitor import BlockMahalanobisMonitor
from run_neighbor_check import _correct, _tune_cutoff, removal_counter
from thresholds import empirical_lines

SEEDS = (42, 43, 44, 45, 46)
TRAIN_FRACTION = 0.70
TREND_LAGS = 3
FIT_WINDOWS = 40_000
EVAL_LIMIT = 4_000
LIE_KINDS = ("obvious lie", "plausible lie", "stuck")


@dataclass
class Setup:
    """What differs between machines."""

    values: np.ndarray  # (n_readings, n_channels), in time order
    stamps: np.ndarray  # datetime64, one per reading
    step_seconds: float  # the usual time between readings
    step_name: str  # "minute", "step", ... used in printed labels
    noise_scales: tuple[float, ...]
    lie_counts: tuple[int, ...]
    durations: tuple[int, ...]  # W options, in steps
    detectors: tuple[str, ...]
    thresholds: tuple[str, ...]
    methods: tuple[str, ...]
    target: float


def run_experiment(setup: Setup) -> dict[int, dict[tuple[str, str, str], pd.DataFrame]]:
    """Five seeds for every duration and configuration. Prints and returns the result tables."""
    values, stamps = setup.values, setup.stamps
    n_train = int(round(TRAIN_FRACTION * len(values)))
    print(f"Readings kept: {len(values)}, {values.shape[1]} channels. Train {stamps[0]} -> {stamps[n_train - 1]}.")
    print(f"Test {stamps[n_train]} -> {stamps[-1]}.")

    results: dict[int, dict[tuple[str, str, str], pd.DataFrame]] = {}
    for minutes in setup.durations:
        length = TREND_LAGS + minutes + 1
        train_starts = window_starts(stamps, length, 0, n_train, setup.step_seconds)
        test_starts = window_starts(stamps, length, n_train, len(values), setup.step_seconds)
        train_win = windows(values, spread_out(train_starts, FIT_WINDOWS), length)
        step_std = np.maximum(np.diff(train_win, axis=1)[:, -1, :].std(axis=0), 1e-8)

        print()
        print("#" * 78)
        print(f"Corruptions last {minutes} {setup.step_name}(s). Windows of {length} consecutive readings: "
              f"{len(train_starts)} train, {len(test_starts)} test.")

        surprises = {d: Surprise(0 if d == "jump" else TREND_LAGS).fit(train_win) for d in setup.detectors}
        for d, s in surprises.items():
            print(f"{d:5s} detector: surprise std / step std, median over channels "
                  f"{np.median(s.residual_std(train_win) / step_std):.2f}")

        configs = [(d, t, m) for d in setup.detectors for t in setup.thresholds for m in setup.methods]
        rows: dict[tuple[str, str, str], list[dict[str, float]]] = {c: [] for c in configs}
        changed_rows: list[dict[str, float]] = []
        for seed in SEEDS:
            cal, eva, tune_sets, eval_sets = corrupted_sets(
                seed, values, test_starts, length, minutes, train_win, step_std, setup
            )
            changed_rows.append({name: float(np.mean(changed(eva, w))) for name, _, w in eval_sets})
            for detector in setup.detectors:
                surprise = surprises[detector]
                monitors = fit_monitors(surprise(train_win, minutes), values.shape[1], minutes)
                features = _features(surprise, minutes, cal, eva, tune_sets, eval_sets)
                for threshold in setup.thresholds:
                    for method in setup.methods:
                        rows[(detector, threshold, method)].append(
                            _score(seed, features, monitors, threshold, method, setup.target)
                        )

        share = pd.DataFrame(changed_rows).median(axis=0)
        print(f"\nShare of corrupted rows whose reading actually changed (median over seeds):")
        print(share[[c for c in share.index if not c.startswith("noise")]].to_string(float_format="{:.3f}".format))

        tables: dict[tuple[str, str, str], pd.DataFrame] = {}
        summary: dict[str, pd.Series] = {}
        for config in configs:
            detector, threshold, method = config
            table = pd.DataFrame(rows[config])
            tables[config] = table
            print()
            print("=" * 78)
            print(f"{minutes} {setup.step_name}(s)   Detector: {detector}   Threshold: {threshold}   Stage 2: {method}")
            print(f"Cutoff per seed: {', '.join(str(int(c)) for c in table['cutoff'])}")
            numbers = table.drop(columns=["seed", "cutoff"])
            spread = pd.DataFrame({"min": numbers.min(axis=0), "median": numbers.median(axis=0), "max": numbers.max(axis=0)})
            print(spread.to_string(float_format="{:.3f}".format))
            summary[f"{detector} / {threshold} / {method}"] = spread["median"]
        results[minutes] = tables

        if len(summary) > 1:
            print()
            print("=" * 78)
            print(f"{minutes} {setup.step_name}(s): medians across the five seeds, side by side:")
            print(pd.DataFrame(summary).to_string(float_format="{:.3f}".format))
    return results


def window_starts(stamps: np.ndarray, length: int, lo: int, hi: int, step_seconds: float) -> np.ndarray:
    """Every start in ``[lo, hi - length]`` whose ``length`` readings are consecutive.

    Consecutive means each gap is within half a step of ``step_seconds``.
    """
    seconds = np.diff(stamps) / np.timedelta64(1, "s")
    gap = np.abs(seconds - step_seconds) > 0.5 * step_seconds
    gaps_before = np.concatenate([[0], np.cumsum(gap)])
    starts = np.arange(lo, hi - length + 1)
    return starts[gaps_before[starts + length - 1] == gaps_before[starts]]


def windows(values: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    """Readings of each window, shape ``(n, length, n_channels)``."""
    return values[starts[:, None] + np.arange(length)]


def spread_out(starts: np.ndarray, limit: int) -> np.ndarray:
    """At most ``limit`` starts, evenly spaced, so they cover the whole period."""
    if len(starts) <= limit:
        return starts
    return starts[:: len(starts) // limit][:limit]


class Surprise:
    """Step minus expected step, per channel.

    With ``lags=0`` the expected step is the average step. Otherwise it is a
    least-squares prediction from the channel's own previous ``lags`` steps,
    fit on clean training windows.
    """

    def __init__(self, lags: int) -> None:
        self.lags = lags
        self.coef_: np.ndarray | None = None  # (n_channels, 1 + lags): intercept, then lag 1..lags

    def fit(self, train: np.ndarray) -> "Surprise":
        jumps = np.diff(train, axis=1)
        if jumps.shape[1] < self.lags + 1:
            raise ValueError("Windows are too short for the number of lags")
        coef = np.empty((jumps.shape[2], 1 + self.lags))
        last = jumps.shape[1] - 1
        for ch in range(jumps.shape[2]):
            design = np.column_stack(
                [np.ones(jumps.shape[0])] + [jumps[:, last - k, ch] for k in range(1, self.lags + 1)]
            )
            coef[ch], *_ = np.linalg.lstsq(design, jumps[:, last, ch], rcond=None)
        self.coef_ = coef
        return self

    def __call__(self, frames: np.ndarray, minutes: int) -> np.ndarray:
        """Surprises of the last ``minutes`` steps, shape ``(n, n_channels * minutes)``.

        Laid out channel by channel, so each channel is one block.
        """
        assert self.coef_ is not None
        jumps = np.diff(frames, axis=1)
        n_jumps = jumps.shape[1]
        if n_jumps - minutes < self.lags:
            raise ValueError("Windows are too short for these steps and lags")
        out = np.empty((jumps.shape[0], jumps.shape[2], minutes))
        for i, t in enumerate(range(n_jumps - minutes, n_jumps)):
            expected = np.full((jumps.shape[0], jumps.shape[2]), self.coef_[:, 0])
            for k in range(1, self.lags + 1):
                expected += self.coef_[:, k] * jumps[:, t - k, :]
            out[:, :, i] = jumps[:, t, :] - expected
        return out.reshape(jumps.shape[0], -1)

    def residual_std(self, frames: np.ndarray) -> np.ndarray:
        return self(frames, 1).std(axis=0)


def _pick(n_rows: int, n_channels: int, n_pick: int, rng: np.random.Generator) -> np.ndarray:
    """``n_pick`` distinct channels per row, shape ``(n_rows, n_pick)``."""
    return np.argsort(rng.random((n_rows, n_channels)), axis=1)[:, :n_pick]


def channel_resolution(values: np.ndarray) -> np.ndarray:
    """The smallest step between distinct readings of each channel, shape ``(n_channels,)``.

    Sensors report in fixed increments: whole degrees, 0.06 degrees, whole
    light units. Float noise in the stored values is rounded away first.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])
    out = np.empty(values.shape[1])
    for ch in range(values.shape[1]):
        steps = np.diff(np.unique(np.round(values[:, ch], 9)))
        out[ch] = steps.min() if len(steps) else 0.0
    return out


def noise(frames, step_std, scale, minutes, rng, resolution=None) -> np.ndarray:
    """Gaussian jitter on every channel, rounded to each channel's resolution.

    A real sensor rounds whatever it measures, so noisy readings stay on its
    grid of possible values. Jitter smaller than half an increment rounds to
    nothing and leaves the reading exactly as it was.
    """
    out = frames.copy()
    shock = rng.normal(0.0, scale, size=(len(out), minutes, out.shape[2])) * step_std
    if resolution is not None:
        res = np.asarray(resolution, dtype=np.float64)
        shock = np.where(res > 0, np.round(shock / np.where(res > 0, res, 1.0)) * res, shock)
    out[:, -minutes:, :] += shock
    return out


def _replace(frames, replacement, n_pick, minutes, rng) -> np.ndarray:
    """``n_pick`` channels per row take ``replacement`` (shape ``(n, minutes, n_channels)``)."""
    out = frames.copy()
    rows = np.arange(len(out))[:, None]
    cols = _pick(len(out), out.shape[2], n_pick, rng)
    tail = out[:, -minutes:, :]
    tail[rows, :, cols] = replacement[rows, :, cols]
    return out


def obvious_lie(frames, donors, n_pick, minutes, rng) -> np.ndarray:
    donor = donors[rng.integers(0, len(donors), size=len(frames))]
    return _replace(frames, donor[:, -minutes:, :], n_pick, minutes, rng)


def plausible_lie(frames, donors, n_pick, minutes, rng) -> np.ndarray:
    donor = donors[rng.integers(0, len(donors), size=len(frames))]
    borrowed = np.diff(donor, axis=1)[:, -minutes:, :]
    replay = frames[:, -minutes - 1, :][:, None, :] + np.cumsum(borrowed, axis=1)
    return _replace(frames, replay, n_pick, minutes, rng)


def stuck(frames, n_pick, minutes, rng) -> np.ndarray:
    frozen = np.repeat(frames[:, -minutes - 1, :][:, None, :], minutes, axis=1)
    return _replace(frames, frozen, n_pick, minutes, rng)


def changed(base: np.ndarray, corrupted: np.ndarray) -> np.ndarray:
    return np.any(corrupted != base, axis=(1, 2))


def corruptions(base, donors, step_std, minutes, rng, setup: Setup) -> list[tuple[str, str, np.ndarray]]:
    """Every corruption of ``base`` as (name, kind, windows), kind "noise" or "lie".

    Noise is rounded to the resolution the clean donors show.
    """
    resolution = channel_resolution(donors)
    out = [
        (f"noise {s:.2f} step-std", "noise", noise(base, step_std, s, minutes, rng, resolution))
        for s in setup.noise_scales
    ]
    for n in setup.lie_counts:
        out.append((f"obvious lie {n}", "lie", obvious_lie(base, donors, n, minutes, rng)))
    for n in setup.lie_counts:
        out.append((f"plausible lie {n}", "lie", plausible_lie(base, donors, n, minutes, rng)))
    for n in setup.lie_counts:
        out.append((f"stuck {n}", "lie", stuck(base, n, minutes, rng)))
    return out


def corrupted_sets(seed, values, test_starts, length, minutes, donors, step_std, setup: Setup):
    """Calibration and evaluation windows for one seed, with their corruptions."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(test_starts)
    mid = len(order) // 2
    cal = windows(values, spread_out(order[:mid], EVAL_LIMIT), length)
    eva = windows(values, spread_out(order[mid:], EVAL_LIMIT), length)
    tune_sets = corruptions(cal, donors, step_std, minutes, np.random.default_rng(seed + 7), setup)
    eval_sets = corruptions(eva, donors, step_std, minutes, np.random.default_rng(seed + 1), setup)
    return cal, eva, tune_sets, eval_sets


def fit_monitors(train_features: np.ndarray, n_channels: int, minutes: int):
    blocks = BlockMahalanobisMonitor(n_blocks=n_channels, block_size=minutes).fit(train_features)
    check = SensorRemovalCheck(n_blocks=n_channels, block_size=minutes).fit(train_features)
    return blocks, check


def set_lines(monitors, cal_features: np.ndarray, threshold: str, target: float) -> None:
    blocks, check = monitors
    if threshold == "empirical":
        blocks.set_block_lines(empirical_lines(blocks.block_distances_batch(cal_features), target))
        check.set_lines(check.empirical_lines(cal_features, target))
    else:
        blocks.set_block_lines(None)
        check.set_lines(None)


def make_counter(monitors, method: str):
    blocks, check = monitors
    return hot_counter(blocks) if method == "hot" else removal_counter(check)


def hot_counter(blocks: BlockMahalanobisMonitor):
    """Channels over their own line. Any of them over the line is the alarm."""

    def count(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hot = blocks.hot_sensors(frames)
        return hot > 0, hot

    return count


def _features(surprise, minutes, cal, eva, tune_sets, eval_sets) -> dict[str, object]:
    def to_features(sets, base):
        return [(name, kind, surprise(w, minutes), changed(base, w)) for name, kind, w in sets]

    return {
        "cal": surprise(cal, minutes),
        "eva": surprise(eva, minutes),
        "tune": to_features(tune_sets, cal),
        "eval": to_features(eval_sets, eva),
    }


def _score(seed, features, monitors, threshold, method, target) -> dict[str, float]:
    cal, eva = features["cal"], features["eva"]
    set_lines(monitors, cal, threshold, target)
    counter = make_counter(monitors, method)

    groups = [(kind, f) for _, kind, f, _ in features["tune"]]
    cutoff = _tune_cutoff(cal, groups, counter, monitors[0].n_blocks)

    clean = ~counter(eva)[0]
    metrics: dict[str, float] = {"seed": seed, "cutoff": cutoff, "clean alarms": float(1.0 - clean.mean())}
    for name, kind, f, was_changed in features["eval"]:
        metrics[name] = _correct(clean & was_changed, f, counter, cutoff, kind)
    return metrics


def tuned_detector(setup: Setup, minutes: int, detector: str, threshold: str, method: str, seed: int = SEEDS[0]):
    """One detector set up exactly as in ``run_experiment`` for one seed.

    Returns ``(surprise, counter, cutoff, length)``. Lines come from the
    seed's calibration half of the test period and the cutoff is tuned on
    its corruptions, so the evaluation half stays unseen.
    """
    values, stamps = setup.values, setup.stamps
    n_train = int(round(TRAIN_FRACTION * len(values)))
    length = TREND_LAGS + minutes + 1
    train_starts = window_starts(stamps, length, 0, n_train, setup.step_seconds)
    test_starts = window_starts(stamps, length, n_train, len(values), setup.step_seconds)
    train_win = windows(values, spread_out(train_starts, FIT_WINDOWS), length)
    step_std = np.maximum(np.diff(train_win, axis=1)[:, -1, :].std(axis=0), 1e-8)
    cal, _, tune_sets, _ = corrupted_sets(seed, values, test_starts, length, minutes, train_win, step_std, setup)

    surprise = Surprise(0 if detector == "jump" else TREND_LAGS).fit(train_win)
    monitors = fit_monitors(surprise(train_win, minutes), values.shape[1], minutes)
    cal_features = surprise(cal, minutes)
    set_lines(monitors, cal_features, threshold, setup.target)
    counter = make_counter(monitors, method)
    groups = [(kind, surprise(w, minutes)) for _, kind, w in tune_sets]
    cutoff = _tune_cutoff(cal_features, groups, counter, monitors[0].n_blocks)
    return surprise, counter, cutoff, length


def calibration_half(setup: Setup, length: int, seed: int = SEEDS[0]) -> np.ndarray:
    """Start indices of the test windows that ``tuned_detector`` calibrated on, for that seed."""
    n_train = int(round(TRAIN_FRACTION * len(setup.values)))
    test_starts = window_starts(setup.stamps, length, n_train, len(setup.values), setup.step_seconds)
    order = np.random.default_rng(seed).permutation(test_starts)
    return order[: len(order) // 2]
