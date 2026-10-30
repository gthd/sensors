"""Walking: six joint angles through one gait cycle, with and without a brace.

Data: UCI "Multivariate Gait Data", in ``DatasetGait/gait.csv``. Ten people
walked ten times in each of three conditions: no brace, a knee brace and an
ankle brace (both braces on the right leg). Each walk is one gait cycle of
101 points (0-100% of the cycle) with six angles: ankle, knee and hip of the
left and right leg.

The six angles are the channels and the points of a cycle are the steps.
Windows never cross from one walk into the next.

What makes this set different:

- The two legs are near twins: the same joint left and right moves almost
  in step (step correlation 0.91-0.98), so every channel has a partner to
  disagree with.
- The curves are smooth (step autocorrelation 0.97-0.99), so the ``trend``
  detector predicts the next step well.
- The braces are a real change that looks like a lie on one channel: the
  knee brace cuts the right knee's range from about 66 to 22 degrees, the
  ankle brace the right ankle's from about 26 to 6.

Clean data is walking without a brace: walks 1-7 of every person train,
walks 8-10 test. The corruption experiment runs on those (``timeseries.py``).
Then the braced walks go through the tuned detectors: do they alarm, are
they called noise or a lie, and which joint is blamed?

--steps     1 | 10 | both (default)
--detector  jump | trend | both (default)
--threshold chi2 | empirical | both (default)
--stage2    hot | removal | both (default)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from thresholds import DEFAULT_TARGET
from timeseries import Setup, calibration_half, run_experiment, tuned_detector, window_starts, windows

DATA_FILE = Path(__file__).resolve().parent / "DatasetGait" / "gait.csv"
CHANNELS = ("left ankle", "left knee", "left hip", "right ankle", "right knee", "right hip")
CONDITIONS = {1: "no brace", 2: "knee brace", 3: "ankle brace"}
BRACED_JOINT = {"knee brace": "knee", "ankle brace": "ankle"}
TRAIN_WALKS = range(1, 8)
TEST_WALKS = range(8, 11)
POINTS = 101
WALK_GAP = 1000  # seconds between walks in the made-up time stamps, so no window spans two walks
NOISE_SCALES = (0.25, 0.50, 1.00, 2.00, 4.00)
LIE_COUNTS = (1, 2, 3)


def main() -> None:
    args = _parse_args()
    walks = load_walks()
    values, stamps = stack_walks(walks, [(1, w) for w in TRAIN_WALKS] + [(1, w) for w in TEST_WALKS])
    setup = Setup(
        values=values,
        stamps=stamps,
        step_seconds=1.0,
        step_name="point",
        noise_scales=NOISE_SCALES,
        lie_counts=LIE_COUNTS,
        durations=(1, 10) if args.steps == "both" else (int(args.steps),),
        detectors=("jump", "trend") if args.detector == "both" else (args.detector,),
        thresholds=("chi2", "empirical") if args.threshold == "both" else (args.threshold,),
        methods=("hot", "removal") if args.stage2 == "both" else (args.stage2,),
        target=args.target,
    )
    run_experiment(setup)
    _braced_walks(setup, walks)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", choices=("1", "10", "both"), default="both")
    parser.add_argument("--detector", choices=("jump", "trend", "both"), default="both")
    parser.add_argument("--threshold", choices=("chi2", "empirical", "both"), default="both")
    parser.add_argument("--stage2", choices=("hot", "removal", "both"), default="both")
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET, help="clean alarm rate for empirical lines")
    return parser.parse_args()


def load_walks() -> pd.DataFrame:
    """One row per (subject, condition, replication, time), one column per channel."""
    frame = pd.read_csv(DATA_FILE)
    wide = frame.pivot_table(
        index=["subject", "condition", "replication", "time"], columns=["leg", "joint"], values="angle"
    )
    wide.columns = [CHANNELS[(leg - 1) * 3 + (joint - 1)] for leg, joint in wide.columns]
    return wide[list(CHANNELS)].sort_index()


def stack_walks(walks: pd.DataFrame, keys: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Walks for (condition, replication) keys, every subject, one after another.

    Returns the readings and made-up time stamps: one second per point and a
    long gap between walks, so ``window_starts`` never joins two walks.
    """
    blocks = []
    for condition, walk in keys:
        part = walks.xs((condition, walk), level=("condition", "replication"))
        for subject in part.index.get_level_values("subject").unique():
            blocks.append(part.xs(subject, level="subject").to_numpy(dtype=np.float64))
    values = np.vstack(blocks)
    offsets = np.concatenate([i * WALK_GAP + np.arange(len(b)) for i, b in enumerate(blocks)])
    stamps = np.datetime64("2000-01-01T00:00:00") + offsets.astype("timedelta64[s]")
    return values, stamps


def _braced_walks(setup: Setup, walks: pd.DataFrame) -> None:
    """Braced walks through detectors tuned on unbraced walking.

    The unbraced row uses test walks never used for the lines or the cutoff.
    Braced walks were never seen at all.
    """
    n_train = int(round(0.70 * len(setup.values)))
    sets = {}
    for condition, name in CONDITIONS.items():
        if condition == 1:
            continue
        sets[name] = stack_walks(walks, [(condition, w) for w in range(1, 11)])

    print()
    print("=" * 78)
    print("Braced walks through detectors tuned on unbraced walking (empirical lines).")
    print("'called lie': share of alarms called a lie. 'blamed': the joint the removal check")
    print("sets aside first, among alarms called a lie, with its share. 'braced joint, either leg':")
    print("share of those where the first joint set aside is the braced joint on either leg.")
    rows = []
    for minutes in setup.durations:
        for detector in setup.detectors:
            for method in setup.methods:
                surprise, counter, cutoff, length, monitors = tuned_detector(setup, minutes, detector, "empirical", method)
                check = monitors[1]
                starts = window_starts(setup.stamps, length, n_train, len(setup.values), setup.step_seconds)
                starts = np.setdiff1d(starts, calibration_half(setup, length))
                groups = {"no brace (test)": windows(setup.values, starts, length)}
                for name, (values, stamps) in sets.items():
                    groups[name] = windows(values, window_starts(stamps, length, 0, len(values), 1.0), length)
                for name, frames in groups.items():
                    features = surprise(frames, minutes)
                    alarm, count = counter(features)
                    lie = alarm & (count < cutoff)
                    row = {
                        "points": minutes,
                        "detector / stage 2": f"{detector} / {method} (cutoff {cutoff})",
                        "walks": name,
                        "alarm rate": float(alarm.mean()),
                        "called lie": float(lie.sum() / alarm.sum()) if alarm.any() else float("nan"),
                        "blamed": "-",
                        "braced joint, either leg": float("nan"),
                    }
                    if lie.any():
                        first = check.removal(features[lie])[1][:, 0]
                        top = np.bincount(first, minlength=len(CHANNELS))
                        row["blamed"] = f"{CHANNELS[int(top.argmax())]} ({top.max() / lie.sum():.0%})"
                        if name in BRACED_JOINT:
                            joint = np.array([c.endswith(BRACED_JOINT[name]) for c in CHANNELS])
                            row["braced joint, either leg"] = float(joint[first].mean())
                    rows.append(row)
    table = pd.DataFrame(rows).set_index(["points", "detector / stage 2", "walks"])
    print(table.to_string(float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
