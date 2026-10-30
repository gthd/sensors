"""A room with four sensor nodes, CO2 and motion, and the true head count.

Data: UCI "Room Occupancy Estimation", in ``DatasetRoom/Occupancy_Estimation.csv``.
Four nodes (S1-S4) each measure temperature, light and sound. S5 measures
CO2 and S6, S7 are motion (PIR, on or off). One reading about every 30
seconds over seven days in December 2017 and January 2018. Every reading
says how many people (0 to 3) were in the room.

Left out, which leaves 13 channels:

- ``S5_CO2_Slope`` is computed from the CO2 reading, not measured, so a lie
  on it means nothing.
- ``S6_PIR`` and ``S7_PIR`` are motion switches, on or off, and 94-96% of
  their steps are exactly 0. A line learned from clean data then sits at
  zero, so any added jitter, however small, crosses it, and gaussian noise
  on an on/off switch is not physical anyway.

The same machinery as the oven and the home (``timeseries.py``). Durations
are one step (30 seconds) and 10 steps (5 minutes).

After the corruption experiment comes the question only this dataset can
answer, because it has the head count: on clean test readings, do alarms
happen in an empty room (false alarms) or when people come and go (real
events)? And when real events alarm, are they called noise or a lie?

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

DATA_FILE = Path(__file__).resolve().parent / "DatasetRoom" / "Occupancy_Estimation.csv"
CHANNELS = (
    "S1_Temp", "S2_Temp", "S3_Temp", "S4_Temp",
    "S1_Light", "S2_Light", "S3_Light", "S4_Light",
    "S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound",
    "S5_CO2",
)
STEP_SECONDS = 30.5
NOISE_SCALES = (0.25, 0.50, 1.00, 2.00, 4.00)
LIE_COUNTS = (1, 2, 4)
EVENT_LOOKBACK = 10  # steps, about 5 minutes


def main() -> None:
    args = _parse_args()
    values, stamps, occupancy = load_room()
    setup = Setup(
        values=values,
        stamps=stamps,
        step_seconds=STEP_SECONDS,
        step_name="step",
        noise_scales=NOISE_SCALES,
        lie_counts=LIE_COUNTS,
        durations=(1, 10) if args.steps == "both" else (int(args.steps),),
        detectors=("jump", "trend") if args.detector == "both" else (args.detector,),
        thresholds=("chi2", "empirical") if args.threshold == "both" else (args.threshold,),
        methods=("hot", "removal") if args.stage2 == "both" else (args.stage2,),
        target=args.target,
    )
    run_experiment(setup)
    _alarms_by_occupancy(setup, occupancy)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", choices=("1", "10", "both"), default="both")
    parser.add_argument("--detector", choices=("jump", "trend", "both"), default="both")
    parser.add_argument("--threshold", choices=("chi2", "empirical", "both"), default="both")
    parser.add_argument("--stage2", choices=("hot", "removal", "both"), default="both")
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET, help="clean alarm rate for empirical lines")
    return parser.parse_args()


def load_room() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Readings in time order, their time stamps, and the head count at each reading."""
    frame = pd.read_csv(DATA_FILE)
    frame["stamp"] = pd.to_datetime(frame["Date"] + " " + frame["Time"], format="%Y/%m/%d %H:%M:%S")
    frame = frame.sort_values("stamp")
    return (
        frame[list(CHANNELS)].to_numpy(dtype=np.float64),
        frame["stamp"].to_numpy(),
        frame["Room_Occupancy_Count"].to_numpy(dtype=np.int64),
    )


def occupancy_state(occupancy: np.ndarray, ends: np.ndarray, lookback: int) -> np.ndarray:
    """For each reading index in ``ends``: "empty", "occupied" or "people came or went".

    "people came or went" means the head count changed within the last
    ``lookback`` readings. Otherwise the room was steadily empty or occupied.
    """
    state = np.empty(len(ends), dtype=object)
    for i, end in enumerate(ends):
        recent = occupancy[max(0, end - lookback) : end + 1]
        if np.any(recent != recent[-1]):
            state[i] = "people came or went"
        elif recent[-1] == 0:
            state[i] = "empty"
        else:
            state[i] = "occupied"
    return state


def _alarms_by_occupancy(setup: Setup, occupancy: np.ndarray) -> None:
    """Alarms on clean, unseen test readings, split by what the people were doing."""
    minutes = 1
    rows = []
    for detector in setup.detectors:
        for method in setup.methods:
            surprise, counter, cutoff, length, _ = tuned_detector(setup, minutes, detector, "empirical", method)
            n_train = int(round(0.70 * len(setup.values)))
            starts = window_starts(setup.stamps, length, n_train, len(setup.values), setup.step_seconds)
            starts = np.setdiff1d(starts, calibration_half(setup, length))
            alarm, count = counter(surprise(windows(setup.values, starts, length), minutes))
            state = occupancy_state(occupancy, starts + length - 1, EVENT_LOOKBACK)
            for name in ("empty", "occupied", "people came or went"):
                here = state == name
                fired = alarm & here
                rows.append(
                    {
                        "detector / stage 2": f"{detector} / {method} (cutoff {cutoff})",
                        "state": name,
                        "readings": int(here.sum()),
                        "alarm rate": float(fired.sum() / max(here.sum(), 1)),
                        "called lie": float(np.mean(count[fired] < cutoff)) if fired.any() else float("nan"),
                    }
                )
    table = pd.DataFrame(rows).set_index(["detector / stage 2", "state"])
    print()
    print("=" * 78)
    print(f"Clean test readings (1 step) never used for lines or cutoff, by what the people were doing "
          f"(change within the last {EVENT_LOOKBACK} readings, about 5 minutes).")
    print("Empirical lines. 'called lie' is the share of alarms the cutoff calls a lie rather than noise.")
    print(table.to_string(float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
