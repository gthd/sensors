"""Home of one elderly person: nine air sensors, one reading every 20 seconds.

Data: UCI "Single Elder Home Monitoring: Gas and Position", in ``DatasetHome/``
(``database_gas.csv`` and ``database_pos.csv``). Channels: temperature,
humidity, two CO2 sensors, four MOX gas sensors and a CO sensor, from
November 2019 to February 2020.

Unlike the oven, these sensors agree with each other: MOX2, MOX3 and MOX4 move
together (step correlation 0.6-0.7). So the removal check, which looks for
channels that disagree with the rest, has something to work with.

The same machinery as the oven (``timeseries.py``): surprises, blocks of W
steps per channel, hot or removal counting, fair scoring. Durations are one
step (20 seconds) and 30 steps (10 minutes).

After the corruption experiment, the clean test readings are split by where
the person was (``database_pos.csv``), to see whether alarms on clean data
are false alarms or real life: cooking, showering, airing a room.

--steps     1 | 30 | both (default)
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
from timeseries import (
    TRAIN_FRACTION,
    TREND_LAGS,
    FIT_WINDOWS,
    Setup,
    Surprise,
    fit_monitors,
    make_counter,
    run_experiment,
    set_lines,
    spread_out,
    window_starts,
    windows,
)

DATA_DIR = Path(__file__).resolve().parent / "DatasetHome"
CHANNELS = ("temperature", "humidity", "CO2CosIRValue", "CO2MG811Value", "MOX1", "MOX2", "MOX3", "MOX4", "COValue")
ROOMS = ("Living room", "Bedroom", "Bathroom", "Kitchen", "Hallway")
STEP_SECONDS = 20.0
NOISE_SCALES = (0.25, 0.50, 1.00, 2.00, 4.00)
LIE_COUNTS = (1, 2, 4)
ROOM_LOOKBACK = np.timedelta64(10, "m")
ROOM_WINDOWS = 20_000


def main() -> None:
    args = _parse_args()
    values, stamps = load_home()
    setup = Setup(
        values=values,
        stamps=stamps,
        step_seconds=STEP_SECONDS,
        step_name="step",
        noise_scales=NOISE_SCALES,
        lie_counts=LIE_COUNTS,
        durations=(1, 30) if args.steps == "both" else (int(args.steps),),
        detectors=("jump", "trend") if args.detector == "both" else (args.detector,),
        thresholds=("chi2", "empirical") if args.threshold == "both" else (args.threshold,),
        methods=("hot", "removal") if args.stage2 == "both" else (args.stage2,),
        target=args.target,
    )
    run_experiment(setup)
    _alarms_by_room(setup)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", choices=("1", "30", "both"), default="both")
    parser.add_argument("--detector", choices=("jump", "trend", "both"), default="both")
    parser.add_argument("--threshold", choices=("chi2", "empirical", "both"), default="both")
    parser.add_argument("--stage2", choices=("hot", "removal", "both"), default="both")
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET, help="clean alarm rate for empirical lines")
    return parser.parse_args()


def load_home() -> tuple[np.ndarray, np.ndarray]:
    """Readings in time order and their UTC time stamps (as naive datetime64)."""
    frame = pd.read_csv(DATA_DIR / "database_gas.csv")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
    frame = frame.sort_values("timestamp").dropna(subset=list(CHANNELS))
    stamps = frame["timestamp"].dt.tz_localize(None).to_numpy()
    return frame[list(CHANNELS)].to_numpy(dtype=np.float64), stamps


def load_positions() -> pd.DataFrame:
    """Presence events: one row per event, with the room and whether it switched on (1) or off (0)."""
    frame = pd.read_csv(DATA_DIR / "database_pos.csv")
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, format="ISO8601").dt.tz_localize(None)
    events = frame.melt(id_vars="datetime", var_name="room", value_name="on").dropna(subset=["on"])
    return events.sort_values("datetime").reset_index(drop=True)


def recent_rooms(times: np.ndarray, events: pd.DataFrame, lookback: np.timedelta64) -> pd.DataFrame:
    """For each time, whether each room saw someone arrive within ``lookback`` before it."""
    out = {}
    for room in ROOMS:
        arrivals = np.sort(events.loc[(events["room"] == room) & (events["on"] == 1), "datetime"].to_numpy())
        last = np.searchsorted(arrivals, times, side="right") - 1
        seen = np.full(len(times), False)
        ok = last >= 0
        seen[ok] = times[ok] - arrivals[last[ok]] <= lookback
        out[room] = seen
    return pd.DataFrame(out)


def _alarms_by_room(setup: Setup) -> None:
    """Alarm rate on clean test readings, split by where the person had just been."""
    values, stamps = setup.values, setup.stamps
    events = load_positions()
    n_train = int(round(TRAIN_FRACTION * len(values)))
    minutes = 1
    length = TREND_LAGS + minutes + 1
    train_starts = window_starts(stamps, length, 0, n_train, setup.step_seconds)
    test_starts = window_starts(stamps, length, n_train, len(values), setup.step_seconds)
    train_win = windows(values, spread_out(train_starts, FIT_WINDOWS), length)
    starts = spread_out(test_starts, 2 * ROOM_WINDOWS)
    cal_starts, eva_starts = starts[0::2], starts[1::2]
    cal, eva = windows(values, cal_starts, length), windows(values, eva_starts, length)
    times = stamps[eva_starts + length - 1]
    seen = recent_rooms(times, events, ROOM_LOOKBACK)

    print()
    print("=" * 78)
    print(f"Clean test readings (1 step), split by the rooms the person entered in the last "
          f"{int(ROOM_LOOKBACK / np.timedelta64(1, 'm'))} minutes.")
    print("Lines from clean calibration readings (empirical), so about "
          f"{setup.target:.1%} of all clean readings alarm by construction.")
    rows = []
    for detector in setup.detectors:
        surprise = Surprise(0 if detector == "jump" else TREND_LAGS).fit(train_win)
        monitors = fit_monitors(surprise(train_win, minutes), values.shape[1], minutes)
        set_lines(monitors, surprise(cal, minutes), "empirical", setup.target)
        for method in setup.methods:
            alarm = make_counter(monitors, method)(surprise(eva, minutes))[0]
            row = {"detector / stage 2": f"{detector} / {method}", "all": alarm.mean()}
            row["nobody moved"] = alarm[~seen.any(axis=1)].mean()
            for room in ROOMS:
                row[room] = alarm[seen[room].to_numpy()].mean()
            rows.append(row)
    table = pd.DataFrame(rows).set_index("detector / stage 2")
    shares = {"all": 1.0, "nobody moved": float((~seen.any(axis=1)).mean())}
    shares.update({room: float(seen[room].mean()) for room in ROOMS})
    print(table.to_string(float_format="{:.3f}".format))
    print("Share of readings in each column: " + ", ".join(f"{k} {v:.2f}" for k, v in shares.items()))


if __name__ == "__main__":
    main()
