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
then apply, ``BlockMahalanobisMonitor`` and ``SensorRemovalCheck``. The
machinery is shared with the home in ``timeseries.py``.

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

from thresholds import DEFAULT_TARGET
from timeseries import Setup, run_experiment

TEMP_LOW = 50.0
TEMP_HIGH = 800.0
COLUMNS = tuple(f"T_data_{chamber}_{sensor}" for chamber in range(1, 6) for sensor in range(1, 4))
OVEN_NOISE_SCALES = (0.10, 0.25, 0.50, 1.00, 2.00, 4.00)
THERMO_COUNTS = (1, 2, 4, 6)


def main() -> None:
    args = _parse_args()
    values, stamps = _load_oven()
    run_experiment(
        Setup(
            values=values,
            stamps=stamps,
            step_seconds=60.0,
            step_name="minute",
            noise_scales=OVEN_NOISE_SCALES,
            lie_counts=THERMO_COUNTS,
            durations=(1, 10) if args.minutes == "both" else (int(args.minutes),),
            detectors=("jump", "trend") if args.detector == "both" else (args.detector,),
            thresholds=("chi2", "empirical") if args.threshold == "both" else (args.threshold,),
            methods=("hot", "removal") if args.stage2 == "both" else (args.stage2,),
            target=args.target,
        )
    )


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


if __name__ == "__main__":
    main()
