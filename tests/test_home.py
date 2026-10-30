import numpy as np
import pandas as pd

from run_home_monitor import ROOMS, recent_rooms
from timeseries import window_starts


def test_window_starts_allow_jitter_but_not_gaps():
    offsets = np.array([0, 19.6, 40.0, 60.3, 80.1, 140.0, 160.2, 180.0])
    stamps = np.datetime64("2020-01-01T00:00:00") + (offsets * 1e6).astype("timedelta64[us]")
    np.testing.assert_array_equal(window_starts(stamps, 3, 0, len(stamps), 20.0), [0, 1, 2, 5])


def test_recent_rooms_uses_arrivals_within_lookback():
    t0 = np.datetime64("2020-01-01T12:00")
    events = pd.DataFrame(
        {
            "datetime": [t0, t0 + np.timedelta64(5, "m"), t0 + np.timedelta64(30, "m")],
            "room": ["Kitchen", "Kitchen", "Bedroom"],
            "on": [1.0, 0.0, 1.0],
        }
    )
    times = t0 + np.array([-1, 9, 11, 35], dtype="timedelta64[m]")
    seen = recent_rooms(times, events, np.timedelta64(10, "m"))
    assert list(seen.columns) == list(ROOMS)
    assert seen["Kitchen"].tolist() == [False, True, False, False]
    assert seen["Bedroom"].tolist() == [False, False, False, True]
