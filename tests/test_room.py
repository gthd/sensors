import numpy as np

from run_room_monitor import CHANNELS, occupancy_state


def test_channels_leave_out_derived_and_switch_sensors():
    assert len(CHANNELS) == 13
    assert not any("Slope" in c or "PIR" in c for c in CHANNELS)


def test_occupancy_state():
    occupancy = np.array([0, 0, 0, 0, 2, 2, 2, 2, 2, 2])
    state = occupancy_state(occupancy, np.array([3, 4, 6, 9]), lookback=3)
    assert state.tolist() == ["empty", "people came or went", "people came or went", "occupied"]
