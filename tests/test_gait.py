import numpy as np
import pandas as pd

from run_gait_monitor import CHANNELS, POINTS, stack_walks
from timeseries import window_starts


def _fake_walks():
    index = pd.MultiIndex.from_product(
        [[1, 2], [1, 2], [1, 2], range(POINTS)], names=["subject", "condition", "replication", "time"]
    )
    values = np.arange(len(index) * len(CHANNELS), dtype=float).reshape(len(index), len(CHANNELS))
    return pd.DataFrame(values, index=index, columns=list(CHANNELS))


def test_stack_walks_order_and_gaps():
    walks = _fake_walks()
    values, stamps = stack_walks(walks, [(1, 1), (1, 2)])
    assert values.shape == (4 * POINTS, len(CHANNELS))
    first = walks.xs((1, 1, 1), level=("subject", "condition", "replication")).to_numpy()
    np.testing.assert_array_equal(values[:POINTS], first)
    starts = window_starts(stamps, 5, 0, len(values), 1.0)
    assert len(starts) == 4 * (POINTS - 4)  # no window spans two walks
