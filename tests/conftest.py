import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def factor_data():
    """Clean rows from a 4-factor model over 16 sensors of 8, so sensors agree with each other.

    Feature scales span several orders of magnitude, like the real corpus.
    """
    rng = np.random.default_rng(0)
    loadings = rng.normal(size=(4, 128))
    scales = 10.0 ** rng.uniform(-1, 4, size=128)

    def draw(n):
        factors = rng.normal(size=(n, 4))
        return (factors @ loadings + 0.3 * rng.normal(size=(n, 128))) * scales

    return draw(4000), draw(600), scales
