import numpy as np
import pytest

from data_loader import default_dataset_dir, load_nominal_dataset, split_random_indices, split_temporal_indices

pytestmark = pytest.mark.skipif(not default_dataset_dir().is_dir(), reason="Dataset/ not present")


@pytest.fixture(scope="module")
def corpus():
    return load_nominal_dataset(return_batches=True)


def test_shapes(corpus):
    x, gases, batches = corpus
    assert x.shape[1] == 128
    assert len(x) == len(gases) == len(batches)
    assert set(np.unique(batches)) == set(range(1, 11))


def test_random_split_is_a_stratified_partition(corpus):
    _, gases, _ = corpus
    train, test = split_random_indices(gases, seed=42)
    assert len(np.intersect1d(train, test)) == 0
    assert len(train) + len(test) == len(gases)
    assert abs(len(train) / len(gases) - 0.70) < 0.01


def test_temporal_split(corpus):
    _, _, batches = corpus
    train, test = split_temporal_indices(batches)
    assert batches[train].max() == 7
    assert batches[test].min() == 8
