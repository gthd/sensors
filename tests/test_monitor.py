import numpy as np

from monitor import BlockMahalanobisMonitor, FullMahalanobis


def test_full_distance_matches_raw_formula(factor_data):
    train, test, _ = factor_data
    full = FullMahalanobis(ridge=0.0).fit(train)
    residual = test - train.mean(axis=0)
    raw = np.einsum("ij,jk,ik->i", residual, np.linalg.inv(np.cov(train, rowvar=False)), residual)
    np.testing.assert_allclose(full.distances(test), raw, rtol=1e-6)
    assert full.distance(test[0]) == full.distances(test[:1])[0]


def test_block_batch_matches_single_rows(factor_data):
    train, test, _ = factor_data
    blocks = BlockMahalanobisMonitor().fit(train)
    batch = blocks.block_distances_batch(test[:20])
    single = np.stack([blocks.block_distances(row) for row in test[:20]])
    np.testing.assert_allclose(batch, single)
    np.testing.assert_array_equal(blocks.hot_sensors(test[:20]), np.sum(single > blocks.tau, axis=1))


def test_per_block_lines(factor_data):
    train, test, _ = factor_data
    blocks = BlockMahalanobisMonitor().fit(train)
    blocks.set_block_lines(np.full(16, np.inf))
    assert blocks.hot_sensors(test).sum() == 0
    blocks.set_block_lines(None)
    assert blocks.tau == blocks.chi2_tau
