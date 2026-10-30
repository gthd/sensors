"""Block-wise Mahalanobis monitor with an analytic chi-square threshold."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

N_FEATURES = 128
BLOCK_SIZE = 8
N_BLOCKS = N_FEATURES // BLOCK_SIZE
SYSTEM_FPR = 1e-3
RIDGE = 1e-6


class BlockMahalanobisMonitor:
    """Runtime detector for 128-dimensional frames grouped as 16 blocks of 8.

    Calibration uses clean data only. For each block the monitor stores the
    mean, the ridge-regularized covariance, and its inverse. The decision
    threshold is the chi-square percent point at the Bonferroni-corrected
    false-positive level. Nothing in that threshold is fit on held-out data.
    Because the readings are not Gaussian, scripts can swap in empirical
    lines from ``thresholds.py`` with ``set_block_lines``.

    A frame is normal when every block stays at or below the threshold.
    An offending frame is a physical fault when its excess Mahalanobis mass
    sits in the high-variance half of the spectrum, and a cyber attack when
    that mass sits in the low-variance half. The split is the midpoint of the
    8 degrees of freedom (4 and 4), not a tuned cutoff.
    """

    def __init__(
        self,
        n_blocks: int = N_BLOCKS,
        block_size: int = BLOCK_SIZE,
        system_fpr: float = SYSTEM_FPR,
        ridge: float = RIDGE,
    ) -> None:
        if n_blocks < 1 or block_size < 1:
            raise ValueError("n_blocks and block_size must be positive")
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.n_features = n_blocks * block_size
        self.system_fpr = system_fpr
        self.alpha_block = system_fpr / n_blocks
        self.ridge = ridge
        self.half = block_size // 2
        self.chi2_tau = float(chi2.ppf(1.0 - self.alpha_block, df=block_size))
        # Scalar chi-square line by default. ``set_block_lines`` can replace it
        # with one empirical line per block (an array of shape (n_blocks,)).
        self.tau: float | np.ndarray = self.chi2_tau

        self.means_: np.ndarray | None = None
        self.inv_covs_: np.ndarray | None = None
        self.eigvals_: np.ndarray | None = None
        self.eigvecs_: np.ndarray | None = None
        self.feature_std_: np.ndarray | None = None

    def fit(self, x_train: np.ndarray) -> "BlockMahalanobisMonitor":
        """Estimate per-block means and precision matrices from clean data."""
        x_train = np.asarray(x_train, dtype=np.float64)
        if x_train.ndim != 2 or x_train.shape[1] != self.n_features:
            raise ValueError(f"X_train must have shape (n, {self.n_features})")
        if x_train.shape[0] <= self.block_size:
            raise ValueError("Need more calibration rows than features in a block")

        blocks = x_train.reshape(x_train.shape[0], self.n_blocks, self.block_size)
        means = np.empty((self.n_blocks, self.block_size), dtype=np.float64)
        inv_covs = np.empty((self.n_blocks, self.block_size, self.block_size), dtype=np.float64)
        eigvals = np.empty((self.n_blocks, self.block_size), dtype=np.float64)
        eigvecs = np.empty((self.n_blocks, self.block_size, self.block_size), dtype=np.float64)
        eye = np.eye(self.block_size, dtype=np.float64)

        for b in range(self.n_blocks):
            xb = blocks[:, b, :]
            means[b] = xb.mean(axis=0)
            covariance = np.cov(xb, rowvar=False)
            regularized = covariance + self.ridge * eye
            inv_covs[b] = np.linalg.inv(regularized)
            evals, evecs = np.linalg.eigh(regularized)
            order = np.argsort(evals)[::-1]
            eigvals[b] = evals[order]
            eigvecs[b] = evecs[:, order]

        self.means_ = means
        self.inv_covs_ = inv_covs
        self.eigvals_ = eigvals
        self.eigvecs_ = eigvecs
        self.feature_std_ = x_train.std(axis=0)
        self._check_quadratic_form(x_train[0])
        return self

    def set_block_lines(self, lines: float | np.ndarray | None) -> "BlockMahalanobisMonitor":
        """Use the given per-block lines, or the chi-square line again when ``None``."""
        if lines is None:
            self.tau = self.chi2_tau
            return self
        lines = np.asarray(lines, dtype=np.float64)
        if lines.ndim == 0:
            self.tau = float(lines)
        elif lines.shape == (self.n_blocks,):
            self.tau = lines
        else:
            raise ValueError(f"lines must be a scalar or have shape ({self.n_blocks},)")
        return self

    def predict(self, frame: np.ndarray) -> int:
        """Classify one 128-dimensional frame as 0 (normal), 1 (fault), or 2 (attack)."""
        self._require_fit()
        assert self.means_ is not None
        assert self.inv_covs_ is not None
        assert self.eigvals_ is not None
        assert self.eigvecs_ is not None

        residual = np.asarray(frame, dtype=np.float64).reshape(self.n_blocks, self.block_size) - self.means_
        whitened = np.einsum("bij,bj->bi", self.inv_covs_, residual)
        distances = np.einsum("bi,bi->b", residual, whitened)
        offending = distances > self.tau
        if not np.any(offending):
            return 0

        coords = np.einsum("bji,bj->bi", self.eigvecs_, residual)
        contributions = (coords * coords) / self.eigvals_
        active = contributions[offending]
        physical = float(active[:, : self.half].sum())
        cyber = float(active[:, self.half :].sum())
        if physical >= cyber:
            return 1
        return 2

    def block_distances(self, frame: np.ndarray) -> np.ndarray:
        """Squared Mahalanobis distance of each block, shape ``(n_blocks,)``."""
        self._require_fit()
        assert self.means_ is not None
        assert self.inv_covs_ is not None
        residual = np.asarray(frame, dtype=np.float64).reshape(self.n_blocks, self.block_size) - self.means_
        whitened = np.einsum("bij,bj->bi", self.inv_covs_, residual)
        return np.einsum("bi,bi->b", residual, whitened)

    def _require_fit(self) -> None:
        if self.means_ is None or self.inv_covs_ is None:
            raise RuntimeError("Monitor must be fit on X_train before inference")

    def _check_quadratic_form(self, frame: np.ndarray) -> None:
        """The inverse-covariance form and the eigen-expansion must agree."""
        assert self.eigvecs_ is not None
        assert self.eigvals_ is not None
        assert self.means_ is not None
        residual = np.asarray(frame, dtype=np.float64).reshape(self.n_blocks, self.block_size) - self.means_
        quadratic = self.block_distances(frame)
        coords = np.einsum("bji,bj->bi", self.eigvecs_, residual)
        expanded = np.sum((coords * coords) / self.eigvals_, axis=1)
        if not np.allclose(quadratic, expanded, rtol=1e-5, atol=1e-5):
            raise RuntimeError("Mahalanobis quadratic form does not match its eigen-expansion")
