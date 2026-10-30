"""Second check by setting sensors aside: how many must go before the rest looks clean?

The per-sensor check in ``monitor.py`` asks whether each sensor's 8 numbers
look normal on their own. A sensor that lies by showing a real reading taken
from another moment passes that test: its numbers are perfectly ordinary,
they just disagree with what the other 15 sensors say. The per-sensor check
finds zero hot sensors and the alarm is thrown away.

Here the question is about agreement instead. Start with the all-128
distance ``d``. For each sensor ``k``, how much of ``d`` goes away when
sensor ``k`` is left out? With the precision matrix ``P`` and
``w = P (x - mu)``, the identity

    d = d_without_k + w_k^T (P_kk)^-1 w_k

gives that amount, the conditional score of sensor ``k``, without refitting
anything. Sensors are set aside in order of their conditional score until the
remaining ones pass their own line. The count is the number set aside:

- 0: the reading passed the all-128 line, no alarm.
- 1 or 2: one or two sensors disagree with the rest, the shape of a lie. The
  sensors set aside are the suspects.
- many: nothing short of dropping most sensors explains the reading, the
  shape of noise.

The line for the remaining sensors depends on how many are left. With
``chi2`` lines it is the chi-square percent point for the remaining degrees
of freedom. With empirical lines it is a quantile of clean calibration rows
with their own most suspicious sensors set aside in the same way.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from monitor import BLOCK_SIZE, N_BLOCKS, RIDGE, SYSTEM_FPR
from thresholds import DEFAULT_TARGET

# Rows are solved in chunks so the stacked sub-matrices stay small in memory.
CHUNK = 512


class SensorRemovalCheck:
    """Count the sensors that must be set aside before a reading looks clean."""

    def __init__(
        self,
        n_blocks: int = N_BLOCKS,
        block_size: int = BLOCK_SIZE,
        system_fpr: float = SYSTEM_FPR,
        ridge: float = RIDGE,
    ) -> None:
        if n_blocks < 2 or block_size < 1:
            raise ValueError("Need at least two blocks and a positive block size")
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.n_features = n_blocks * block_size
        self.system_fpr = system_fpr
        self.ridge = ridge
        # lines[m] is the line for the reading with m sensors set aside.
        # Setting aside every sensor always passes.
        self.chi2_lines = self._chi2_lines()
        self.lines = self.chi2_lines.copy()

        self.mu_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.cov_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None
        self.block_precision_inv_: np.ndarray | None = None

    def fit(self, x_train: np.ndarray) -> "SensorRemovalCheck":
        """Estimate the standardized covariance and its inverse from clean rows."""
        x_train = np.asarray(x_train, dtype=np.float64)
        if x_train.ndim != 2 or x_train.shape[1] != self.n_features:
            raise ValueError(f"X_train must have shape (n, {self.n_features})")
        if x_train.shape[0] <= self.n_features:
            raise ValueError("Need more calibration rows than features")
        self.mu_ = x_train.mean(axis=0)
        self.scale_ = np.maximum(x_train.std(axis=0), 1e-12)
        z = (x_train - self.mu_) / self.scale_
        self.cov_ = np.cov(z, rowvar=False) + self.ridge * np.eye(self.n_features)
        self.precision_ = np.linalg.inv(self.cov_)
        bs = self.block_size
        self.block_precision_inv_ = np.stack(
            [np.linalg.inv(self.precision_[k * bs : (k + 1) * bs, k * bs : (k + 1) * bs]) for k in range(self.n_blocks)]
        )
        return self

    def set_lines(self, lines: np.ndarray | None) -> "SensorRemovalCheck":
        """Use the given lines (one per number set aside), or the chi-square lines when ``None``."""
        if lines is None:
            self.lines = self.chi2_lines.copy()
            return self
        lines = np.asarray(lines, dtype=np.float64)
        if lines.shape != (self.n_blocks + 1,):
            raise ValueError(f"lines must have shape ({self.n_blocks + 1},)")
        self.lines = lines
        return self

    def empirical_lines(self, x_clean: np.ndarray, target: float = DEFAULT_TARGET) -> np.ndarray:
        """Lines from clean rows: for each m, about ``target`` of clean rows exceed it.

        For ``m`` sensors set aside, the clean reference is the same rows with
        their own ``m`` most suspicious sensors set aside, exactly as a test
        row is treated. Lines never rise as more sensors are set aside.
        """
        if not 0.0 < target < 1.0:
            raise ValueError("target must lie strictly between 0 and 1")
        w, d = self._whiten(x_clean)
        order = np.argsort(-self._conditional(w), axis=1, kind="stable")
        lines = np.empty(self.n_blocks + 1, dtype=np.float64)
        lines[0] = np.quantile(d, 1.0 - target, method="higher")
        for m in range(1, self.n_blocks):
            remainder = self._remainder(w, d, order[:, :m])
            lines[m] = np.quantile(remainder, 1.0 - target, method="higher")
        lines[: self.n_blocks] = np.minimum.accumulate(lines[: self.n_blocks])
        lines[self.n_blocks] = np.inf
        return lines

    def distances(self, frames: np.ndarray) -> np.ndarray:
        """All-128 squared distance per row."""
        return self._whiten(frames)[1]

    def conditional_scores(self, frames: np.ndarray) -> np.ndarray:
        """How much of the distance each sensor carries, shape ``(n_rows, n_blocks)``."""
        w, _ = self._whiten(frames)
        return self._conditional(w)

    def removal(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sensors set aside per row, and every row's sensors in suspicion order.

        Returns ``(counts, order)``. ``counts[i]`` is 0 when row ``i`` passes
        the all-128 line. Otherwise the suspects are ``order[i, :counts[i]]``.
        """
        w, d = self._whiten(frames)
        order = np.argsort(-self._conditional(w), axis=1, kind="stable")
        counts = np.full(len(d), self.n_blocks, dtype=np.int64)
        active = d > self.lines[0]
        counts[~active] = 0
        for m in range(1, self.n_blocks):
            rows = np.flatnonzero(active)
            if rows.size == 0:
                break
            remainder = self._remainder(w[rows], d[rows], order[rows, :m])
            passed = remainder <= self.lines[m]
            counts[rows[passed]] = m
            active[rows[passed]] = False
        return counts, order

    def counts(self, frames: np.ndarray) -> np.ndarray:
        """Number of sensors set aside per row, 0 when there is no alarm."""
        return self.removal(frames)[0]

    def _chi2_lines(self) -> np.ndarray:
        lines = np.empty(self.n_blocks + 1, dtype=np.float64)
        for m in range(self.n_blocks):
            lines[m] = chi2.ppf(1.0 - self.system_fpr, df=(self.n_blocks - m) * self.block_size)
        lines[self.n_blocks] = np.inf
        return lines

    def _whiten(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.precision_ is None or self.mu_ is None or self.scale_ is None:
            raise RuntimeError("SensorRemovalCheck must be fit on X_train before inference")
        frames = np.asarray(frames, dtype=np.float64)
        if frames.ndim != 2 or frames.shape[1] != self.n_features:
            raise ValueError(f"frames must have shape (n, {self.n_features})")
        z = (frames - self.mu_) / self.scale_
        w = z @ self.precision_
        return w, np.sum(w * z, axis=1)

    def _conditional(self, w: np.ndarray) -> np.ndarray:
        assert self.block_precision_inv_ is not None
        wb = w.reshape(-1, self.n_blocks, self.block_size)
        return np.einsum("nbi,bij,nbj->nb", wb, self.block_precision_inv_, wb)

    def _remainder(self, w: np.ndarray, d: np.ndarray, removed: np.ndarray) -> np.ndarray:
        """Distance of each row with the sensors in ``removed[i]`` left out.

        All rows remove the same number of sensors ``m``. For small ``m`` the
        removed part is solved through the precision matrix; for large ``m``
        the kept part is solved directly through the covariance, so no system
        is ever larger than half the features.
        """
        assert self.precision_ is not None and self.cov_ is not None and self.mu_ is not None
        n, m = removed.shape
        bs = self.block_size
        offsets = np.arange(bs)
        out = np.empty(n, dtype=np.float64)
        if 2 * m <= self.n_blocks:
            for start in range(0, n, CHUNK):
                part = slice(start, start + CHUNK)
                cols = (removed[part, :, None] * bs + offsets).reshape(-1, m * bs)
                wi = np.take_along_axis(w[part], cols, axis=1)
                sub = self.precision_[cols[:, :, None], cols[:, None, :]]
                explained = np.sum(wi * np.linalg.solve(sub, wi[..., None])[..., 0], axis=1)
                out[part] = d[part] - explained
            return out

        # Few sensors left: z_kept^T (cov_kept)^-1 z_kept, with z = cov w.
        keep_mask = np.ones((n, self.n_blocks), dtype=bool)
        np.put_along_axis(keep_mask, removed, False, axis=1)
        kept = np.nonzero(keep_mask)[1].reshape(n, self.n_blocks - m)
        z = w @ self.cov_
        for start in range(0, n, CHUNK):
            part = slice(start, start + CHUNK)
            cols = (kept[part, :, None] * bs + offsets).reshape(-1, (self.n_blocks - m) * bs)
            zi = np.take_along_axis(z[part], cols, axis=1)
            sub = self.cov_[cols[:, :, None], cols[:, None, :]]
            out[part] = np.sum(zi * np.linalg.solve(sub, zi[..., None])[..., 0], axis=1)
        return out
