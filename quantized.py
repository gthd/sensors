"""Fixed-point simulation of the online removal kernel.

The fit stays in float64. What would sit in flash — the standardized
vector ``z``, the precision ``P``, and the stored block inverses — is
quantized to Q15 (int16) or Q7 (int8). Each product is accumulated in
int64. A power-of-two scale, taken from the training vectors, maps each
matrix into the Q-format range. Pure Q15 cannot hold ``P``: its entries
reach the hundreds of thousands.

Q15 means 15 fractional bits and Q7 means 7. With scale ``2^e`` the
stored integer is ``round(value / 2^e * 2^{bits-1})``, clipped to the
signed range.
"""

from __future__ import annotations

import math

import numpy as np

from consistency import SensorRemovalCheck

INT64_MAX = np.iinfo(np.int64).max
INT64_MIN = np.iinfo(np.int64).min


def pow2_scale(absmax: float) -> float:
    """Smallest power of two that is at least ``absmax``. Zero maps to 1."""
    if absmax <= 0.0 or not math.isfinite(absmax):
        return 1.0
    exponent = math.ceil(math.log2(absmax))
    scale = math.ldexp(1.0, exponent)
    if scale < absmax:
        scale = math.ldexp(1.0, exponent + 1)
    if scale / 2.0 >= absmax:
        scale /= 2.0
    return float(scale)


def quantize(values: np.ndarray, bits: int, scale: float) -> tuple[np.ndarray, float]:
    """Quantize to a signed Q format. Returns int64 codes and the fraction clipped."""
    if bits < 2:
        raise ValueError("bits must include a sign bit and at least one magnitude bit")
    frac = bits - 1
    levels = 1 << frac
    qmax = levels - 1
    qmin = -levels
    scaled = np.asarray(values, dtype=np.float64) / scale * levels
    q = np.rint(scaled)
    clipped = float(np.mean((q < qmin) | (q > qmax))) if q.size else 0.0
    return np.clip(q, qmin, qmax).astype(np.int64), clipped


def _div_round(numerator: int, denominator: int) -> int:
    """Nearest integer division. Ties go away from zero."""
    if denominator == 0:
        raise ZeroDivisionError
    if denominator < 0:
        numerator, denominator = -numerator, -denominator
    half = denominator // 2
    if numerator >= 0:
        return (numerator + half) // denominator
    return -((-numerator + half) // denominator)


def solve_int64(matrix_q: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Rounded Gaussian elimination. Solves ``matrix_q @ y = rhs``.

    Products use arbitrary-width integers and every stored entry is rounded
    back to int64, which is the accumulator width. The returned ``y`` is the
    solution of that rounded system.
    """
    matrix = np.array(matrix_q, dtype=np.int64, copy=True)
    right = np.array(rhs, dtype=np.int64, copy=True)
    size = matrix.shape[0]
    if matrix.shape != (size, size) or right.shape != (size,):
        raise ValueError("solve_int64 expects a square system")

    def store(value: int) -> int:
        if value > INT64_MAX or value < INT64_MIN:
            raise OverflowError("int64 accumulator overflowed during elimination")
        return int(value)

    for pivot_index in range(size):
        column = matrix[pivot_index:, pivot_index]
        pivot_row = pivot_index + int(np.argmax(np.abs(column)))
        if matrix[pivot_row, pivot_index] == 0:
            return np.full(size, np.nan)
        if pivot_row != pivot_index:
            matrix[[pivot_index, pivot_row]] = matrix[[pivot_row, pivot_index]]
            right[[pivot_index, pivot_row]] = right[[pivot_row, pivot_index]]
        pivot = int(matrix[pivot_index, pivot_index])
        for row in range(pivot_index + 1, size):
            factor = int(matrix[row, pivot_index])
            if factor == 0:
                continue
            for col in range(pivot_index, size):
                matrix[row, col] = store(
                    int(matrix[row, col]) - _div_round(factor * int(matrix[pivot_index, col]), pivot)
                )
            right[row] = store(int(right[row]) - _div_round(factor * int(right[pivot_index]), pivot))

    solution = np.empty(size, dtype=np.float64)
    for row in range(size - 1, -1, -1):
        total = float(right[row])
        if row + 1 < size:
            total -= float(np.dot(matrix[row, row + 1 :].astype(np.float64), solution[row + 1 :]))
        pivot = float(matrix[row, row])
        if pivot == 0.0:
            return np.full(size, np.nan)
        solution[row] = total / pivot
    return solution


class QuantizedRemoval:
    """Online removal with ``z`` and ``P`` stored in a Q format."""

    def __init__(self, check: SensorRemovalCheck, bits: int, z_absmax: float) -> None:
        if check.precision_ is None or check.block_precision_inv_ is None or check.scale_ is None or check.mu_ is None:
            raise RuntimeError("Fit the float64 monitor before quantizing it")
        self.bits = bits
        self.frac = bits - 1
        self.n_blocks = check.n_blocks
        self.block_size = check.block_size
        self.n_features = check.n_features
        self.mu = check.mu_
        self.feature_scale = check.scale_
        self._check = check

        precision = 0.5 * (check.precision_ + check.precision_.T)
        self.z_scale = pow2_scale(z_absmax)
        self.p_scale = pow2_scale(float(np.max(np.abs(precision))))
        self.inv_scale = pow2_scale(float(np.max(np.abs(check.block_precision_inv_))))
        self.precision_q, self.p_clipped = quantize(precision, bits, self.p_scale)
        self.inv_q, self.inv_clipped = quantize(check.block_precision_inv_, bits, self.inv_scale)
        self.p_zero = float(np.mean(self.precision_q == 0))
        self.inv_zero = float(np.mean(self.inv_q == 0))
        self._w_unit = self.p_scale * self.z_scale / float(1 << (2 * self.frac))
        self._d_unit = self.p_scale * (self.z_scale ** 2) / float(1 << (3 * self.frac))
        self._inv_unit = self.inv_scale / float(1 << self.frac)
        self._rhs_shift = int(round(math.log2(self.z_scale))) - self.frac

    def standardize(self, frames: np.ndarray) -> np.ndarray:
        frames = np.asarray(frames, dtype=np.float64)
        if frames.ndim != 2 or frames.shape[1] != self.n_features:
            raise ValueError(f"frames must have shape (n, {self.n_features})")
        return (frames - self.mu) / self.feature_scale

    def quantize_z(self, frames: np.ndarray) -> tuple[np.ndarray, float]:
        return quantize(self.standardize(frames), self.bits, self.z_scale)

    def distances_and_scores(self, z_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Int64 matvec and the per-sensor conditional scores, both dequantized."""
        w_acc = z_q @ self.precision_q
        products = w_acc * z_q
        d_acc = products.sum(axis=1, dtype=np.int64)
        distances = d_acc.astype(np.float64) * self._d_unit
        scores = self._conditional_scores(w_acc)
        return distances, scores

    def removal(self, frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Counts and suspicion order, using the float64 lines already on the monitor."""
        z_q, _ = self.quantize_z(frames)
        w_acc = z_q @ self.precision_q
        products = w_acc * z_q
        distances = products.sum(axis=1, dtype=np.int64).astype(np.float64) * self._d_unit
        scores = self._conditional_scores(w_acc)
        order = np.argsort(-scores, axis=1, kind="stable")
        counts = np.full(frames.shape[0], self.n_blocks, dtype=np.int64)
        lines = self._check.lines
        active = distances > lines[0]
        counts[~active] = 0
        for removed_count in range(1, self.n_blocks):
            rows = np.flatnonzero(active)
            if rows.size == 0:
                break
            remainder = np.array(
                [
                    self._remainder_row(w_acc[row], distances[row], order[row, :removed_count], scores[row])
                    for row in rows
                ]
            )
            passed = np.isfinite(remainder) & (remainder <= lines[removed_count])
            counts[rows[passed]] = removed_count
            active[rows[passed]] = False
        return counts, order

    def _conditional_scores(self, w_acc: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(w_acc), axis=1)
        shifts = np.zeros(w_acc.shape[0], dtype=np.int64)
        nonzero = peak > 0
        # Leave bits-1 magnitude bits, so the shifted vector fits in the same Q format.
        shifts[nonzero] = np.floor(np.log2(peak[nonzero].astype(np.float64))).astype(np.int64) + 1 - self.frac
        shifts = np.maximum(shifts, 0)
        shifted = np.empty_like(w_acc)
        for shift in np.unique(shifts):
            rows = shifts == shift
            if shift == 0:
                shifted[rows] = w_acc[rows]
            else:
                shifted[rows] = np.rint(w_acc[rows] / float(1 << int(shift))).astype(np.int64)
        limit = (1 << self.frac) - 1
        shifted = np.clip(shifted, -(limit + 1), limit)
        blocks = shifted.reshape(-1, self.n_blocks, self.block_size)
        mixed = np.einsum("kij,nkj->nki", self.inv_q, blocks, optimize=True)
        accumulated = np.einsum("nki,nki->nk", mixed, blocks, optimize=True)
        restored = (2.0 ** (2.0 * shifts.astype(np.float64))) * (self._w_unit ** 2) * self._inv_unit
        return accumulated.astype(np.float64) * restored[:, None]

    def _remainder_row(
        self,
        w_row: np.ndarray,
        distance: float,
        removed: np.ndarray,
        scores_row: np.ndarray,
    ) -> float:
        """Distance after dropping ``removed``. One sensor uses the stored inverse."""
        if removed.size == 1:
            return float(distance - scores_row[int(removed[0])])
        columns = (removed[:, None] * self.block_size + np.arange(self.block_size)).ravel()
        block = self.precision_q[np.ix_(columns, columns)]
        right = _shift_rhs(w_row[columns], self._rhs_shift)
        try:
            solved = solve_int64(block, right)
        except OverflowError:
            return float("nan")
        if not np.all(np.isfinite(solved)):
            return float("nan")
        explained = float(np.dot(w_row[columns].astype(np.float64), solved)) * self._w_unit
        return float(distance - explained)


def _shift_rhs(values: np.ndarray, shift: int) -> np.ndarray:
    """Multiply an int64 vector by ``2^shift`` with rounding on a right shift."""
    out = np.empty(values.shape, dtype=np.int64)
    if shift >= 0:
        factor = 1 << shift
        wide = values.astype(object) * factor
        if np.any(wide > INT64_MAX) or np.any(wide < INT64_MIN):
            raise OverflowError("right-hand side does not fit in int64")
        return wide.astype(np.int64)
    divisor = 1 << (-shift)
    half = divisor // 2
    positive = values >= 0
    out[positive] = (values[positive] + half) // divisor
    out[~positive] = -((-values[~positive] + half) // divisor)
    return out
