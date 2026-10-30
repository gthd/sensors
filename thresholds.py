"""Alarm lines set from clean data instead of the chi-square formula.

The chi-square lines assume Gaussian readings. The real readings are not
Gaussian, so the actual clean false-alarm rate was about 8-9% instead of the
intended 0.1%. Here every line is a quantile of clean calibration scores, so
the clean alarm rate is whatever ``target`` asks for.

A line sits halfway between the quantile and the next larger clean score,
not on the quantile itself. Coarse sensors (whole degrees, 0.06 degree
steps) produce many tied scores, and a line exactly on a tied value lets the
smallest nudge to any of those rows cross it. Halfway leaves the clean rate
unchanged, since no clean score lies in between.

Choose ``target`` with the calibration size in mind: a 0.1% line from 2,000
clean rows rests on two rows. The default of 1% rests on about twenty.
"""

from __future__ import annotations

import numpy as np

DEFAULT_TARGET = 0.01


def empirical_line(scores: np.ndarray, target: float = DEFAULT_TARGET) -> float:
    """One line: clean rows exceed it on about ``target`` of the time."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if not 0.0 < target < 1.0:
        raise ValueError("target must lie strictly between 0 and 1")
    return float(line_between(scores, 1.0 - target))


def empirical_lines(scores: np.ndarray, target: float = DEFAULT_TARGET, iters: int = 60) -> np.ndarray:
    """One line per column, jointly tuned.

    ``scores`` has shape (n_rows, n_columns), for example one Mahalanobis
    distance per sensor. Every column gets the same tail level alpha, and
    alpha is found by bisection so that "any column over its line" happens on
    at most ``target`` of the clean rows. This is the empirical version of the
    Bonferroni-corrected chi-square line, without the Gaussian assumption.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("scores must have shape (n_rows, n_columns)")
    if not 0.0 < target < 1.0:
        raise ValueError("target must lie strictly between 0 and 1")

    lo, hi = 0.0, target
    best = _lines_at(scores, lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lines = _lines_at(scores, mid)
        rate = float(np.mean(np.any(scores > lines, axis=1)))
        if rate <= target:
            lo, best = mid, lines
        else:
            hi = mid
    return best


def _lines_at(scores: np.ndarray, alpha: float) -> np.ndarray:
    return line_between(scores, 1.0 - alpha)


def line_between(scores: np.ndarray, level: float) -> np.ndarray:
    """Per column, halfway from the ``level`` quantile to the next larger score.

    Scores above the result are exactly the scores above the quantile, so the
    clean rate is the same; ties at the quantile no longer sit on the line.
    With no larger score the quantile itself is returned.
    """
    scores = np.asarray(scores, dtype=np.float64)
    q = np.quantile(scores, level, axis=0, method="higher")
    above = np.where(scores > q, scores, np.inf)
    nxt = above.min(axis=0)
    return np.where(np.isfinite(nxt), 0.5 * (q + nxt), q)
