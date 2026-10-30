"""Load the nominal sensor corpus and hold out a clean calibration split."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

N_FEATURES = 128
TRAIN_FRACTION = 0.70
# Batches 1-7 hold about 69% of the rows, the closest batch boundary to 70%.
TEMPORAL_TRAIN_BATCHES = 7


def default_dataset_dir() -> Path:
    return Path(__file__).resolve().parent / "Dataset"


def load_nominal_dataset(
    dataset_dir: Path | None = None,
    return_batches: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load every ``batch*.dat`` file as clean nominal readings.

    Each row is one 128-dimensional sensor vector stored in LIBSVM format.
    The leading token is a gas identifier from the source corpus. It is used
    to stratify the random split and to pick same-gas donors for quiet lies.
    It is not a monitor label.

    With ``return_batches=True`` a third array gives each row's batch number
    (1-10), which is the time order of the recordings.
    """
    root = Path(dataset_dir) if dataset_dir is not None else default_dataset_dir()
    if not root.is_dir():
        raise FileNotFoundError(f"Sensor dataset directory not found: {root}")

    paths = sorted(root.glob("batch*.dat"), key=lambda path: int(path.stem.replace("batch", "")))
    if not paths:
        raise FileNotFoundError(f"No batch*.dat files in {root}")

    features: list[np.ndarray] = []
    gas_ids: list[np.ndarray] = []
    batch_ids: list[np.ndarray] = []
    for path in paths:
        matrix, labels = load_svmlight_file(str(path), n_features=N_FEATURES, dtype=np.float64)
        features.append(np.asarray(matrix.toarray(), dtype=np.float64))
        gas_ids.append(np.asarray(labels, dtype=np.int64))
        batch_ids.append(np.full(len(labels), int(path.stem.replace("batch", "")), dtype=np.int64))

    x = np.vstack(features)
    gases = np.concatenate(gas_ids)
    batches = np.concatenate(batch_ids)
    if x.shape[1] != N_FEATURES:
        raise ValueError(f"Expected {N_FEATURES} features, found {x.shape[1]}")
    if return_batches:
        return x, gases, batches
    return x, gases


def split_random_indices(
    gas_ids: np.ndarray,
    train_fraction: float = TRAIN_FRACTION,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Shuffled, gas-stratified split. Returns (train_idx, test_idx)."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between 0 and 1")
    idx = np.arange(len(gas_ids))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=1.0 - train_fraction,
        random_state=seed,
        shuffle=True,
        stratify=gas_ids,
    )
    return np.asarray(train_idx), np.asarray(test_idx)


def split_temporal_indices(
    batch_ids: np.ndarray,
    train_batches: int = TEMPORAL_TRAIN_BATCHES,
) -> tuple[np.ndarray, np.ndarray]:
    """Early batches train, later batches test. Returns (train_idx, test_idx).

    This is what a deployed monitor faces: it learns from the past and runs
    on recordings made after the sensors have drifted.
    """
    train_idx = np.flatnonzero(batch_ids <= train_batches)
    test_idx = np.flatnonzero(batch_ids > train_batches)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise ValueError(f"train_batches={train_batches} leaves an empty side")
    return train_idx, test_idx


def split_nominal(
    x: np.ndarray,
    gas_ids: np.ndarray,
    train_fraction: float = TRAIN_FRACTION,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split clean readings into calibration data and a held-out nominal pool.

    ``train_fraction`` of the rows (default 70%) become ``X_train`` and are
    the only samples used to estimate block means and covariances. The
    remaining 30% is ``X_test``, the pool the evaluation stream is drawn from.
    Same partition as before: shuffling indices with the same seed and
    stratification gives the same rows as shuffling the matrix.
    """
    train_idx, test_idx = split_random_indices(gas_ids, train_fraction, seed)
    return np.asarray(x[train_idx], dtype=np.float64), np.asarray(x[test_idx], dtype=np.float64)
