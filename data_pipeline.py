"""Data pipeline for CICIDS2017: load, clean, normalize, split, window."""

from __future__ import annotations

import glob
import os
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURES: List[str] = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Fwd IAT Mean",
    "Bwd IAT Mean",
    "Packet Length Mean",
    "Packet Length Std",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
]


def load_cicids(
    data_dir: str,
    features: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Load CICIDS2017 CSVs from a directory and produce a clean DataFrame.

    Strips whitespace from column names, replaces +/-inf with NaN, drops rows
    with NaN in the selected feature subset, and adds two label columns:
    `label` (0 = BENIGN, 1 = attack) and `attack_type` (original string).
    """
    pattern = os.path.join(data_dir, "*.csv")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for path in paths:
        df = pd.read_csv(path, low_memory=False, encoding="latin-1")
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    if "Label" not in df.columns:
        raise KeyError("Expected a 'Label' column after stripping whitespace.")

    feature_list = list(features) if features is not None else DEFAULT_FEATURES
    missing = [c for c in feature_list if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected feature columns: {missing}")

    df["attack_type"] = df["Label"].astype(str).str.strip()
    df["label"] = (df["attack_type"].str.upper() != "BENIGN").astype(int)

    keep = feature_list + ["label", "attack_type"]
    df = df[keep].copy()

    df[feature_list] = df[feature_list].apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=feature_list, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def preprocess(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    features: Optional[Iterable[str]] = None,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit a StandardScaler on BENIGN rows and transform every sample.

    Returns the scaled feature matrix X, the binary label vector y, and the
    fitted scaler. If a scaler is provided, it is reused (no refit).
    """
    if features is not None:
        feature_list = list(features)
    else:
        feature_list = [c for c in df.columns if c not in {"label", "attack_type"}]

    X = df[feature_list].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int64)

    if scaler is None:
        normal_mask = y == 0
        if not normal_mask.any():
            raise ValueError("No BENIGN samples available to fit the scaler.")
        scaler = StandardScaler().fit(X[normal_mask])

    X_scaled = scaler.transform(X)
    return X_scaled, y, scaler


def train_test_split_topo(
    X: np.ndarray,
    y: np.ndarray,
    attack_types: Optional[np.ndarray] = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Topology-friendly split.

    - X_train: BENIGN-only (so the detector sees a clean baseline).
    - X_val:   BENIGN-only subset, used to calibrate a threshold.
    - X_test:  Stratified mix of BENIGN + attack rows for evaluation.
    """
    rng = np.random.default_rng(random_state)
    n = len(X)
    if y.shape[0] != n:
        raise ValueError("X and y length mismatch.")
    if attack_types is not None and len(attack_types) != n:
        raise ValueError("attack_types length mismatch.")
    if not (0.0 <= val_ratio < 1.0) or not (0.0 <= test_ratio < 1.0):
        raise ValueError("val_ratio and test_ratio must be in [0, 1).")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be less than 1.")

    normal_idx = np.flatnonzero(y == 0)
    attack_idx = np.flatnonzero(y == 1)
    if len(normal_idx) == 0:
        raise ValueError("No BENIGN samples to build train/val on.")

    normal_idx = rng.permutation(normal_idx)
    attack_idx = rng.permutation(attack_idx)

    n_normal_test = int(len(normal_idx) * test_ratio)
    n_normal_val = int(len(normal_idx) * val_ratio)
    if n_normal_test + n_normal_val >= len(normal_idx):
        raise ValueError("val_ratio + test_ratio leave no BENIGN data for training.")

    test_normal = normal_idx[:n_normal_test]
    val_normal = normal_idx[n_normal_test : n_normal_test + n_normal_val]
    train_normal = normal_idx[n_normal_test + n_normal_val :]

    test_idx = np.concatenate([test_normal, attack_idx])
    test_idx = rng.permutation(test_idx)

    out = {
        "X_train": X[train_normal],
        "X_val": X[val_normal],
        "X_test": X[test_idx],
        "y_test": y[test_idx],
        "train_idx": train_normal,
        "val_idx": val_normal,
        "test_idx": test_idx,
    }
    if attack_types is not None:
        out["attack_types_test"] = np.asarray(attack_types)[test_idx]
    return out


def extract_windows(
    X: np.ndarray,
    window_size: int = 50,
    step: int = 25,
) -> np.ndarray:
    """Slide a fixed-size window over X and stack into a (n_windows, window_size, n_features) array.

    Incomplete trailing windows are skipped. Returns an array with shape
    `(0, window_size, n_features)` when X is shorter than window_size.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2-D (samples x features).")
    if window_size <= 0 or step <= 0:
        raise ValueError("window_size and step must be positive.")

    n, f = X.shape
    if n < window_size:
        return np.empty((0, window_size, f), dtype=X.dtype)

    starts = range(0, n - window_size + 1, step)
    windows = np.stack([X[s : s + window_size] for s in starts], axis=0)
    return windows
