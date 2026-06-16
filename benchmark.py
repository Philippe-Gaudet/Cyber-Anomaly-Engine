"""End-to-end benchmark of detectors on CICIDS2017."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from barcode.baselines import (
    AutoencoderDetector,
    IsolationForestDetector,
    evaluate_detector,
)
from barcode.data_pipeline import (
    extract_windows,
    load_cicids,
    preprocess,
    train_test_split_topo,
)
from barcode.gnn_model import GNNDetector, fuse_scores
from barcode.topo_detector import TopoDetector


class FusedDetector:
    """Late-fusion of `TopoDetector` and `GNNDetector` scores.

    Fits both children on the same data, then combines their per-window scores
    via `fuse_scores(topo, gnn, alpha)`. Exposes the same interface as the
    other detectors so the benchmark can treat it uniformly.
    """

    def __init__(
        self,
        topo: "TopoDetector",
        gnn: "GNNDetector",
        alpha: float = 0.5,
        threshold: Optional[float] = None,
    ) -> None:
        self.topo = topo
        self.gnn = gnn
        self.alpha = alpha
        self.threshold = threshold

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "FusedDetector":
        self.topo.fit(X)
        self.gnn.fit(X, y)
        self.threshold_ = self.threshold
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        topo_scores = self.topo.score_samples(X)
        gnn_scores = self.gnn.score_samples(X)
        # Both produce one score per sliding window with matching window_size/step,
        # so shapes line up by construction.
        return fuse_scores(topo_scores, gnn_scores, alpha=self.alpha)

    def fit_threshold(self, X_val: np.ndarray, percentile: float = 95.0) -> float:
        if not (0.0 <= percentile <= 100.0):
            raise ValueError("percentile must be in [0, 100].")
        scores = self.score_samples(X_val)
        if scores.size == 0:
            raise ValueError("Validation set produced no windows; cannot calibrate.")
        self.threshold_ = float(np.percentile(scores, percentile))
        return self.threshold_

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        thr = threshold if threshold is not None else self.threshold_
        if thr is None:
            raise ValueError("No threshold available. Call fit_threshold(...) first.")
        return (self.score_samples(X) > thr).astype(np.int64)


METRIC_COLS = ["auc_roc", "f1", "precision", "recall", "avg_precision"]
RESULT_COLS = ["model", "attack_type"] + METRIC_COLS


def make_window_labels(
    y: np.ndarray,
    attack_types: Optional[np.ndarray],
    window_size: int,
    step: int,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Aggregate per-sample labels into per-window labels.

    A window is flagged as attack if any sample in it is an attack. The
    window's attack_type is set to the most frequent non-BENIGN label inside,
    or "BENIGN" if no attack sample is present.
    """
    if window_size <= 0 or step <= 0:
        raise ValueError("window_size and step must be positive.")

    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be 1-D.")

    n = len(y)
    if attack_types is not None:
        attack_types = np.asarray(attack_types)
        if attack_types.shape[0] != n:
            raise ValueError("attack_types length mismatch.")

    if n < window_size:
        empty_y = np.empty((0,), dtype=np.int64)
        empty_a = np.empty((0,), dtype=object) if attack_types is not None else None
        return empty_y, empty_a

    starts = range(0, n - window_size + 1, step)
    win_y = np.array([int(y[s : s + window_size].max()) for s in starts], dtype=np.int64)

    win_a: Optional[np.ndarray] = None
    if attack_types is not None:
        labels: list[str] = []
        for s in starts:
            slc = attack_types[s : s + window_size]
            non_benign = [str(x) for x in slc if str(x).upper() != "BENIGN"]
            if not non_benign:
                labels.append("BENIGN")
            else:
                vals, counts = np.unique(non_benign, return_counts=True)
                labels.append(str(vals[counts.argmax()]))
        win_a = np.asarray(labels, dtype=object)

    return win_y, win_a


def _result_rows(model_name: str, eval_result: dict) -> list[dict]:
    rows = [{"model": model_name, "attack_type": "overall", **eval_result["overall"]}]
    for atype, metrics in eval_result["per_attack_type"].items():
        rows.append({"model": model_name, "attack_type": atype, **metrics})
    return rows


def benchmark_models(
    detectors: dict,
    split: dict,
    window_size: int,
    step: int,
    threshold_percentile: float = 95.0,
) -> pd.DataFrame:
    """Fit + evaluate each detector against the same train/val/test split.

    `detectors` is a mapping of display-name -> instantiated detector. Window-based
    detectors (currently `TopoDetector`) are handed window-aligned labels; everyone
    else gets sample-level `y_test` / `attack_types_test` directly.
    """
    X_train = split["X_train"]
    X_val = split["X_val"]
    X_test = split["X_test"]
    y_test = split["y_test"]
    attack_types_test = split.get("attack_types_test")

    win_y_test, win_atypes_test = make_window_labels(
        y_test, attack_types_test, window_size=window_size, step=step
    )

    rows: list[dict] = []
    for name, det in detectors.items():
        det.fit(X_train)
        det.fit_threshold(X_val, percentile=threshold_percentile)

        if isinstance(det, (TopoDetector, GNNDetector, FusedDetector)):
            eval_result = evaluate_detector(det, X_test, win_y_test, win_atypes_test)
        else:
            eval_result = evaluate_detector(det, X_test, y_test, attack_types_test)

        rows.extend(_result_rows(name, eval_result))

    df = pd.DataFrame(rows, columns=RESULT_COLS)
    return df


def default_detectors(window_size: int, step: int, n_jobs: int = 1) -> dict:
    """Full detector roster: Topo, IF, AE, GNN, Fused."""
    topo = TopoDetector(window_size=window_size, step=step, n_jobs=n_jobs)
    gnn = GNNDetector(window_size=window_size, step=step, n_epochs=20, device="cpu")
    return {
        "TopoDetector": topo,
        "IsolationForest": IsolationForestDetector(n_estimators=200),
        "Autoencoder": AutoencoderDetector(n_epochs=50, batch_size=256, device="cpu"),
        "GNN": gnn,
        "Fused": FusedDetector(
            topo=TopoDetector(window_size=window_size, step=step, n_jobs=n_jobs),
            gnn=GNNDetector(window_size=window_size, step=step, n_epochs=20, device="cpu"),
            alpha=0.5,
        ),
    }


def run_full_benchmark(
    data_dir: str,
    output_dir: str = "data/results/",
    window_size: int = 50,
    step: int = 25,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    threshold_percentile: float = 95.0,
    n_jobs: int = 1,
    detectors: Optional[dict] = None,
) -> pd.DataFrame:
    """Load CICIDS2017, prepare splits, run benchmark, write `full_benchmark.csv`."""
    df = load_cicids(data_dir)
    X, y, _ = preprocess(df)
    split = train_test_split_topo(
        X,
        y,
        attack_types=df["attack_type"].to_numpy(),
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    if detectors is None:
        detectors = default_detectors(window_size=window_size, step=step, n_jobs=n_jobs)

    results = benchmark_models(
        detectors,
        split,
        window_size=window_size,
        step=step,
        threshold_percentile=threshold_percentile,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path / "full_benchmark.csv", index=False)
    return results
