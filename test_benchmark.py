"""Tests for src/barcode/benchmark.py."""

from __future__ import annotations

import numpy as np
import pytest

from barcode.baselines import AutoencoderDetector, IsolationForestDetector
from barcode.benchmark import (
    FusedDetector,
    METRIC_COLS,
    RESULT_COLS,
    benchmark_models,
    make_window_labels,
    run_full_benchmark,
)
from barcode.topo_detector import TopoDetector


# --- make_window_labels ---------------------------------------------------

def test_make_window_labels_aggregates_max():
    y = np.array([0, 0, 1, 0, 0, 0, 0, 1])
    win_y, _ = make_window_labels(y, attack_types=None, window_size=4, step=2)
    # windows: [0:4]=[0,0,1,0]->1, [2:6]=[1,0,0,0]->1, [4:8]=[0,0,0,1]->1
    assert win_y.tolist() == [1, 1, 1]


def test_make_window_labels_pure_benign():
    y = np.zeros(10, dtype=int)
    win_y, _ = make_window_labels(y, None, window_size=5, step=5)
    assert win_y.tolist() == [0, 0]


def test_make_window_labels_attack_type_majority():
    y = np.array([0, 0, 1, 1, 0])
    atk = np.array(["BENIGN", "BENIGN", "DDoS", "DDoS", "BENIGN"])
    win_y, win_a = make_window_labels(y, atk, window_size=5, step=5)
    assert win_y.tolist() == [1]
    assert win_a.tolist() == ["DDoS"]


def test_make_window_labels_too_short_returns_empty():
    y = np.zeros(5, dtype=int)
    win_y, win_a = make_window_labels(y, None, window_size=10, step=5)
    assert win_y.shape == (0,)
    assert win_a is None


def test_make_window_labels_rejects_bad_inputs():
    y = np.zeros(10, dtype=int)
    with pytest.raises(ValueError):
        make_window_labels(y, None, window_size=0, step=5)
    with pytest.raises(ValueError):
        make_window_labels(y.reshape(2, 5), None, window_size=5, step=5)
    with pytest.raises(ValueError):
        make_window_labels(y, np.array(["BENIGN"] * 9), window_size=5, step=5)


class _DummyWindowDetector:
    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=np.float64)

    def fit(self, X, y=None):
        return self

    def score_samples(self, X):
        return self.scores


def test_fused_detector_fit_threshold_validates_inputs():
    det = FusedDetector(
        topo=_DummyWindowDetector([0.1]),
        gnn=_DummyWindowDetector([0.2]),
    )
    with pytest.raises(ValueError):
        det.fit_threshold(np.zeros((1, 2)), percentile=101.0)

    empty = FusedDetector(
        topo=_DummyWindowDetector([]),
        gnn=_DummyWindowDetector([]),
    )
    with pytest.raises(ValueError):
        empty.fit_threshold(np.zeros((10, 2)), percentile=95.0)


# --- benchmark_models -----------------------------------------------------

@pytest.fixture(scope="module")
def split():
    rng = np.random.default_rng(0)
    X_train = rng.normal(0, 1, size=(400, 5))
    X_val = rng.normal(0, 1, size=(120, 5))
    # 200 benign + 200 attack
    X_test_benign = rng.normal(0, 1, size=(200, 5))
    X_test_attack = rng.normal(4, 1.5, size=(200, 5))
    X_test = np.vstack([X_test_benign, X_test_attack])
    y_test = np.concatenate([np.zeros(200, dtype=int), np.ones(200, dtype=int)])
    atk = np.concatenate([
        np.array(["BENIGN"] * 200),
        np.array(["DDoS"] * 100 + ["PortScan"] * 100),
    ])
    return {
        "X_train": X_train.astype(np.float64),
        "X_val": X_val.astype(np.float64),
        "X_test": X_test.astype(np.float64),
        "y_test": y_test,
        "attack_types_test": atk,
    }


def test_benchmark_models_returns_expected_columns(split):
    detectors = {
        "TopoDetector": TopoDetector(window_size=40, step=20, n_jobs=1),
        "IsolationForest": IsolationForestDetector(n_estimators=20),
        "Autoencoder": AutoencoderDetector(n_epochs=3, batch_size=64, device="cpu"),
    }
    df = benchmark_models(detectors, split, window_size=40, step=20)
    assert list(df.columns) == RESULT_COLS
    assert set(df["model"].unique()) == {"TopoDetector", "IsolationForest", "Autoencoder"}
    # Each model should produce an "overall" row plus one per attack type.
    overall_rows = df[df["attack_type"] == "overall"]
    assert len(overall_rows) == 3


def test_benchmark_metrics_in_valid_ranges(split):
    detectors = {"IsolationForest": IsolationForestDetector(n_estimators=20)}
    df = benchmark_models(detectors, split, window_size=40, step=20)
    finite = df[METRIC_COLS].to_numpy()
    finite = finite[~np.isnan(finite)]
    assert (finite >= 0.0).all() and (finite <= 1.0).all()


# --- run_full_benchmark (integration via synthetic CICIDS dir) ------------

def test_benchmark_with_gnn_and_fused(split):
    from barcode.benchmark import FusedDetector
    from barcode.gnn_model import GNNDetector

    detectors = {
        "TopoDetector": TopoDetector(window_size=40, step=20, n_jobs=1),
        "GNN": GNNDetector(window_size=40, step=20, n_epochs=2, batch_size=8, device="cpu"),
        "Fused": FusedDetector(
            topo=TopoDetector(window_size=40, step=20, n_jobs=1),
            gnn=GNNDetector(window_size=40, step=20, n_epochs=2, batch_size=8, device="cpu"),
            alpha=0.5,
        ),
    }
    df = benchmark_models(detectors, split, window_size=40, step=20)
    assert set(df["model"].unique()) == {"TopoDetector", "GNN", "Fused"}
    # Each model still produces an "overall" row.
    overall = df[df["attack_type"] == "overall"]
    assert len(overall) == 3


def test_run_full_benchmark_writes_csv(synthetic_cicids_dir, tmp_path):
    # Use lightweight detectors so the integration test is fast.
    from barcode.benchmark import default_detectors  # local import to avoid top-of-file noise

    detectors = {
        "IsolationForest": IsolationForestDetector(n_estimators=20),
    }
    df = run_full_benchmark(
        data_dir=synthetic_cicids_dir,
        output_dir=str(tmp_path),
        window_size=30,
        step=15,
        detectors=detectors,
    )
    assert (tmp_path / "full_benchmark.csv").exists()
    assert list(df.columns) == RESULT_COLS
    assert (df["model"] == "IsolationForest").any()
