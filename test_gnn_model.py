"""Tests for src/barcode/gnn_model.py."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from barcode.gnn_model import GNNDetector, TrafficGNN, fuse_scores
from barcode.graph_builder import windows_to_graphs


def test_traffic_gnn_forward_shape():
    rng = np.random.default_rng(0)
    windows = rng.normal(size=(3, 20, 6)).astype(np.float64)
    graphs = windows_to_graphs(windows, labels=np.array([0, 1, 0]))
    from torch_geometric.loader import DataLoader

    loader = DataLoader(graphs, batch_size=3, shuffle=False)
    net = TrafficGNN(n_features=6)
    for batch in loader:
        logits = net(batch.x, batch.edge_index, batch.batch)
        assert logits.shape == (3,)
        break


@pytest.fixture(scope="module")
def supervised_data():
    rng = np.random.default_rng(42)
    n = 300
    benign = rng.normal(0, 1, size=(n, 6))
    attack = rng.normal(5, 1.5, size=(n, 6))
    X = np.vstack([benign, attack])
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    # Shuffle so windows mix labels.
    perm = rng.permutation(2 * n)
    return X[perm].astype(np.float64), y[perm]


def test_gnn_detector_fits_and_scores(supervised_data):
    X, y = supervised_data
    det = GNNDetector(window_size=30, step=15, n_epochs=2, batch_size=8, device="cpu")
    det.fit(X, y)
    scores = det.score_samples(X)
    assert scores.shape[0] > 0
    assert (scores >= 0).all() and (scores <= 1).all()


def test_gnn_predict_binary_and_threshold(supervised_data):
    X, y = supervised_data
    det = GNNDetector(window_size=30, step=15, n_epochs=2, batch_size=8, device="cpu").fit(X, y)
    det.fit_threshold(X, percentile=80.0)
    preds = det.predict(X)
    assert set(np.unique(preds).tolist()) <= {0, 1}


def test_gnn_score_before_fit_raises():
    det = GNNDetector()
    with pytest.raises(RuntimeError):
        det.score_samples(np.zeros((100, 5)))


def test_gnn_fit_raises_with_too_few_samples():
    det = GNNDetector(window_size=50, step=25)
    with pytest.raises(ValueError):
        det.fit(np.zeros((10, 6)), np.zeros(10, dtype=int))


def test_gnn_fit_rejects_label_length_mismatch(supervised_data):
    X, y = supervised_data
    det = GNNDetector(window_size=30, step=15, n_epochs=1, batch_size=8, device="cpu")
    with pytest.raises(ValueError):
        det.fit(X, y[:-1])


# --- fuse_scores ----------------------------------------------------------

def test_fuse_scores_in_unit_interval():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 10, size=50)
    b = rng.normal(0, 1, size=50)
    fused = fuse_scores(a, b, alpha=0.5)
    assert (fused >= 0).all() and (fused <= 1).all()


def test_fuse_scores_alpha_extremes():
    a = np.array([0.0, 5.0, 10.0])
    b = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(fuse_scores(a, b, alpha=1.0), np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(fuse_scores(a, b, alpha=0.0), np.zeros(3))


def test_fuse_scores_shape_mismatch_raises():
    with pytest.raises(ValueError):
        fuse_scores(np.zeros(5), np.zeros(6))


def test_fuse_scores_bad_alpha_raises():
    with pytest.raises(ValueError):
        fuse_scores(np.zeros(3), np.zeros(3), alpha=1.5)
