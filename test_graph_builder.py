"""Tests for src/barcode/graph_builder.py."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from barcode.graph_builder import window_to_graph, windows_to_graphs


def test_window_to_graph_has_required_fields():
    rng = np.random.default_rng(0)
    window = rng.normal(size=(20, 6)).astype(np.float64)
    graph = window_to_graph(window, y=1)
    assert isinstance(graph, Data)
    assert graph.x.dim() == 2
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.shape[1] == 20  # one edge per flow
    assert graph.edge_attr.shape == (20, 6)
    assert graph.y.shape == (1,)
    assert graph.y.item() == 1.0


def test_window_to_graph_node_count_bounded(rng):
    window = rng.normal(size=(50, 4))
    g = window_to_graph(window, n_buckets=3)
    # 3*3 src + 3*3 dst buckets at most; in practice fewer if some buckets are empty.
    assert g.x.shape[0] <= 3 * 3 * 2
    assert g.x.shape[0] >= 1


def test_window_to_graph_rejects_empty():
    with pytest.raises(ValueError):
        window_to_graph(np.zeros((0, 4)))


def test_window_to_graph_rejects_1d():
    with pytest.raises(ValueError):
        window_to_graph(np.zeros(20))


def test_window_to_graph_rejects_bad_feat_idx():
    with pytest.raises(ValueError):
        window_to_graph(np.zeros((10, 3)), src_feat_idx=5)


def test_window_to_graph_rejects_bad_bucket_count():
    with pytest.raises(ValueError):
        window_to_graph(np.zeros((10, 3)), n_buckets=0)


def test_windows_to_graphs_matches_labels():
    rng = np.random.default_rng(1)
    windows = rng.normal(size=(4, 15, 5))
    labels = np.array([0, 1, 0, 1])
    graphs = windows_to_graphs(windows, labels=labels)
    assert len(graphs) == 4
    for g, y in zip(graphs, labels):
        assert g.y.item() == float(y)


def test_windows_to_graphs_label_length_mismatch():
    with pytest.raises(ValueError):
        windows_to_graphs(np.zeros((3, 10, 4)), labels=np.array([0, 1]))


def test_node_features_are_finite():
    rng = np.random.default_rng(2)
    window = rng.normal(size=(25, 7))
    g = window_to_graph(window)
    assert torch.isfinite(g.x).all()
    assert torch.isfinite(g.edge_attr).all()
