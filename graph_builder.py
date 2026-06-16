"""Build PyG `Data` graphs from sliding windows of network-flow features.

CICIDS2017's preprocessed feature matrix doesn't expose IP/port pairs, so we
synthesize pseudo-endpoints by quantizing two feature dimensions into a small
grid of buckets. Each window row (flow) becomes an edge between its
source-bucket and destination-bucket. Node features are the mean of the
connected edge features. This keeps the GNN's signal driven by traffic
structure rather than identity, which is what the topological view also relies on.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data


def _quantize(values: np.ndarray, n_buckets: int) -> np.ndarray:
    """Map a 1-D vector to bucket indices [0, n_buckets) by linear binning."""
    if n_buckets <= 0:
        raise ValueError("n_buckets must be positive.")
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if hi - lo < 1e-12:
        return np.zeros_like(values, dtype=np.int64)
    norm = (values - lo) / (hi - lo)
    idx = np.clip((norm * n_buckets).astype(np.int64), 0, n_buckets - 1)
    return idx


def window_to_graph(
    window: np.ndarray,
    y: int = 0,
    src_feat_idx: int = 0,
    dst_feat_idx: int = 1,
    n_buckets: int = 4,
) -> Data:
    """Convert one (window_size, n_features) numpy window into a PyG `Data` graph.

    - Nodes: at most `n_buckets * n_buckets` synthesized endpoints (only those
      that appear in at least one flow are kept).
    - Edges: one per window row, from src-bucket to dst-bucket.
    - Edge attrs: the flow's feature vector.
    - Node features: mean of attrs of all incident edges.
    - y: window-level label (1 if attack present, else 0).
    """
    if window.ndim != 2:
        raise ValueError("window must be 2-D (window_size, n_features).")
    n_rows, n_feats = window.shape
    if n_rows == 0:
        raise ValueError("window has zero rows.")
    if not (0 <= src_feat_idx < n_feats and 0 <= dst_feat_idx < n_feats):
        raise ValueError("src/dst feature indices out of range.")

    src_buckets = _quantize(window[:, src_feat_idx], n_buckets)
    dst_buckets = _quantize(window[:, dst_feat_idx], n_buckets)

    # Compose a single int id from (src, dst) bucket coordinates.
    src_ids = src_buckets
    dst_ids = dst_buckets + n_buckets  # offset so src and dst don't collide

    # Collect unique node ids and remap to a contiguous [0, n_nodes).
    used_ids = np.concatenate([src_ids, dst_ids])
    unique_ids, inverse = np.unique(used_ids, return_inverse=True)
    n_nodes = len(unique_ids)

    src_idx = inverse[: n_rows]
    dst_idx = inverse[n_rows :]

    edge_index = torch.tensor(np.stack([src_idx, dst_idx], axis=0), dtype=torch.long)
    edge_attr = torch.tensor(window, dtype=torch.float32)

    # Node features = mean of features of all incident edges.
    x = np.zeros((n_nodes, n_feats), dtype=np.float32)
    counts = np.zeros((n_nodes,), dtype=np.int64)
    for i in range(n_rows):
        for node in (src_idx[i], dst_idx[i]):
            x[node] += window[i].astype(np.float32)
            counts[node] += 1
    counts = np.maximum(counts, 1)
    x = x / counts[:, None]

    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor([float(y)], dtype=torch.float32),
    )


def windows_to_graphs(
    windows: np.ndarray,
    labels: Optional[np.ndarray] = None,
    **kwargs,
) -> list[Data]:
    """Vectorized convenience over `window_to_graph`."""
    if windows.ndim != 3:
        raise ValueError("windows must be 3-D (n_windows, window_size, n_features).")
    n = windows.shape[0]
    if labels is None:
        labels = np.zeros(n, dtype=np.int64)
    if len(labels) != n:
        raise ValueError("labels length must match number of windows.")
    return [window_to_graph(windows[i], y=int(labels[i]), **kwargs) for i in range(n)]
