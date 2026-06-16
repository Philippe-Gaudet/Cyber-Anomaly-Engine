"""GraphSAGE-based traffic classifier over per-window flow graphs."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
from sklearn.base import BaseEstimator
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from tqdm import tqdm

from barcode.data_pipeline import extract_windows
from barcode.graph_builder import windows_to_graphs


class TrafficGNN(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64) -> None:
        super().__init__()
        self.conv1 = SAGEConv(n_features, hidden)
        self.conv2 = SAGEConv(hidden, 32)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        h = torch.relu(self.conv2(h, edge_index))
        h = global_mean_pool(h, batch)
        return self.head(h).squeeze(-1)


class GNNDetector(BaseEstimator):
    """sklearn-style wrapper around `TrafficGNN`.

    Trained as a supervised binary classifier on per-window graphs. Score =
    sigmoid(logit). Operates on the same `X_train`/`X_test` ndarrays as the
    other detectors; window extraction and graph construction happen internally.
    """

    def __init__(
        self,
        window_size: int = 50,
        step: int = 25,
        n_epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
        hidden: int = 64,
        device: Optional[str] = None,
        threshold: Optional[float] = None,
        verbose: bool = False,
        n_buckets: int = 4,
    ) -> None:
        self.window_size = window_size
        self.step = step
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.hidden = hidden
        self.device = device
        self.threshold = threshold
        self.verbose = verbose
        self.n_buckets = n_buckets

    def _device(self) -> torch.device:
        return torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def _to_graphs(self, X: np.ndarray, y_samples: Optional[np.ndarray]) -> list[Data]:
        windows = extract_windows(X, window_size=self.window_size, step=self.step)
        if windows.shape[0] == 0:
            return []
        if y_samples is None:
            labels = np.zeros(windows.shape[0], dtype=np.int64)
        else:
            if len(y_samples) != len(X):
                raise ValueError("X and y length mismatch.")
            from barcode.benchmark import make_window_labels  # local import to avoid cycle at module import
            labels, _ = make_window_labels(y_samples, None, self.window_size, self.step)
        return windows_to_graphs(windows, labels=labels, n_buckets=self.n_buckets)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "GNNDetector":
        """Train on a balanced(ish) sample set.

        If `y` is None, fit on BENIGN-only (all labels=0). The model still
        learns to reconstruct a normal-class manifold via score patterns, but
        for meaningful supervised training the caller should provide labels.
        """
        device = self._device()
        graphs = self._to_graphs(X, y)
        if not graphs:
            raise ValueError("Not enough samples to build any graphs.")

        n_features = graphs[0].x.shape[1]
        self.model_ = TrafficGNN(n_features=n_features, hidden=self.hidden).to(device)
        self.device_ = device

        loader = DataLoader(graphs, batch_size=self.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        self.model_.train()
        epoch_iter = range(self.n_epochs)
        if self.verbose:
            epoch_iter = tqdm(epoch_iter, desc="GNN training")
        for _ in epoch_iter:
            for batch in loader:
                batch = batch.to(device)
                opt.zero_grad()
                logits = self.model_(batch.x, batch.edge_index, batch.batch)
                loss = loss_fn(logits, batch.y.view(-1))
                loss.backward()
                opt.step()

        self.threshold_ = self.threshold
        return self

    @torch.no_grad()
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        graphs = self._to_graphs(X, None)
        if not graphs:
            return np.empty((0,), dtype=np.float64)
        loader = DataLoader(graphs, batch_size=self.batch_size, shuffle=False)
        self.model_.eval()
        scores: list[float] = []
        for batch in loader:
            batch = batch.to(self.device_)
            logits = self.model_(batch.x, batch.edge_index, batch.batch)
            scores.extend(torch.sigmoid(logits).cpu().tolist())
        return np.asarray(scores, dtype=np.float64)

    def fit_threshold(self, X_val: np.ndarray, percentile: float = 95.0) -> float:
        self._check_fitted()
        if not (0.0 <= percentile <= 100.0):
            raise ValueError("percentile must be in [0, 100].")
        scores = self.score_samples(X_val)
        if scores.size == 0:
            raise ValueError("Validation set produced no windows; cannot calibrate.")
        self.threshold_ = float(np.percentile(scores, percentile))
        return self.threshold_

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        self._check_fitted()
        thr = threshold if threshold is not None else self.threshold_
        if thr is None:
            thr = 0.5  # sensible default for sigmoid output
        return (self.score_samples(X) > thr).astype(np.int64)

    def _check_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise RuntimeError("GNNDetector is not fitted.")


# --- score fusion ---------------------------------------------------------

def _normalize(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def fuse_scores(topo: np.ndarray, gnn: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Late-fuse two score vectors by min-max normalizing each and convex-combining."""
    if topo.shape != gnn.shape:
        raise ValueError(f"topo and gnn must have the same shape: {topo.shape} vs {gnn.shape}")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1].")
    return alpha * _normalize(topo) + (1.0 - alpha) * _normalize(gnn)
