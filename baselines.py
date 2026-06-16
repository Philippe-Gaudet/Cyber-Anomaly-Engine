"""Baseline anomaly detectors (IsolationForest, Autoencoder) and a shared evaluator."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class IsolationForestDetector(BaseEstimator):
    """sklearn IsolationForest wrapped to match the TopoDetector interface."""

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        random_state: int = 42,
        threshold: Optional[float] = None,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.threshold = threshold

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "IsolationForestDetector":
        self.model_ = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        ).fit(X)
        self.threshold_ = self.threshold
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Higher score = more anomalous (flipped from sklearn's convention)."""
        self._check_fitted()
        return -self.model_.score_samples(X)

    def fit_threshold(self, X_val: np.ndarray, percentile: float = 95.0) -> float:
        self._check_fitted()
        if not (0.0 <= percentile <= 100.0):
            raise ValueError("percentile must be in [0, 100].")
        if len(X_val) == 0:
            raise ValueError("Validation set is empty; cannot calibrate.")
        scores = self.score_samples(X_val)
        self.threshold_ = float(np.percentile(scores, percentile))
        return self.threshold_

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        self._check_fitted()
        thr = threshold if threshold is not None else self.threshold_
        if thr is None:
            raise ValueError("No threshold available. Call fit_threshold(...) or pass `threshold=`.")
        return (self.score_samples(X) > thr).astype(np.int64)

    def _check_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise RuntimeError("IsolationForestDetector is not fitted.")


class _AutoencoderNet(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoencoderDetector(BaseEstimator):
    """Reconstruction-error autoencoder trained on BENIGN samples only."""

    def __init__(
        self,
        n_epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: Optional[str] = None,
        threshold: Optional[float] = None,
        verbose: bool = False,
    ) -> None:
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.threshold = threshold
        self.verbose = verbose

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "AutoencoderDetector":
        device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.device_ = device
        self.model_ = _AutoencoderNet(n_features=X.shape[1]).to(device)

        ds = TensorDataset(torch.from_numpy(X.astype(np.float32)))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.model_.train()
        iterator = range(self.n_epochs)
        if self.verbose:
            iterator = tqdm(iterator, desc="AE training")
        for _ in iterator:
            for (batch,) in loader:
                batch = batch.to(device)
                opt.zero_grad()
                recon = self.model_(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                opt.step()

        self.threshold_ = self.threshold
        return self

    @torch.no_grad()
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        self.model_.eval()
        x = torch.from_numpy(X.astype(np.float32)).to(self.device_)
        recon = self.model_(x)
        err = ((recon - x) ** 2).mean(dim=1).cpu().numpy()
        return err.astype(np.float64)

    def fit_threshold(self, X_val: np.ndarray, percentile: float = 95.0) -> float:
        self._check_fitted()
        if not (0.0 <= percentile <= 100.0):
            raise ValueError("percentile must be in [0, 100].")
        if len(X_val) == 0:
            raise ValueError("Validation set is empty; cannot calibrate.")
        scores = self.score_samples(X_val)
        self.threshold_ = float(np.percentile(scores, percentile))
        return self.threshold_

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        self._check_fitted()
        thr = threshold if threshold is not None else self.threshold_
        if thr is None:
            raise ValueError("No threshold available. Call fit_threshold(...) or pass `threshold=`.")
        return (self.score_samples(X) > thr).astype(np.int64)

    def _check_fitted(self) -> None:
        if not hasattr(self, "model_"):
            raise RuntimeError("AutoencoderDetector is not fitted.")


# --- shared evaluation -------------------------------------------------------

def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict:
    """Compute classification metrics, tolerating single-class slices."""
    out = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) < 2:
        out["auc_roc"] = float("nan")
        out["avg_precision"] = float("nan")
    else:
        out["auc_roc"] = float(roc_auc_score(y_true, scores))
        out["avg_precision"] = float(average_precision_score(y_true, scores))
    return out


def evaluate_detector(
    detector,
    X_test: np.ndarray,
    y_test: np.ndarray,
    attack_types: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
) -> dict:
    """Score, predict, and compute overall + per-attack-type metrics.

    Returns
    -------
    dict with keys:
        scores : ndarray of anomaly scores
        predictions : ndarray of binary predictions
        overall : dict of metrics on the full test set
        per_attack_type : dict mapping attack_type -> metrics (BENIGN + each attack)
    """
    scores = detector.score_samples(X_test)
    predictions = detector.predict(X_test, threshold=threshold) if threshold is not None else detector.predict(X_test)

    # Window-based detectors return one score per window, not per sample.
    # Trim y_test/attack_types to the score length so evaluation lines up.
    n = scores.shape[0]
    y_aligned = y_test[:n]
    atk_aligned = attack_types[:n] if attack_types is not None else None

    overall = _safe_metrics(y_aligned, predictions, scores)

    per_attack: dict = {}
    if atk_aligned is not None:
        benign_mask = y_aligned == 0
        for atype in np.unique(atk_aligned):
            atype_str = str(atype)
            if atype_str.upper() == "BENIGN":
                continue
            attack_mask = atk_aligned == atype
            mask = benign_mask | attack_mask
            if mask.sum() < 2:
                continue
            per_attack[atype_str] = _safe_metrics(
                y_aligned[mask], predictions[mask], scores[mask]
            )

    return {
        "scores": scores,
        "predictions": predictions,
        "overall": overall,
        "per_attack_type": per_attack,
    }
