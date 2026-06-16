"""Sklearn-style anomaly detector built on persistent-homology features."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import joblib
import numpy as np
from sklearn.base import BaseEstimator

from barcode.data_pipeline import extract_windows
from barcode.topo_features import (
    build_persistence_pipeline,
    compute_diagrams,
    compute_reference_diagram,
    wasserstein_scores,
)


class TopoDetector(BaseEstimator):
    """Topological anomaly detector.

    Workflow:
        1. Slide windows over the input matrix.
        2. Build a Vietoris-Rips persistence diagram per window.
        3. Compute Wasserstein distance from each diagram to the BENIGN
           reference diagram (the element-wise mean of training-set diagrams).
        4. Flag a window as anomalous when its distance exceeds `threshold_`.
    """

    def __init__(
        self,
        window_size: int = 50,
        step: int = 25,
        homology_dimensions: Sequence[int] = (0, 1),
        vector_method: str = "persistence_image",
        threshold: Optional[float] = None,
        n_jobs: int = -1,
        scoring_dimension: int = 1,
    ) -> None:
        self.window_size = window_size
        self.step = step
        self.homology_dimensions = tuple(homology_dimensions)
        self.vector_method = vector_method
        self.threshold = threshold
        self.n_jobs = n_jobs
        self.scoring_dimension = scoring_dimension

    # --- fit / score / predict -------------------------------------------------

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "TopoDetector":
        """Fit on BENIGN samples only. y is accepted for sklearn compatibility."""
        windows = extract_windows(X, window_size=self.window_size, step=self.step)
        if windows.shape[0] == 0:
            raise ValueError(
                f"Not enough samples ({X.shape[0]}) to build a single window of size {self.window_size}."
            )

        self.pipeline_ = build_persistence_pipeline(
            homology_dimensions=self.homology_dimensions,
            n_jobs=self.n_jobs,
        )
        normal_diagrams = compute_diagrams(windows, self.pipeline_)
        self.reference_ = compute_reference_diagram(normal_diagrams)
        self.normal_scores_ = wasserstein_scores(
            normal_diagrams,
            self.reference_,
            dimension=self.scoring_dimension,
            homology_dimensions=self.homology_dimensions,
        )
        # Persist the threshold attribute for sklearn-style introspection.
        self.threshold_ = self.threshold
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Per-window Wasserstein distance to the BENIGN reference."""
        self._check_fitted()
        windows = extract_windows(X, window_size=self.window_size, step=self.step)
        diagrams = compute_diagrams(windows, self.pipeline_)
        return wasserstein_scores(
            diagrams,
            self.reference_,
            dimension=self.scoring_dimension,
            homology_dimensions=self.homology_dimensions,
        )

    def fit_threshold(self, X_val: np.ndarray, percentile: float = 95.0) -> float:
        """Calibrate the alarm threshold on a held-out BENIGN slice."""
        self._check_fitted()
        if not (0.0 <= percentile <= 100.0):
            raise ValueError("percentile must be in [0, 100].")
        val_scores = self.score_samples(X_val)
        if val_scores.size == 0:
            raise ValueError("Validation set produced no windows; cannot calibrate.")
        self.threshold_ = float(np.percentile(val_scores, percentile))
        return self.threshold_

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Binary alerts: 1 when the window's score exceeds threshold, else 0."""
        self._check_fitted()
        thr = threshold if threshold is not None else self.threshold_
        if thr is None:
            raise ValueError(
                "No threshold available. Call fit_threshold(...) or pass `threshold=`."
            )
        scores = self.score_samples(X)
        return (scores > thr).astype(np.int64)

    # --- persistence ----------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        self._check_fitted()
        joblib.dump(self, str(path))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TopoDetector":
        obj = joblib.load(str(path))
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}: {type(obj)}")
        return obj

    # --- helpers --------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not hasattr(self, "reference_") or not hasattr(self, "pipeline_"):
            raise RuntimeError("TopoDetector is not fitted. Call fit(X_normal) first.")
