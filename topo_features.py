"""Persistent homology features for sliding windows of network traffic.

Wraps giotto-tda's Vietoris-Rips persistence and diagram-vectorization
transformers, plus a Wasserstein scoring helper against a reference diagram.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from gtda.diagrams import (
    BettiCurve,
    PairwiseDistance,
    PersistenceImage,
    PersistenceLandscape,
)
from gtda.homology import VietorisRipsPersistence


def build_persistence_pipeline(
    homology_dimensions: Sequence[int] = (0, 1),
    n_jobs: int = -1,
) -> VietorisRipsPersistence:
    """Construct a Vietoris-Rips persistence transformer.

    Returns the transformer directly (single-step "pipeline"). Callers can wrap
    it in `sklearn.pipeline.Pipeline` if they want a multi-stage flow.
    """
    return VietorisRipsPersistence(
        metric="euclidean",
        homology_dimensions=tuple(homology_dimensions),
        n_jobs=n_jobs,
    )


def compute_diagrams(
    windows: np.ndarray,
    pipeline: VietorisRipsPersistence,
) -> np.ndarray:
    """Run the persistence pipeline on a stack of windows.

    Parameters
    ----------
    windows : ndarray of shape (n_windows, window_size, n_features)
    pipeline : a fitted-or-fittable VietorisRipsPersistence instance

    Returns
    -------
    diagrams : ndarray of shape (n_windows, n_points, 3)
        Each row is (birth, death, homology_dim).
    """
    if windows.ndim != 3:
        raise ValueError("windows must be 3-D (n_windows, window_size, n_features).")
    if windows.shape[0] == 0:
        return np.empty((0, 0, 3), dtype=np.float64)
    return pipeline.fit_transform(windows)


def diagrams_to_vectors(
    diagrams: np.ndarray,
    method: str = "persistence_image",
    n_bins: int = 20,
) -> np.ndarray:
    """Vectorize a batch of persistence diagrams.

    Methods: 'persistence_image', 'betti_curve', 'landscape'. The output is
    flattened per sample to shape (n_samples, vector_dim).
    """
    if diagrams.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float64)

    if method == "persistence_image":
        transformer = PersistenceImage(n_bins=n_bins)
    elif method == "betti_curve":
        transformer = BettiCurve(n_bins=n_bins)
    elif method == "landscape":
        transformer = PersistenceLandscape(n_bins=n_bins)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Use 'persistence_image', 'betti_curve', or 'landscape'."
        )

    vectors = transformer.fit_transform(diagrams)
    return vectors.reshape(vectors.shape[0], -1)


def compute_reference_diagram(normal_diagrams: np.ndarray) -> np.ndarray:
    """Element-wise mean of normal-class diagrams.

    Returns a diagram of shape (1, n_points, 3) suitable as the `reference`
    argument to `wasserstein_scores`. The homology-dimension column is preserved
    because giotto-tda pads diagrams consistently across a batch.
    """
    if normal_diagrams.ndim != 3 or normal_diagrams.shape[-1] != 3:
        raise ValueError("normal_diagrams must have shape (n, n_points, 3).")
    if normal_diagrams.shape[0] == 0:
        raise ValueError("Cannot build a reference from an empty diagram batch.")

    mean = normal_diagrams.mean(axis=0, keepdims=True)
    # Restore the homology-dimension column to integers (mean of equal ints is still
    # the int, but float==int comparisons later are cleaner if we explicitly cast).
    mean[..., 2] = normal_diagrams[0, :, 2]
    return mean


def align_diagrams(
    *batches: np.ndarray,
    homology_dimensions: Sequence[int] = (0, 1),
) -> list[np.ndarray]:
    """Pad two or more diagram batches so they share the same point count per H-block.

    giotto-tda pads diagrams *within a batch* so every sample has the same
    number of (birth, death, dim) rows, but the max point count differs
    between independent `fit_transform` calls — making cross-batch
    `np.concatenate` blow up. This helper takes any number of batches and
    pads each one's H-d block out to the common max, using (0, 0, d) filler
    rows that all gtda vectorizers treat as no-ops (zero-persistence points
    on the diagonal).

    Parameters
    ----------
    *batches : ndarrays of shape (n_i, n_points_i, 3)
        At least one batch. Each must follow the gtda convention of grouping
        rows by homology dimension in the order given by `homology_dimensions`.

    Returns
    -------
    list[ndarray]
        Same number of batches, all reshaped to (n_i, max_total_points, 3)
        where `max_total_points = sum_d max_i(block_size_d_in_batch_i)`.
    """
    if not batches:
        return []

    def split_blocks(diags: np.ndarray) -> list[np.ndarray]:
        if diags.shape[0] == 0:
            return [np.empty((0, 0, 3), dtype=diags.dtype) for _ in homology_dimensions]
        return [diags[:, diags[0, :, 2] == d, :] for d in homology_dimensions]

    all_blocks = [split_blocks(b) for b in batches]
    targets = [
        max(blocks[i].shape[1] for blocks in all_blocks)
        for i in range(len(homology_dimensions))
    ]

    out: list[np.ndarray] = []
    for blocks in all_blocks:
        padded_blocks = [
            _pad_block(block, targets[i], homology_dimensions[i])
            for i, block in enumerate(blocks)
        ]
        out.append(np.concatenate(padded_blocks, axis=1))
    return out


def _pad_block(block: np.ndarray, target: int, dim: int) -> np.ndarray:
    pad = target - block.shape[1]
    if pad <= 0:
        return block
    filler = np.zeros((block.shape[0], pad, 3), dtype=block.dtype)
    filler[..., 2] = dim
    return np.concatenate([block, filler], axis=1)


def _pad_diagrams_to_match(
    a: np.ndarray,
    b: np.ndarray,
    homology_dimensions: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Backwards-compatible 2-batch wrapper around `align_diagrams`."""
    out = align_diagrams(a, b, homology_dimensions=homology_dimensions)
    return out[0], out[1]


def wasserstein_scores(
    diagrams: np.ndarray,
    reference: np.ndarray,
    dimension: int = 1,
    homology_dimensions: Sequence[int] = (0, 1),
    order: float = 2.0,
) -> np.ndarray:
    """Wasserstein distance between each diagram and the reference.

    Uses `gtda.diagrams.PairwiseDistance(metric='wasserstein')`, which giotto-tda
    exposes in place of a standalone `WassersteinDistance` class in 0.6.x.

    Parameters
    ----------
    diagrams : ndarray of shape (n, n_points, 3)
    reference : ndarray of shape (1, n_points, 3)
    dimension : homology dimension to extract distances for (default H1).
    homology_dimensions : the tuple that was used when building the diagrams;
        needed to map `dimension` to a column index in the distance tensor.
    order : Wasserstein order (default 2).
    """
    if reference.shape[0] != 1:
        raise ValueError("reference must have shape (1, n_points, 3).")
    if diagrams.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)

    homology_dimensions = tuple(homology_dimensions)
    if dimension not in homology_dimensions:
        raise ValueError(
            f"dimension={dimension} not in homology_dimensions={homology_dimensions}."
        )
    dim_index = homology_dimensions.index(dimension)

    reference, diagrams = _pad_diagrams_to_match(reference, diagrams, homology_dimensions)
    stacked = np.concatenate([reference, diagrams], axis=0)
    pd = PairwiseDistance(
        metric="wasserstein",
        metric_params={"p": order},
        order=None,
        n_jobs=1,
    )
    dists = pd.fit_transform(stacked)
    # dists has shape (n+1, n+1, n_homology_dims). Row 0 = reference vs all.
    scores = dists[0, 1:, dim_index]
    return np.asarray(scores, dtype=np.float64)
