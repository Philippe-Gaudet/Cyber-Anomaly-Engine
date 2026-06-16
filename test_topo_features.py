"""Tests for src/barcode/topo_features.py."""

from __future__ import annotations

import numpy as np
import pytest

from barcode.topo_features import (
    align_diagrams,
    build_persistence_pipeline,
    compute_diagrams,
    compute_reference_diagram,
    diagrams_to_vectors,
    wasserstein_scores,
)


@pytest.fixture(scope="module")
def normal_windows():
    rng = np.random.default_rng(0)
    return rng.normal(0.0, 1.0, size=(4, 20, 5)).astype(np.float64)


@pytest.fixture(scope="module")
def attack_windows():
    rng = np.random.default_rng(1)
    # Wider spread + outliers -> different topology than the normal cluster.
    base = rng.normal(0.0, 3.0, size=(4, 20, 5))
    base[:, ::4, :] += 8.0
    return base.astype(np.float64)


@pytest.fixture(scope="module")
def pipeline():
    return build_persistence_pipeline(homology_dimensions=(0, 1), n_jobs=1)


@pytest.fixture(scope="module")
def normal_diagrams(normal_windows, pipeline):
    return compute_diagrams(normal_windows, pipeline)


@pytest.fixture(scope="module")
def attack_diagrams(attack_windows, pipeline):
    return compute_diagrams(attack_windows, pipeline)


def test_compute_diagrams_shape(normal_diagrams, normal_windows):
    assert normal_diagrams.ndim == 3
    assert normal_diagrams.shape[0] == normal_windows.shape[0]
    assert normal_diagrams.shape[-1] == 3
    # Homology-dim column contains only valid dims.
    dims = set(np.unique(normal_diagrams[..., 2]).tolist())
    assert dims <= {0.0, 1.0}


def test_compute_diagrams_empty_input(pipeline):
    empty = np.empty((0, 20, 5), dtype=np.float64)
    out = compute_diagrams(empty, pipeline)
    assert out.shape[0] == 0


def test_compute_diagrams_rejects_2d(pipeline):
    with pytest.raises(ValueError):
        compute_diagrams(np.zeros((20, 5)), pipeline)


@pytest.mark.parametrize("method", ["persistence_image", "betti_curve", "landscape"])
def test_vectors_finite_and_2d(normal_diagrams, method):
    vecs = diagrams_to_vectors(normal_diagrams, method=method, n_bins=10)
    assert vecs.ndim == 2
    assert vecs.shape[0] == normal_diagrams.shape[0]
    assert np.isfinite(vecs).all()
    assert vecs.shape[1] > 0


def test_vectors_unknown_method_raises(normal_diagrams):
    with pytest.raises(ValueError):
        diagrams_to_vectors(normal_diagrams, method="bogus")


def test_reference_diagram_preserves_dim_column(normal_diagrams):
    ref = compute_reference_diagram(normal_diagrams)
    assert ref.shape == (1, normal_diagrams.shape[1], 3)
    # Last column (homology dim) must match the source.
    np.testing.assert_array_equal(ref[0, :, 2], normal_diagrams[0, :, 2])


def test_reference_diagram_rejects_empty():
    with pytest.raises(ValueError):
        compute_reference_diagram(np.empty((0, 5, 3)))


def test_wasserstein_scores_non_negative(normal_diagrams):
    ref = compute_reference_diagram(normal_diagrams)
    scores = wasserstein_scores(normal_diagrams, ref, dimension=1)
    assert scores.shape == (normal_diagrams.shape[0],)
    assert (scores >= 0).all()


def test_wasserstein_attack_scores_higher_than_normal(
    normal_diagrams, attack_diagrams
):
    ref = compute_reference_diagram(normal_diagrams)
    normal_scores = wasserstein_scores(normal_diagrams, ref, dimension=1)
    attack_scores = wasserstein_scores(attack_diagrams, ref, dimension=1)
    # Attacks should drift further from the normal reference on average.
    assert attack_scores.mean() > normal_scores.mean()


def test_wasserstein_rejects_bad_dimension(normal_diagrams):
    ref = compute_reference_diagram(normal_diagrams)
    with pytest.raises(ValueError):
        wasserstein_scores(normal_diagrams, ref, dimension=2, homology_dimensions=(0, 1))


def test_wasserstein_rejects_non_singleton_reference(normal_diagrams):
    bad_ref = normal_diagrams[:2]  # shape (2, n_points, 3)
    with pytest.raises(ValueError):
        wasserstein_scores(normal_diagrams, bad_ref)


# --- align_diagrams -------------------------------------------------------

def test_align_diagrams_makes_concatenable(normal_diagrams, attack_diagrams):
    """Independently computed batches must be paddable to a shared shape."""
    a, b = align_diagrams(normal_diagrams, attack_diagrams)
    assert a.shape[1] == b.shape[1]
    assert a.shape[-1] == 3 and b.shape[-1] == 3
    # And concatenation now succeeds, which is the whole point.
    stacked = np.concatenate([a, b], axis=0)
    assert stacked.shape[0] == normal_diagrams.shape[0] + attack_diagrams.shape[0]


def test_align_diagrams_preserves_real_points(normal_diagrams):
    """Padding rows must be (0, 0, d) — no real birth/death values get dropped or perturbed."""
    short = normal_diagrams[:1]
    long = normal_diagrams  # has the same per-batch shape, but use it as a "wider" target
    a, _ = align_diagrams(short, long)
    real_short = short[short[..., 1] > 0]
    real_a = a[a[..., 1] > 0]
    # The same non-trivial (birth, death, dim) rows should still be present.
    np.testing.assert_allclose(
        np.sort(real_short.reshape(-1)), np.sort(real_a.reshape(-1))
    )


def test_align_diagrams_three_batches(normal_diagrams, attack_diagrams):
    """N-batch alignment, not just 2."""
    smaller = normal_diagrams[:1, : normal_diagrams.shape[1] // 2, :]
    out = align_diagrams(normal_diagrams, attack_diagrams, smaller)
    sizes = {x.shape[1] for x in out}
    assert len(sizes) == 1


def test_align_diagrams_empty_batches():
    assert align_diagrams() == []


def test_align_diagrams_vectorizers_accept_padded(normal_diagrams, attack_diagrams):
    """PersistenceImage and BettiCurve must accept the padded output."""
    from barcode.topo_features import BettiCurve, PersistenceImage

    a, b = align_diagrams(normal_diagrams, attack_diagrams)
    combined = np.concatenate([a, b], axis=0)
    img = PersistenceImage(n_bins=10).fit_transform(combined)
    assert img.shape[0] == combined.shape[0]
    assert np.isfinite(img).all()
    bc = BettiCurve(n_bins=10).fit_transform(combined)
    assert bc.shape[0] == combined.shape[0]
    assert np.isfinite(bc).all()
