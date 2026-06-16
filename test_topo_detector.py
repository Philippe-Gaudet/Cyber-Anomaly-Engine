"""Tests for src/barcode/topo_detector.py."""

from __future__ import annotations

import numpy as np
import pytest

from barcode.topo_detector import TopoDetector


@pytest.fixture(scope="module")
def normal_data():
    rng = np.random.default_rng(0)
    return rng.normal(0.0, 1.0, size=(400, 5)).astype(np.float64)


@pytest.fixture(scope="module")
def attack_data():
    rng = np.random.default_rng(1)
    base = rng.normal(0.0, 3.5, size=(400, 5))
    base[::4] += 10.0
    return base.astype(np.float64)


@pytest.fixture(scope="module")
def fitted_detector(normal_data):
    det = TopoDetector(window_size=40, step=20, n_jobs=1)
    det.fit(normal_data)
    return det


def test_fit_returns_self(normal_data):
    det = TopoDetector(window_size=40, step=20, n_jobs=1)
    out = det.fit(normal_data)
    assert out is det
    assert hasattr(det, "reference_")
    assert hasattr(det, "normal_scores_")


def test_fit_raises_when_too_few_samples():
    det = TopoDetector(window_size=50, step=25, n_jobs=1)
    with pytest.raises(ValueError):
        det.fit(np.zeros((10, 5)))


def test_score_samples_length_matches_windows(fitted_detector, normal_data):
    scores = fitted_detector.score_samples(normal_data)
    expected = (len(normal_data) - 40) // 20 + 1
    assert scores.shape == (expected,)
    assert np.isfinite(scores).all()
    assert (scores >= 0).all()


def test_score_before_fit_raises(normal_data):
    det = TopoDetector(window_size=40, step=20, n_jobs=1)
    with pytest.raises(RuntimeError):
        det.score_samples(normal_data)


def test_fit_threshold_sets_attribute(fitted_detector, normal_data):
    thr = fitted_detector.fit_threshold(normal_data, percentile=95.0)
    assert isinstance(thr, float)
    assert thr >= 0
    assert fitted_detector.threshold_ == thr


def test_fit_threshold_rejects_bad_percentile(fitted_detector, normal_data):
    with pytest.raises(ValueError):
        fitted_detector.fit_threshold(normal_data, percentile=150.0)


def test_predict_is_binary(fitted_detector, normal_data):
    fitted_detector.fit_threshold(normal_data, percentile=95.0)
    preds = fitted_detector.predict(normal_data)
    assert set(np.unique(preds).tolist()) <= {0, 1}
    # On the calibration data itself, no more than ~5% should be flagged.
    assert preds.mean() <= 0.2


def test_predict_without_threshold_raises(normal_data):
    det = TopoDetector(window_size=40, step=20, n_jobs=1)
    det.fit(normal_data)
    with pytest.raises(ValueError):
        det.predict(normal_data)


def test_attack_scores_higher_than_normal(fitted_detector, normal_data, attack_data):
    normal_scores = fitted_detector.score_samples(normal_data)
    attack_scores = fitted_detector.score_samples(attack_data)
    assert attack_scores.mean() > normal_scores.mean()


def test_save_and_load_roundtrip(fitted_detector, normal_data, tmp_path):
    fitted_detector.fit_threshold(normal_data, percentile=95.0)
    path = tmp_path / "det.joblib"
    fitted_detector.save(path)
    loaded = TopoDetector.load(path)
    assert isinstance(loaded, TopoDetector)
    np.testing.assert_allclose(
        loaded.score_samples(normal_data),
        fitted_detector.score_samples(normal_data),
    )
    assert loaded.threshold_ == fitted_detector.threshold_
