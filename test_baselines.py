"""Tests for src/barcode/baselines.py."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from barcode.baselines import (
    AutoencoderDetector,
    IsolationForestDetector,
    _AutoencoderNet,
    evaluate_detector,
)


@pytest.fixture(scope="module")
def split():
    rng = np.random.default_rng(0)
    n_normal_train = 400
    n_normal_test = 100
    n_attack_test = 100
    X_train = rng.normal(0, 1, size=(n_normal_train, 8))
    X_test = np.vstack([
        rng.normal(0, 1, size=(n_normal_test, 8)),
        rng.normal(0, 1, size=(n_normal_test, 8)),  # second BENIGN block for variety
        rng.normal(5, 2, size=(n_attack_test, 8)),
    ])
    y_test = np.array([0] * (2 * n_normal_test) + [1] * n_attack_test)
    atypes = np.array(
        ["BENIGN"] * n_normal_test
        + ["BENIGN"] * n_normal_test
        + ["DDoS"] * (n_attack_test // 2)
        + ["PortScan"] * (n_attack_test - n_attack_test // 2)
    )
    return {"X_train": X_train.astype(np.float64), "X_test": X_test.astype(np.float64),
            "y_test": y_test, "attack_types": atypes}


# --- IsolationForestDetector ----------------------------------------------

def test_if_fits_and_scores(split):
    det = IsolationForestDetector(n_estimators=20)
    det.fit(split["X_train"])
    scores = det.score_samples(split["X_test"])
    assert scores.shape == (split["X_test"].shape[0],)
    assert np.isfinite(scores).all()


def test_if_attack_scores_higher_than_normal(split):
    det = IsolationForestDetector(n_estimators=50, contamination=0.05).fit(split["X_train"])
    scores = det.score_samples(split["X_test"])
    y = split["y_test"]
    assert scores[y == 1].mean() > scores[y == 0].mean()


def test_if_predict_is_binary(split):
    det = IsolationForestDetector(n_estimators=30).fit(split["X_train"])
    det.fit_threshold(split["X_train"], percentile=95.0)
    preds = det.predict(split["X_test"])
    assert set(np.unique(preds).tolist()) <= {0, 1}


def test_if_predict_without_threshold_raises(split):
    det = IsolationForestDetector(n_estimators=10).fit(split["X_train"])
    with pytest.raises(ValueError):
        det.predict(split["X_test"])


def test_if_fit_threshold_rejects_bad_inputs(split):
    det = IsolationForestDetector(n_estimators=10).fit(split["X_train"])
    with pytest.raises(ValueError):
        det.fit_threshold(split["X_train"], percentile=-1.0)
    with pytest.raises(ValueError):
        det.fit_threshold(split["X_train"][:0], percentile=95.0)


# --- AutoencoderDetector --------------------------------------------------

def test_ae_forward_shape():
    net = _AutoencoderNet(n_features=8)
    x = torch.randn(4, 8)
    out = net(x)
    assert out.shape == (4, 8)


def test_ae_fits_and_scores(split):
    det = AutoencoderDetector(n_epochs=5, batch_size=64, device="cpu")
    det.fit(split["X_train"])
    scores = det.score_samples(split["X_test"])
    assert scores.shape == (split["X_test"].shape[0],)
    assert np.isfinite(scores).all()
    assert (scores >= 0).all()


def test_ae_attack_scores_higher_than_normal(split):
    det = AutoencoderDetector(n_epochs=20, batch_size=64, device="cpu").fit(split["X_train"])
    scores = det.score_samples(split["X_test"])
    y = split["y_test"]
    assert scores[y == 1].mean() > scores[y == 0].mean()


def test_ae_predict_threshold_required(split):
    det = AutoencoderDetector(n_epochs=2, batch_size=64, device="cpu").fit(split["X_train"])
    with pytest.raises(ValueError):
        det.predict(split["X_test"])


def test_ae_fit_threshold_rejects_bad_inputs(split):
    det = AutoencoderDetector(n_epochs=1, batch_size=64, device="cpu").fit(split["X_train"])
    with pytest.raises(ValueError):
        det.fit_threshold(split["X_train"], percentile=101.0)
    with pytest.raises(ValueError):
        det.fit_threshold(split["X_train"][:0], percentile=95.0)


# --- evaluate_detector ----------------------------------------------------

def test_evaluate_detector_returns_expected_keys(split):
    det = IsolationForestDetector(n_estimators=30).fit(split["X_train"])
    det.fit_threshold(split["X_train"], percentile=95.0)
    result = evaluate_detector(det, split["X_test"], split["y_test"], split["attack_types"])
    assert set(result.keys()) == {"scores", "predictions", "overall", "per_attack_type"}
    for key in ("auc_roc", "f1", "precision", "recall", "avg_precision"):
        assert key in result["overall"]
    # Per-attack-type breakdown should cover both attack labels.
    assert {"DDoS", "PortScan"} <= set(result["per_attack_type"].keys())


def test_evaluate_detector_auc_in_unit_interval(split):
    det = IsolationForestDetector(n_estimators=30).fit(split["X_train"])
    det.fit_threshold(split["X_train"], percentile=95.0)
    overall = evaluate_detector(det, split["X_test"], split["y_test"], split["attack_types"])["overall"]
    assert 0.0 <= overall["auc_roc"] <= 1.0
    assert 0.0 <= overall["avg_precision"] <= 1.0


def test_evaluate_detector_handles_missing_attack_types(split):
    det = IsolationForestDetector(n_estimators=30).fit(split["X_train"])
    det.fit_threshold(split["X_train"], percentile=95.0)
    result = evaluate_detector(det, split["X_test"], split["y_test"])
    assert result["per_attack_type"] == {}
