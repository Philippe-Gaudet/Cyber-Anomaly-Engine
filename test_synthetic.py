"""Tests for the synthetic CICIDS-like traffic generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from barcode.data_pipeline import DEFAULT_FEATURES, preprocess
from barcode.ui import components
from barcode.ui.components import (
    _CLASS_PROFILES,
    _generate_class_block,
    synthetic_traffic,
)


def test_synthetic_columns_match_loader_schema():
    df = synthetic_traffic(n_samples=300, seed=0)
    assert set(DEFAULT_FEATURES) <= set(df.columns)
    assert "label" in df.columns and "attack_type" in df.columns
    assert set(df["label"].unique()) <= {0, 1}


def test_synthetic_class_imbalance_matches_ratio():
    df = synthetic_traffic(n_samples=2000, benign_ratio=0.8, seed=0)
    benign_frac = (df["label"] == 0).mean()
    # within ±2% of target
    assert abs(benign_frac - 0.8) < 0.02


def test_synthetic_respects_requested_sample_count():
    assert len(synthetic_traffic(n_samples=8000, seed=0)) == 8000
    assert len(synthetic_traffic(n_samples=333, benign_ratio=0.73, seed=0)) == 333
    assert len(synthetic_traffic(n_samples=100, attack_classes=(), seed=0)) == 100


def test_synthetic_rejects_bad_generation_params():
    with pytest.raises(ValueError):
        synthetic_traffic(n_samples=-1)
    with pytest.raises(ValueError):
        synthetic_traffic(benign_ratio=1.1)


def test_synthetic_attack_classes_distinct():
    df = synthetic_traffic(n_samples=5000, seed=0)
    atypes = set(df["attack_type"].unique())
    expected = {"BENIGN", "DDoS", "PortScan", "FTP-Patator", "SSH-Patator",
                "Web Attack - Brute Force", "Infiltration", "Bot"}
    assert expected <= atypes


def test_synthetic_features_finite_and_in_realistic_ranges():
    df = synthetic_traffic(n_samples=500, seed=0)
    arr = df[DEFAULT_FEATURES].to_numpy()
    assert np.isfinite(arr).all()
    # Sanity: packet lengths capped at 1500 (MTU), durations positive.
    for col in ("Fwd Packet Length Max", "Bwd Packet Length Max",
                "Fwd Packet Length Min", "Bwd Packet Length Min"):
        assert (df[col] >= 0).all() and (df[col] <= 1500).all()
    assert (df["Flow Duration"] > 0).all()
    assert (df["Total Fwd Packets"] >= 1).all()


def test_synthetic_ddos_differs_from_benign_on_signature_features():
    df = synthetic_traffic(n_samples=5000, seed=0)
    benign = df[df["attack_type"] == "BENIGN"]
    ddos = df[df["attack_type"] == "DDoS"]
    # DDoS should have much higher Flow Packets/s and much smaller packets.
    assert ddos["Flow Packets/s"].median() > benign["Flow Packets/s"].median() * 5
    assert ddos["Fwd Packet Length Mean"].median() < benign["Fwd Packet Length Mean"].median()


def test_synthetic_portscan_has_shorter_durations_than_benign():
    df = synthetic_traffic(n_samples=4000, seed=0)
    benign = df[df["attack_type"] == "BENIGN"]
    portscan = df[df["attack_type"] == "PortScan"]
    assert portscan["Flow Duration"].median() < benign["Flow Duration"].median() / 10


def test_synthetic_infiltration_is_hard_close_to_benign():
    """Infiltration profile should overlap heavily with BENIGN (hard case)."""
    df = synthetic_traffic(n_samples=5000, seed=0)
    benign = df[df["attack_type"] == "BENIGN"]
    infil = df[df["attack_type"] == "Infiltration"]
    # Means should be within 1 order of magnitude for the size features.
    ratio = infil["Fwd Packet Length Mean"].median() / max(benign["Fwd Packet Length Mean"].median(), 1e-9)
    assert 0.3 < ratio < 3.0


def test_synthetic_segment_sizes_duplicate_packet_means():
    df = synthetic_traffic(n_samples=200, seed=0)
    np.testing.assert_allclose(df["Avg Fwd Segment Size"], df["Fwd Packet Length Mean"])
    np.testing.assert_allclose(df["Avg Bwd Segment Size"], df["Bwd Packet Length Mean"])


def test_synthetic_flows_through_preprocess():
    """The full preprocess path (scaler fit on BENIGN, transform all) must succeed."""
    df = synthetic_traffic(n_samples=500, seed=0)
    X, y, scaler = preprocess(df)
    assert np.isfinite(X).all()
    benign_mean = X[y == 0].mean(axis=0)
    np.testing.assert_allclose(benign_mean, 0.0, atol=1e-6)


def test_load_or_synthesize_scales_synthetic_branch(monkeypatch):
    components.load_or_synthesize.clear()

    def small_synthetic():
        return synthetic_traffic(n_samples=500, seed=123)

    monkeypatch.setattr(components, "cicids_available", lambda: False)
    monkeypatch.setattr(components, "synthetic_traffic", small_synthetic)
    data = components.load_or_synthesize()
    components.load_or_synthesize.clear()

    assert data["synthetic"] is True
    benign_mean = data["X"][data["y"] == 0].mean(axis=0)
    np.testing.assert_allclose(benign_mean, 0.0, atol=1e-6)


def test_synthetic_rejects_unknown_attack_class():
    with pytest.raises(KeyError):
        synthetic_traffic(n_samples=100, attack_classes=("Nope",), seed=0)


def test_generate_class_block_shape_matches_features():
    rng = np.random.default_rng(0)
    arr = _generate_class_block(rng, 10, _CLASS_PROFILES["DDoS"])
    assert arr.shape == (10, len(DEFAULT_FEATURES))
