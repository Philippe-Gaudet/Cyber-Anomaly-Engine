"""Tests for src/barcode/data_pipeline.py."""

from __future__ import annotations

import numpy as np
import pytest

from barcode.data_pipeline import (
    DEFAULT_FEATURES,
    extract_windows,
    load_cicids,
    preprocess,
    train_test_split_topo,
)


def test_load_cicids_strips_columns_and_labels(synthetic_cicids_dir):
    df = load_cicids(synthetic_cicids_dir)
    assert "label" in df.columns
    assert "attack_type" in df.columns
    for feat in DEFAULT_FEATURES:
        assert feat in df.columns
    assert set(df["label"].unique()) <= {0, 1}
    assert (df["attack_type"] == "BENIGN").any()
    assert (df["attack_type"] == "DDoS").any()


def test_load_cicids_drops_inf_and_nan(synthetic_cicids_dir):
    df = load_cicids(synthetic_cicids_dir)
    feats = df[DEFAULT_FEATURES].to_numpy()
    assert np.isfinite(feats).all()


def test_load_cicids_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cicids(str(tmp_path / "does-not-exist"))


def test_preprocess_returns_finite_array_and_fits_on_benign(synthetic_cicids_dir):
    df = load_cicids(synthetic_cicids_dir)
    X, y, scaler = preprocess(df)
    assert X.shape[0] == len(df)
    assert np.isfinite(X).all()
    assert set(np.unique(y).tolist()) <= {0, 1}
    # Mean of the BENIGN rows should be ~0 because the scaler was fit on them.
    benign_mean = X[y == 0].mean(axis=0)
    assert np.allclose(benign_mean, 0.0, atol=1e-6)


def test_preprocess_reuses_provided_scaler(synthetic_cicids_dir):
    df = load_cicids(synthetic_cicids_dir)
    X1, _, scaler = preprocess(df)
    X2, _, scaler2 = preprocess(df, scaler=scaler)
    assert scaler is scaler2
    np.testing.assert_allclose(X1, X2)


def test_train_test_split_train_is_normal_only():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 5))
    y = np.concatenate([np.zeros(200, dtype=int), np.ones(100, dtype=int)])
    atk = np.array(["BENIGN"] * 200 + ["DDoS"] * 100)
    split = train_test_split_topo(X, y, attack_types=atk, val_ratio=0.1, test_ratio=0.2)
    # Train must be drawn only from BENIGN samples.
    assert split["X_train"].shape[1] == 5
    assert split["X_val"].shape[1] == 5
    assert split["X_test"].shape[1] == 5
    assert "y_test" in split and "attack_types_test" in split
    # All attack rows go to the test set.
    assert (split["y_test"] == 1).sum() == 100
    # No index overlap.
    assert len(set(split["train_idx"]) & set(split["val_idx"])) == 0
    assert len(set(split["train_idx"]) & set(split["test_idx"])) == 0


def test_train_test_split_requires_benign_samples():
    X = np.ones((10, 3))
    y = np.ones(10, dtype=int)
    with pytest.raises(ValueError):
        train_test_split_topo(X, y)


def test_train_test_split_rejects_bad_ratios():
    X = np.ones((20, 3))
    y = np.zeros(20, dtype=int)
    with pytest.raises(ValueError):
        train_test_split_topo(X, y, val_ratio=-0.1)
    with pytest.raises(ValueError):
        train_test_split_topo(X, y, test_ratio=1.0)
    with pytest.raises(ValueError):
        train_test_split_topo(X, y, val_ratio=0.5, test_ratio=0.5)


def test_extract_windows_shape_and_count(small_normal_array):
    w = extract_windows(small_normal_array, window_size=20, step=10)
    expected = (100 - 20) // 10 + 1
    assert w.shape == (expected, 20, 10)


def test_extract_windows_skips_incomplete_trailing(small_normal_array):
    w = extract_windows(small_normal_array, window_size=30, step=40)
    # Starts: 0, 40 -> windows [0:30], [40:70]. 80 would go to 110 > 100 so skipped.
    assert w.shape[0] == 2


def test_extract_windows_too_short_returns_empty():
    X = np.zeros((10, 4))
    w = extract_windows(X, window_size=50, step=25)
    assert w.shape == (0, 50, 4)


def test_extract_windows_rejects_bad_inputs(small_normal_array):
    with pytest.raises(ValueError):
        extract_windows(small_normal_array.ravel(), window_size=10, step=5)
    with pytest.raises(ValueError):
        extract_windows(small_normal_array, window_size=0, step=5)
