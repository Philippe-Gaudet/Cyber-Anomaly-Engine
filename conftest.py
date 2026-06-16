"""Shared synthetic fixtures. Tests must not depend on CICIDS2017."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from barcode.data_pipeline import DEFAULT_FEATURES


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def small_normal_array(rng) -> np.ndarray:
    return rng.normal(0.0, 1.0, size=(100, 10)).astype(np.float64)


@pytest.fixture
def small_attack_array(rng) -> np.ndarray:
    return rng.normal(5.0, 2.0, size=(50, 10)).astype(np.float64)


@pytest.fixture
def sample_windows(rng) -> np.ndarray:
    return rng.normal(0.0, 1.0, size=(5, 20, 10)).astype(np.float64)


@pytest.fixture
def synthetic_cicids_dir(tmp_path, rng) -> str:
    """Write two CSV files that look like CICIDS2017 output (with leading-space columns)."""
    cols = [" " + c for c in DEFAULT_FEATURES] + [" Label"]

    n_benign = 200
    n_attack = 50

    benign = rng.normal(0.0, 1.0, size=(n_benign, len(DEFAULT_FEATURES)))
    attack = rng.normal(3.0, 1.5, size=(n_attack, len(DEFAULT_FEATURES)))

    benign_df = pd.DataFrame(benign, columns=cols[:-1])
    benign_df[" Label"] = "BENIGN"
    attack_df = pd.DataFrame(attack, columns=cols[:-1])
    attack_df[" Label"] = "DDoS"

    # Inject an inf to exercise cleaning.
    benign_df.iloc[0, 0] = np.inf

    out1 = tmp_path / "Monday-WorkingHours.pcap_ISCX.csv"
    out2 = tmp_path / "Friday-DDoS.pcap_ISCX.csv"
    benign_df.to_csv(out1, index=False)
    attack_df.to_csv(out2, index=False)
    return str(tmp_path)
