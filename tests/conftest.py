"""Shared test helpers."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def sine():
    """Factory for a pure sine tone at `freq` Hz."""

    def _make(
        freq: float, dur: float = 0.5, sr: int = 16000, amp: float = 0.8
    ) -> np.ndarray:
        t = np.arange(int(dur * sr)) / sr
        return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float64)

    return _make


@pytest.fixture
def noise_burst():
    """Factory for a deterministic white-noise burst."""

    def _make(dur: float = 0.1, sr: int = 16000, amp: float = 0.8) -> np.ndarray:
        rng = np.random.default_rng(0)
        return (amp * rng.standard_normal(int(dur * sr))).astype(np.float64)

    return _make
