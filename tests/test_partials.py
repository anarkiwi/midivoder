"""Tests for analysis + partial tracking on synthetic signals."""

from __future__ import annotations

import numpy as np
import pytest

from midivoder.analysis import analyze
from midivoder.config import EncodeConfig
from midivoder.partials import track_partials
from midivoder.percussion import extract_percussion


def _median_freq(partial):
    return float(np.median(partial.freqs))


def test_single_tone_yields_one_dominant_partial(sine):
    """A 440 Hz sine tracks to one long partial at 440 Hz."""
    cfg = EncodeConfig()
    frames = analyze(sine(440.0, dur=0.5), cfg)
    assert any(f.voiced for f in frames)
    partials = track_partials(frames, cfg)
    assert partials
    longest = max(partials, key=lambda p: p.duration)
    assert _median_freq(longest) == pytest.approx(440.0, rel=0.03)
    assert longest.duration > 0.3


def test_two_tones_yield_two_partials(sine):
    """A two-tone mix yields a partial per tone."""
    cfg = EncodeConfig()
    mix = sine(440.0, dur=0.5) + sine(660.0, dur=0.5)
    frames = analyze(mix, cfg)
    partials = track_partials(frames, cfg)
    long_partials = [p for p in partials if p.duration > 0.3]
    freqs = sorted(_median_freq(p) for p in long_partials)
    assert len(long_partials) >= 2
    assert any(abs(f - 440.0) < 20 for f in freqs)
    assert any(abs(f - 660.0) < 20 for f in freqs)


def test_partial_count_capped_by_channels(sine):
    """Simultaneous partials never exceed the tonal channel budget."""
    cfg = EncodeConfig(max_channels=3)  # 2 tonal channels
    mix = sum(sine(f, dur=0.4) for f in (300.0, 500.0, 700.0, 900.0, 1100.0))
    frames = analyze(mix, cfg)
    partials = track_partials(frames, cfg)
    # No more than max_partials should ever be simultaneously active.
    edges = []
    for p in partials:
        edges.append((p.times[0], 1))
        edges.append((p.times[-1], -1))
    edges.sort()
    active = peak = 0
    for _t, delta in edges:
        active += delta
        peak = max(peak, active)
    assert peak <= cfg.max_partials


def test_noise_routed_to_percussion(noise_burst):
    """A noise burst produces percussion hits with amplitudes relative to the peak."""
    cfg = EncodeConfig()
    # Silence then a noise burst -> at least one percussion hit, few/no tonal partials.
    sig = np.concatenate([np.zeros(8000), noise_burst(dur=0.2)])
    frames = analyze(sig, cfg)
    perc = extract_percussion(frames, cfg)
    assert perc
    assert all(0.0 < e.amp <= 1.0 for e in perc)
