"""Unit tests for frequency/amplitude -> MIDI mapping and channel allocation."""

from __future__ import annotations

import numpy as np
import pytest

from midivoder.config import EncodeConfig
from midivoder.midi import (
    _assign_channels,
    amp_to_velocity,
    bend_value,
    midi_float,
    nearest_note,
)
from midivoder.partials import Partial


@pytest.fixture
def cfg():
    """Default encoder configuration."""
    return EncodeConfig()


def test_a440_is_note_69(cfg):
    """A4 maps to MIDI note 69 exactly."""
    assert nearest_note(440.0, cfg) == 69
    assert abs(midi_float(440.0) - 69.0) < 1e-9


def test_octave_is_twelve_semitones(cfg):
    """Doubling/halving frequency moves by twelve notes."""
    assert nearest_note(880.0, cfg) == 81
    assert nearest_note(220.0, cfg) == 57


def test_note_clamped_to_range(cfg):
    """Out-of-range frequencies clamp to note_min/note_max."""
    assert nearest_note(20.0, cfg) >= cfg.note_min
    assert nearest_note(15000.0, cfg) <= cfg.note_max


def test_bend_zero_when_on_pitch():
    """An on-pitch note needs no bend."""
    assert bend_value(69.0, 69, 2.0) == 0


def test_bend_sign_and_range():
    """Bend scales linearly with offset and saturates at 14-bit limits."""
    # +1 semitone over a 2-semitone range = +half scale.
    assert bend_value(70.0, 69, 2.0) == pytest.approx(4096, abs=1)
    assert bend_value(68.0, 69, 2.0) == pytest.approx(-4096, abs=1)
    # Beyond range clamps to the 14-bit limits.
    assert bend_value(80.0, 69, 2.0) == 8191
    assert bend_value(60.0, 69, 2.0) == -8192


def test_velocity_monotonic_and_clamped(cfg):
    """Louder amplitudes give higher, clamped velocities."""
    quiet = amp_to_velocity(0.001, 1.0, cfg)
    loud = amp_to_velocity(1.0, 1.0, cfg)
    assert cfg.velocity_min <= quiet < loud <= cfg.velocity_max
    assert loud == cfg.velocity_max


def _partial(start: float, end: float) -> Partial:
    t = np.array([start, end])
    return Partial(
        times=t,
        freqs=np.array([440.0, 440.0]),
        amps=np.array([0.5, 0.5]),
        start_frame=0,
        end_frame=1,
    )


def test_non_overlapping_partials_reuse_channel(cfg):
    """Sequential partials share one channel."""
    partials = [_partial(0.0, 1.0), _partial(1.1, 2.0)]
    assignment = _assign_channels(partials, cfg)
    assert assignment[0] == assignment[1]


def test_overlapping_partials_use_distinct_channels(cfg):
    """Concurrent partials get separate channels."""
    partials = [_partial(0.0, 1.0), _partial(0.5, 1.5)]
    assignment = _assign_channels(partials, cfg)
    assert assignment[0] != assignment[1]


def test_percussion_channel_never_used_for_tonal(cfg):
    """Tonal partials never land on the percussion channel."""
    # More overlapping partials than channels: assigned channels stay in the tonal set.
    partials = [_partial(0.0, 1.0) for _ in range(20)]
    assignment = _assign_channels(partials, cfg)
    assert set(assignment.values()) <= set(cfg.tonal_channels)
    assert 9 not in assignment.values()
