"""Tests for soundfont discovery, render guards and instrument profiling helpers."""

from __future__ import annotations

import numpy as np
import pytest

from midivoder import instruments, render
from midivoder.instruments import _sinusoidality


def test_env_soundfont_takes_precedence(monkeypatch, tmp_path):
    """An existing $MIDIVODER_SOUNDFONT wins over the built-in candidate list."""
    sf2 = tmp_path / "custom.sf2"
    sf2.write_bytes(b"")
    monkeypatch.setenv("MIDIVODER_SOUNDFONT", str(sf2))
    assert render.find_default_soundfont() == str(sf2)


def test_missing_soundfont_everywhere_returns_none(monkeypatch, tmp_path):
    """A dangling env path and no installed candidates yield None."""
    monkeypatch.setenv("MIDIVODER_SOUNDFONT", str(tmp_path / "absent.sf2"))
    monkeypatch.setattr(render, "_SOUNDFONT_CANDIDATES", [str(tmp_path / "nope.sf2")])
    assert render.find_default_soundfont() is None


def test_render_requires_fluidsynth(monkeypatch, tmp_path):
    """render_midi refuses to run without the fluidsynth binary."""
    monkeypatch.setattr(render, "fluidsynth_available", lambda: False)
    with pytest.raises(RuntimeError, match="fluidsynth is not installed"):
        render.render_midi("in.mid", str(tmp_path / "o.wav"))


def test_render_requires_soundfont(monkeypatch, tmp_path):
    """render_midi refuses to run when no soundfont can be located."""
    monkeypatch.setattr(render, "fluidsynth_available", lambda: True)
    monkeypatch.setattr(render, "find_default_soundfont", lambda: None)
    with pytest.raises(RuntimeError, match="no GM soundfont"):
        render.render_midi("in.mid", str(tmp_path / "o.wav"))


def test_sinusoidality_orders_tone_above_noise(sine, noise_burst):
    """A pure tone scores near 1, noise scores far lower, too-short input scores 0."""
    tone = _sinusoidality(sine(440.0, dur=0.25, sr=22050))
    noise = _sinusoidality(noise_burst(dur=0.25, sr=22050))
    assert tone > 0.9
    assert noise < 0.1
    assert _sinusoidality(np.ones(10)) == 0.0


def test_program_name_falls_back_to_number():
    """Programs outside the curated table get a generic name."""
    assert instruments.program_name(73) == "Flute"
    assert instruments.program_name(0) == "GM program 0"


def test_select_program_skips_failed_probes(monkeypatch):
    """Probe failures are skipped and the best remaining candidate is chosen."""
    instruments.select_tonal_program.cache_clear()
    sr = 22050
    t = np.arange(4096) / sr
    rng = np.random.default_rng(0)

    def probe(prog, _soundfont):
        if prog == instruments.DEFAULT_TONAL_PROGRAM:
            raise RuntimeError("probe failed")
        if prog == 71:
            return np.sin(2 * np.pi * 440.0 * t), sr
        return rng.standard_normal(t.size), sr

    monkeypatch.setattr(render, "fluidsynth_available", lambda: True)
    monkeypatch.setattr(render, "synth_probe_note", probe)
    try:
        assert instruments.select_tonal_program("fake.sf2") == 71
    finally:
        instruments.select_tonal_program.cache_clear()


def test_select_program_default_without_fluidsynth(monkeypatch):
    """Without fluidsynth the curated default program is returned."""
    instruments.select_tonal_program.cache_clear()
    monkeypatch.setattr(render, "fluidsynth_available", lambda: False)
    try:
        assert (
            instruments.select_tonal_program("fake.sf2")
            == instruments.DEFAULT_TONAL_PROGRAM
        )
    finally:
        instruments.select_tonal_program.cache_clear()
