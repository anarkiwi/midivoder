"""Percussion onset extraction."""

from __future__ import annotations

from midivoder.analysis import Frame
from midivoder.config import EncodeConfig
from midivoder.midi import build_midi
from midivoder.percussion import extract_percussion


def _frames(second_silent: bool) -> list[Frame]:
    loud = Frame(index=0, time=0.0, energy=1.0, flux=1.0, centroid=3000.0)
    quiet = Frame(index=1, time=0.5, energy=0.1, flux=2.0, silent=second_silent)
    if not second_silent:
        quiet.centroid = 6000.0
    return [loud, quiet]


def test_silent_frames_are_not_percussion():
    """Frames gated as silent have no spectrum and must not become hits."""
    cfg = EncodeConfig()
    assert [ev.time for ev in extract_percussion(_frames(False), cfg)] == [0.0, 0.5]
    events = extract_percussion(_frames(True), cfg)
    assert [ev.time for ev in events] == [0.0]


def test_cr2_percussion_from_gated_frames_encodes():
    """A silence-gated onset candidate must not reach cr2 centroid-to-note mapping."""
    cfg = EncodeConfig(synth="cr2")
    mid = build_midi([], extract_percussion(_frames(True), cfg), cfg, 0)
    notes = [m.note for t in mid.tracks for m in t if m.type == "note_on"]
    assert len(notes) == 1
