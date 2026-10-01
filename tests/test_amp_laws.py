"""Per-synth amplitude laws and the CR2 note ceiling, checked against each synth's model."""

from __future__ import annotations

import numpy as np
import pytest

from midivoder import render
from midivoder.analysis import analyze
from midivoder.config import CR2_MAX_PITCH, PERCUSSION_CHANNEL, EncodeConfig
from midivoder.midi import build_midi, cr2_hz_scale, midi_float
from midivoder.partials import Partial, track_partials
from midivoder.percussion import PercEvent, extract_percussion

needs_fluidsynth = pytest.mark.skipif(
    not render.fluidsynth_available() or render.find_default_soundfont() is None,
    reason="fluidsynth/soundfont required",
)


def _partial(t0: float, t1: float, f0: float, f1: float, amp: float) -> Partial:
    times = np.linspace(t0, t1, 50)
    return Partial(
        times=times,
        freqs=np.geomspace(f0, f1, times.size),
        amps=np.full(times.size, amp),
        start_frame=0,
        end_frame=times.size - 1,
    )


def _tracks(mid):
    """Per-channel list of (absolute tick, message), in file order."""
    out: dict[int, list] = {}
    for track in mid.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if not msg.is_meta:
                out.setdefault(msg.channel, []).append((tick, msg))
    return out


@needs_fluidsynth
def test_gm_velocity_renders_amplitude_ratio(tmp_path):
    """A -20 dB partial renders 20 dB below the reference through FluidSynth."""
    cfg = EncodeConfig()
    partials = [
        _partial(0.0, 0.6, 440.0, 440.0, 1.0),
        _partial(0.8, 1.4, 440.0, 440.0, 0.1),
    ]
    path = str(tmp_path / "gm.mid")
    build_midi(partials, [], cfg, 73).save(path)
    sig, sr = render.render_to_array(path)

    def rms(a: float, b: float) -> float:
        return float(np.sqrt(np.mean(sig[int(a * sr) : int(b * sr)] ** 2)))

    diff_db = 20.0 * np.log10(rms(1.0, 1.3) / rms(0.2, 0.5))
    assert diff_db == pytest.approx(-20.0, abs=2.0)


def test_cr2_cc7_recovers_amplitude_with_hz_scale():
    """CC7/127 * hzScale(base note) equals the amplitude ratio, including on retriggers."""
    cfg = EncodeConfig(synth="cr2", enable_percussion=False)
    cases = [
        (220.0, 220.0, 1.0),
        (880.0, 880.0, 0.3),
        (1500.0, 1500.0, 0.05),
        (440.0, 1320.0, 0.2),  # glide forces retriggers onto new base notes
        (2000.0, 2000.0, 0.5),  # beyond CC7's reach: saturates at 127
    ]
    partials = [_partial(0.1, 1.0, f0, f1, a) for f0, f1, a in cases]
    tracks = _tracks(build_midi(partials, [], cfg, 0))
    for ch, (_, _, amp) in enumerate(cases):
        msgs = tracks[ch]
        checked, note = 0, None
        for tick, msg in msgs:
            if msg.type == "note_on" and msg.velocity:
                note = msg.note
                assert msg.velocity == 127
            if msg.type != "control_change" or msg.control != 7:
                continue
            on_now = [m.note for t, m in msgs if t == tick and m.type == "note_on"]
            base = on_now[0] if on_now else note
            if base is None:
                continue  # channel setup
            scale = cr2_hz_scale(base)
            if amp > scale:
                assert msg.value == 127
            else:
                assert msg.value / 127.0 * scale == pytest.approx(amp, abs=0.5 / 127)
            checked += 1
        assert checked
    assert len({m.note for _, m in tracks[3] if m.type == "note_on"}) > 1


def test_cr2_hz_scale_matches_firmware_formula():
    """hzScale = 1 - hz(note)/hz(96), clipped at zero above the ceiling."""
    hz96 = 440.0 * 2.0 ** ((CR2_MAX_PITCH - 69) / 12)
    for note in (24, 60, 81, 95):
        hz = 440.0 * 2.0 ** ((note - 69) / 12)
        assert cr2_hz_scale(note) == pytest.approx(1.0 - hz / hz96)
    assert cr2_hz_scale(CR2_MAX_PITCH) == 0.0
    assert cr2_hz_scale(CR2_MAX_PITCH + 5) == 0.0


def _bright_signal(sr: int = 16000) -> np.ndarray:
    t = np.arange(int(0.5 * sr)) / sr
    tones = 0.5 * np.sin(2 * np.pi * 300.0 * t) + 0.5 * np.sin(2 * np.pi * 2500.0 * t)
    rng = np.random.default_rng(0)
    hiss = np.diff(rng.standard_normal(int(0.2 * sr) + 2), n=2)
    hiss *= 0.8 / np.max(np.abs(hiss))
    return np.concatenate([tones, np.zeros(sr // 5), hiss])


def _note_ons(cfg: EncodeConfig) -> dict[int, list[int]]:
    frames = analyze(_bright_signal(), cfg)
    perc = extract_percussion(frames, cfg)
    assert perc and max(ev.centroid for ev in perc) > 4000.0
    mid = build_midi(track_partials(frames, cfg), perc, cfg, 0)
    return {
        ch: [m.note for _, m in msgs if m.type == "note_on" and m.velocity]
        for ch, msgs in _tracks(mid).items()
    }


def test_cr2_notes_never_exceed_ceiling():
    """In cr2 mode no channel, percussion included, receives an unsoundable note."""
    cfg = EncodeConfig(synth="cr2")
    assert cfg.note_ceiling == CR2_MAX_PITCH - 1 and cr2_hz_scale(cfg.note_ceiling) > 0
    notes = _note_ons(cfg)
    assert notes[PERCUSSION_CHANNEL]
    assert max(n for ns in notes.values() for n in ns) <= cfg.note_ceiling
    tonal = [n for ch, ns in notes.items() if ch != PERCUSSION_CHANNEL for n in ns]
    assert tonal and round(midi_float(2500.0)) not in tonal


def test_gm_notes_may_exceed_cr2_ceiling():
    """GM keeps the full note range: the 2500 Hz tone lands above note 96."""
    notes = _note_ons(EncodeConfig())
    tonal = [n for ch, ns in notes.items() if ch != PERCUSSION_CHANNEL for n in ns]
    assert round(midi_float(2500.0)) in tonal


def test_cr2_peaks_stop_below_unplayable_base_note():
    """cr2 analysis yields no peak whose nearest note exceeds the ceiling."""
    ceiling = EncodeConfig(synth="cr2").note_ceiling
    for synth, expect_high in (("cr2", False), ("gm", True)):
        frames = analyze(_bright_signal(), EncodeConfig(synth=synth))
        top = max(f for fr in frames for f, _ in fr.peaks)
        assert (round(midi_float(top)) > ceiling) is expect_high


@pytest.mark.parametrize(
    "synth,velocities", [("gm", [127, 40, 13]), ("cr2", [127, 13, 1])]
)
def test_percussion_velocity_law(synth, velocities):
    """Drum velocity follows the synth's amplitude law: sqrt for GM, linear for CR2."""
    cfg = EncodeConfig(synth=synth)
    events = [
        PercEvent(time=0.1 * i, note=38, amp=amp, centroid=6000.0)
        for i, amp in enumerate((1.0, 0.1, 0.01))
    ]
    msgs = _tracks(build_midi([], events, cfg, 0))[PERCUSSION_CHANNEL]
    ons = [m for _, m in msgs if m.type == "note_on" and m.velocity]
    assert [m.velocity for m in ons] == velocities
    assert {m.note for m in ons} == ({cfg.note_ceiling} if synth == "cr2" else {38})
