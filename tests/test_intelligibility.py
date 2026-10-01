"""Ideal-sine resynthesis of midivoder MIDI follows each synth's pitch and level laws."""

from __future__ import annotations

import mido
import numpy as np
import pytest
import soundfile as sf

from midivoder import render as fluid
from midivoder.midi import cr2_hz_scale
from tools import intelligibility

SR = 22050


def _write(path, channel_msgs, ppq=480):
    mid = mido.MidiFile(type=1, ticks_per_beat=ppq)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    for msg in channel_msgs:
        track.append(msg)
    mid.save(path)


def _cc(control, value, time=0, ch=0):
    return mido.Message(
        "control_change", channel=ch, control=control, value=value, time=time
    )


def _note(kind, note, vel=0, time=0, ch=0):
    return mido.Message(kind, channel=ch, note=note, velocity=vel, time=time)


def _peak(y, a, b):
    seg = y[int(a * SR) : int(b * SR)]
    spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size), 1 << 20))
    hz = np.fft.rfftfreq(1 << 20, 1.0 / SR)
    i = int(np.argmax(spec))
    return hz[i], 2.0 * spec[i] / np.hanning(seg.size).sum()


def test_gm_note_level_and_pitch(tmp_path):
    """GM: amplitude (v/127)^2 (CC7/127)^2, pitch-bend linear in semitones via RPN 0."""
    path = str(tmp_path / "gm.mid")
    msgs = [_cc(101, 0), _cc(100, 0), _cc(6, 12), _cc(7, 100)]
    msgs += [mido.Message("pitchwheel", channel=0, pitch=-4096)]
    msgs += [_note("note_on", 69, 64), _note("note_off", 69, time=960)]
    _write(path, msgs + [_cc(7, 127, time=480)])
    y = intelligibility.render(path, "gm", SR)
    hz, amp = _peak(y, 0.1, 0.9)
    assert hz == pytest.approx(440.0 * 2 ** (-6 / 12), rel=1e-3)
    assert amp == pytest.approx((64 / 127) ** 2 * (100 / 127) ** 2, rel=0.01)
    assert np.abs(y[int(1.05 * SR) :]).max() == 0.0


def test_cr2_note_level_pitch_and_ceiling(tmp_path):
    """CR2: amplitude v/127 * CC7/127 * hzScale(note), period-linear bend, ceiling silent."""
    path = str(tmp_path / "cr2.mid")
    msgs = [
        _cc(31, 12),
        _cc(7, 100),
        mido.Message("pitchwheel", channel=0, pitch=-4096),
    ]
    msgs += [_note("note_on", 81, 64), _note("note_on", 96, 127)]
    msgs += [_note("note_off", 81, time=960), _note("note_off", 96)]
    _write(path, msgs)
    y = intelligibility.render(path, "cr2", SR)
    hz, amp = _peak(y, 0.1, 0.9)
    assert hz == pytest.approx(880.0 / 1.5, rel=1e-3)
    assert amp == pytest.approx(64 / 127 * 100 / 127 * cr2_hz_scale(81), rel=0.01)


def test_cr2_restrike_before_expiry_keeps_velocity(tmp_path):
    """CR2 keeps a note's velocity when it is re-struck before the note expired."""
    path = str(tmp_path / "cr2.mid")
    msgs = [_cc(72, 0), _note("note_on", 69, 127)]
    msgs += [_note("note_off", 69, time=480), _note("note_on", 69, 32)]
    msgs += [_note("note_off", 69, time=480), _note("note_on", 69, 32, time=48)]
    msgs += [_note("note_off", 69, time=480)]
    _write(path, msgs)
    y = intelligibility.render(path, "cr2", SR)
    first, kept, fresh = (_peak(y, a, a + 0.4)[1] for a in (0.05, 0.55, 1.1))
    assert kept == pytest.approx(first, rel=0.01)
    assert 20 * np.log10(fresh / first) == pytest.approx(
        20 * np.log10(32 / 127), abs=0.2
    )


def test_render_cli_writes_wav(tmp_path):
    """The `render` subcommand writes a normalised WAV of the MIDI's tonal notes."""
    mid, wav = str(tmp_path / "a.mid"), str(tmp_path / "a.wav")
    _write(mid, [_note("note_on", 60, 100), _note("note_off", 60, time=480)])
    assert intelligibility.main(["render", mid, wav, "--rate", str(SR)]) == 0
    y, sr = sf.read(wav)
    assert sr == SR and np.abs(y).max() == pytest.approx(0.9, abs=1e-3)


needs_fluidsynth = pytest.mark.skipif(
    not fluid.fluidsynth_available() or fluid.find_default_soundfont() is None,
    reason="fluidsynth/soundfont required",
)


@needs_fluidsynth
def test_gm_cc7_law_matches_fluidsynth(tmp_path):
    """FluidSynth attenuates CC7 like velocity: CC7=64 renders (64/127)^2 of CC7=127."""

    def level(volume):
        path = str(tmp_path / f"cc7_{volume}.mid")
        msgs = [mido.Message("program_change", channel=0, program=79), _cc(7, volume)]
        _write(
            path, msgs + [_note("note_on", 69, 127), _note("note_off", 69, time=480)]
        )
        sig, sr = fluid.render_to_array(path)
        return float(np.sqrt(np.mean(sig[int(0.1 * sr) : int(0.4 * sr)] ** 2)))

    expect = 20 * np.log10(intelligibility.rendered_level(127, 64, 69, "gm"))
    assert 20 * np.log10(level(64) / level(127)) == pytest.approx(expect, abs=0.5)
