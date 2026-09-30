"""End-to-end tests for the ``midivoder`` command-line interface."""

from __future__ import annotations

from collections import defaultdict

import mido
import numpy as np
import pytest
import soundfile as sf

from midivoder import instruments, render
from midivoder.cli import main
from midivoder.config import PERCUSSION_CHANNEL

_IN_SR = 22050

needs_fluidsynth = pytest.mark.skipif(
    not render.fluidsynth_available() or render.find_default_soundfont() is None,
    reason="fluidsynth/soundfont required",
)


@pytest.fixture
def voice_wav(tmp_path):
    """Harmonic tone, a silent gap, then a noise burst: exercises tonal and percussion paths."""
    t = np.arange(int(0.6 * _IN_SR)) / _IN_SR
    tone = sum(
        (0.5 / k) * np.sin(2 * np.pi * 220.0 * k * t) for k in (1, 2, 3)
    ) * np.hanning(t.size)
    gap = np.zeros(int(0.2 * _IN_SR))
    burst = 0.8 * np.random.default_rng(1).standard_normal(int(0.15 * _IN_SR))
    path = tmp_path / "voice.wav"
    sf.write(path, np.concatenate([tone, gap, burst]), _IN_SR)
    return str(path)


def _encode(voice_wav, tmp_path, *extra):
    out = str(tmp_path / "out.mid")
    assert main(["encode", voice_wav, "-o", out, *extra]) == 0
    return mido.MidiFile(out)


def _by_channel(mid):
    msgs = defaultdict(list)
    for track in mid.tracks:
        for msg in track:
            if not msg.is_meta:
                msgs[msg.channel].append(msg)
    return msgs


def _note_ons(msgs):
    return [m for m in msgs if m.type == "note_on" and m.velocity > 0]


def _ccs(msgs, control):
    return [
        m.value for m in msgs if m.type == "control_change" and m.control == control
    ]


def test_encode_gm_respects_timing_program_and_bend_flags(voice_wav, tmp_path, capsys):
    """GM output carries the requested PPQ, tempo, program and RPN bend range."""
    mid = _encode(
        voice_wav,
        tmp_path,
        "--instrument",
        "73",
        "--ppq",
        "96",
        "--bpm",
        "90",
        "--bend-range",
        "12",
    )
    assert mid.type == 1
    assert mid.ticks_per_beat == 96
    tempos = [m.tempo for m in mid.tracks[0] if m.type == "set_tempo"]
    assert tempos == [mido.bpm2tempo(90)]

    chans = _by_channel(mid)
    tonal = {ch: msgs for ch, msgs in chans.items() if ch != PERCUSSION_CHANNEL}
    assert tonal
    for msgs in tonal.values():
        assert {m.program for m in msgs if m.type == "program_change"} == {73}
        assert _ccs(msgs, 101) == [0] and _ccs(msgs, 100) == [0]
        assert _ccs(msgs, 6) == [12]
    notes = [m.note for msgs in tonal.values() for m in _note_ons(msgs)]
    assert 57 in notes  # 220 Hz fundamental
    assert _note_ons(chans[PERCUSSION_CHANNEL])

    out = capsys.readouterr().out
    assert "encoded" in out and "program: 73" in out and "synth: gm" in out


def test_encode_no_percussion_and_channel_cap(voice_wav, tmp_path, capsys):
    """--no-percussion makes all channels tonal; --max-channels bounds channel use."""
    mid = _encode(voice_wav, tmp_path, "--instrument", "0", "--no-percussion")
    assert "percussion_hits: 0" in capsys.readouterr().out
    for msgs in _by_channel(mid).values():
        assert [m.program for m in msgs if m.type == "program_change"] == [0]

    mid = _encode(
        voice_wav,
        tmp_path,
        "--instrument",
        "0",
        "--no-percussion",
        "--max-channels",
        "2",
    )
    chans = _by_channel(mid)
    assert chans and set(chans) <= {0, 1}
    assert len(mid.tracks) == 1 + len(chans)


def test_encode_cr2_dialect(voice_wav, tmp_path, capsys):
    """CR2 output uses CC-based setup, channels 1-8 plus 10, and CC7 amplitude."""
    mid = _encode(voice_wav, tmp_path, "--synth", "cr2", "--bend-range", "5")
    chans = _by_channel(mid)
    assert set(chans) <= set(range(8)) | {PERCUSSION_CHANNEL}
    assert PERCUSSION_CHANNEL in chans

    all_msgs = [m for msgs in chans.values() for m in msgs]
    assert not [m for m in all_msgs if m.type == "program_change"]
    assert not _ccs(all_msgs, 101)

    tonal = {ch: msgs for ch, msgs in chans.items() if ch != PERCUSSION_CHANNEL}
    assert tonal
    for msgs in tonal.values():
        assert _ccs(msgs, 121) == [0]
        assert _ccs(msgs, 31) == [5]
        assert {m.velocity for m in _note_ons(msgs)} == {127}
        assert len(_ccs(msgs, 7)) > 1

    perc = chans[PERCUSSION_CHANNEL]
    assert _ccs(perc, 73) == [0] and _ccs(perc, 24) == [0]
    assert _note_ons(perc)
    assert "program_name: cr2-oscillator" in capsys.readouterr().out


@needs_fluidsynth
def test_encode_auto_instrument_profiles_candidates(voice_wav, tmp_path):
    """Auto instrument selection picks one of the curated near-sinusoidal programs."""
    mid = _encode(voice_wav, tmp_path, "--instrument", "auto")
    programs = {
        m.program
        for msgs in _by_channel(mid).values()
        for m in msgs
        if m.type == "program_change"
    }
    assert len(programs) == 1
    assert programs <= set(instruments.CANDIDATE_PROGRAMS)


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["encode", "in.wav"],
        ["encode", "in.wav", "-o", "x.mid", "--synth", "sid"],
        ["encode", "in.wav", "-o", "x.mid", "--ppq", "many"],
        ["render", "in.mid"],
    ],
)
def test_bad_arguments_exit_with_usage_error(argv, capsys):
    """argparse rejects missing or invalid arguments with exit status 2."""
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    assert "usage:" in capsys.readouterr().err


@pytest.mark.parametrize(
    "flags,match",
    [
        (["--max-channels", "0"], "max_channels"),
        (["--ppq", "8"], "ppq"),
    ],
)
def test_invalid_config_values_rejected(voice_wav, tmp_path, flags, match):
    """Values argparse accepts but the config forbids raise ValueError before encoding."""
    out = tmp_path / "x.mid"
    with pytest.raises(ValueError, match=match):
        main(["encode", voice_wav, "-o", str(out), *flags])
    assert not out.exists()


@needs_fluidsynth
def test_render_roundtrip(voice_wav, tmp_path):
    """render turns an encoded .mid back into a non-silent WAV at the requested rate."""
    _encode(voice_wav, tmp_path, "--instrument", "73")
    wav = tmp_path / "back.wav"
    assert (
        main(
            [
                "render",
                str(tmp_path / "out.mid"),
                "-o",
                str(wav),
                "--sample-rate",
                "22050",
            ]
        )
        == 0
    )
    data, sr = sf.read(wav)
    assert sr == 22050
    assert np.max(np.abs(data)) > 1e-3


def test_render_without_fluidsynth(monkeypatch, tmp_path, capsys):
    """render reports a missing fluidsynth with exit status 2 and writes nothing."""
    monkeypatch.setattr(render, "fluidsynth_available", lambda: False)
    wav = tmp_path / "x.wav"
    assert main(["render", "in.mid", "-o", str(wav)]) == 2
    assert "fluidsynth not installed" in capsys.readouterr().err
    assert not wav.exists()


def test_profile_falls_back_without_soundfont(monkeypatch, capsys):
    """profile-instruments reports the default program when no soundfont is available."""
    monkeypatch.setattr(render, "find_default_soundfont", lambda: None)
    assert main(["profile-instruments"]) == 0
    out = capsys.readouterr().out
    assert f"default program: {instruments.DEFAULT_TONAL_PROGRAM}" in out
    assert instruments.program_name(instruments.DEFAULT_TONAL_PROGRAM) in out


@needs_fluidsynth
def test_profile_selects_candidate(capsys):
    """profile-instruments selects a curated program and names the soundfont used."""
    sf2 = render.find_default_soundfont()
    assert main(["profile-instruments", "--soundfont", sf2]) == 0
    out = capsys.readouterr().out
    prog = instruments.select_tonal_program(sf2)
    assert prog in instruments.CANDIDATE_PROGRAMS
    assert (
        f"selected tonal program: {prog} ({instruments.CANDIDATE_PROGRAMS[prog]})"
        in out
    )
    assert f"soundfont: {sf2}" in out
