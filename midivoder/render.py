"""FluidSynth wrapper: render a MIDI file back to audio (for the CLI and fidelity tests)."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import mido

_SOUNDFONT_CANDIDATES = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/default-GM.sf2",
    "/usr/share/sounds/sf2/TimGM6mb.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
    "/usr/share/soundfonts/default.sf2",
]


def fluidsynth_available() -> bool:
    """Return True if the fluidsynth executable is on PATH."""
    return shutil.which("fluidsynth") is not None


def find_default_soundfont() -> str | None:
    """Return $MIDIVODER_SOUNDFONT or the first installed GM soundfont, else None."""
    env = os.environ.get("MIDIVODER_SOUNDFONT")
    if env and os.path.exists(env):
        return env
    for path in _SOUNDFONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def render_midi(
    midi_path: str,
    wav_path: str,
    soundfont: str | None = None,
    sample_rate: int = 44100,
    gain: float = 1.0,
) -> str:
    """Render `midi_path` to `wav_path` with fluidsynth; return `wav_path`."""
    if not fluidsynth_available():
        raise RuntimeError("fluidsynth is not installed")
    soundfont = soundfont or find_default_soundfont()
    if soundfont is None:
        raise RuntimeError("no GM soundfont found; set MIDIVODER_SOUNDFONT")
    subprocess.run(
        # -R 0 -C 0 disable reverb/chorus so there is no long decay tail and the
        # rendered length tracks the MIDI's actual note events.
        [
            "fluidsynth",
            "-ni",
            "-R",
            "0",
            "-C",
            "0",
            "-g",
            str(gain),
            "-r",
            str(sample_rate),
            "-F",
            wav_path,
            soundfont,
            midi_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return wav_path


def render_to_array(
    midi_path: str, soundfont: str | None = None, sample_rate: int = 44100
) -> tuple[np.ndarray, int]:
    """Render `midi_path` and return the mono signal and its sample rate."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        render_midi(midi_path, wav_path, soundfont, sample_rate)
        data, sr = sf.read(wav_path, always_2d=True, dtype="float64")
        return data.mean(axis=1), sr
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def synth_probe_note(
    program: int, soundfont: str, note: int = 69, sample_rate: int = 22050
) -> tuple[np.ndarray, int]:
    """Render a single sustained note on `program` and return its steady-state segment."""
    mid = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    track.append(mido.Message("program_change", channel=0, program=program, time=0))
    track.append(mido.Message("note_on", channel=0, note=note, velocity=100, time=0))
    track.append(
        mido.Message("note_off", channel=0, note=note, velocity=0, time=480)
    )  # 0.5s
    track.append(mido.MetaMessage("end_of_track", time=0))

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        mid_path = tmp.name
    try:
        mid.save(mid_path)
        sig, sr = render_to_array(mid_path, soundfont, sample_rate)
    finally:
        if os.path.exists(mid_path):
            os.remove(mid_path)

    # Trim to the sustained middle, avoiding attack/release transients.
    if sig.size > sr // 2:
        a, b = int(sr * 0.15), int(sr * 0.4)
        sig = sig[a:b]
    return sig, sr
