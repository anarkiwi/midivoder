"""End-to-end orchestration: voice WAV -> General MIDI file."""

from __future__ import annotations

from midivoder.analysis import analyze
from midivoder.audio import load_wav
from midivoder.config import EncodeConfig
from midivoder.instruments import program_name, resolve_program
from midivoder.midi import build_midi
from midivoder.partials import track_partials
from midivoder.percussion import extract_percussion


def encode_wav_to_midi(
    in_path: str,
    out_path: str,
    cfg: EncodeConfig | None = None,
    soundfont: str | None = None,
) -> dict:
    """Encode `in_path` into a multi-channel SMF at `out_path`; return summary stats."""
    cfg = cfg or EncodeConfig()
    signal = load_wav(in_path, cfg.sample_rate)
    frames = analyze(signal, cfg)
    partials = track_partials(frames, cfg)
    perc = extract_percussion(frames, cfg)
    # CR2 has no GM program map (its tonal voice is a coil-pulse oscillator), so skip the
    # fluidsynth-based profiling entirely; the program number is never emitted in that mode.
    program = 0 if cfg.synth == "cr2" else resolve_program(cfg.instrument, soundfont)
    mid = build_midi(partials, perc, cfg, program)
    mid.save(out_path)

    return {
        "duration_s": round(len(signal) / cfg.sample_rate, 3),
        "synth": cfg.synth,
        "partials": len(partials),
        "percussion_hits": len(perc),
        "tonal_channels": len(cfg.tonal_channels),
        "program": program,
        "program_name": (
            program_name(program) if cfg.synth == "gm" else "cr2-oscillator"
        ),
        "tracks": len(mid.tracks),
    }
