"""General MIDI instrument table and automatic tonal-program selection.

Each tonal MIDI channel renders roughly one sinusoidal partial, so the ideal patch is as
close to a pure tone as possible. ``select_tonal_program`` profiles candidate GM programs by
rendering a probe note (when fluidsynth + a soundfont are available) and picking the most
sinusoidal; otherwise it falls back to a curated default.
"""

from __future__ import annotations

import functools
import subprocess

import numpy as np

from midivoder import render

# Curated near-sinusoidal GM programs (0-based program numbers), best-first fallback order.
CANDIDATE_PROGRAMS: dict[int, str] = {
    79: "Ocarina",
    73: "Flute",
    74: "Recorder",
    75: "Pan Flute",
    72: "Piccolo",
    71: "Clarinet",
    68: "Oboe",
    54: "Voice Oohs",
}

DEFAULT_TONAL_PROGRAM = 79  # Ocarina


def program_name(program: int) -> str:
    """Human-readable name for a GM `program` number."""
    return CANDIDATE_PROGRAMS.get(program, f"GM program {program}")


def _sinusoidality(signal: np.ndarray) -> float:
    """Fraction of spectral energy carried by the strongest partial — higher = purer tone."""
    if signal.size < 256:
        return 0.0
    n = int(2 ** np.ceil(np.log2(signal.size)))
    mag = np.abs(np.fft.rfft(signal * np.hanning(signal.size), n=n))
    power = mag**2
    total = float(power.sum()) + 1e-12
    k = int(np.argmax(power))
    # Energy within a small neighborhood of the dominant bin.
    lo, hi = max(0, k - 2), min(power.size, k + 3)
    return float(power[lo:hi].sum() / total)


@functools.lru_cache(maxsize=1)
def select_tonal_program(soundfont: str | None = None) -> int:
    """Pick the most sinusoidal candidate GM program, profiling via fluidsynth if possible."""
    if soundfont is None:
        soundfont = render.find_default_soundfont()
    if soundfont is None or not render.fluidsynth_available():
        return DEFAULT_TONAL_PROGRAM

    best_prog, best_score = DEFAULT_TONAL_PROGRAM, -1.0
    for prog in CANDIDATE_PROGRAMS:
        try:
            sig, _ = render.synth_probe_note(prog, soundfont)
        except (subprocess.CalledProcessError, RuntimeError, OSError):
            continue
        score = _sinusoidality(sig)
        if score > best_score:
            best_prog, best_score = prog, score
    return best_prog


def resolve_program(instrument: str, soundfont: str | None = None) -> int:
    """Resolve the config ``instrument`` field ("auto" or a GM number) to a program."""
    if instrument == "auto":
        return select_tonal_program(soundfont)
    return int(instrument)
