"""Map unvoiced/transient energy (plosives, fricatives) onto GM percussion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from midivoder.analysis import Frame
from midivoder.config import EncodeConfig

# GM percussion key numbers (channel 10). All short, decaying sounds — long-ringing
# cymbals (e.g. crash 49) are avoided because their multi-second tail does not match the
# brief noise bursts of speech consonants.
KICK = 36
SNARE = 38
CLOSED_HAT = 42
OPEN_HAT = 46


@dataclass
class PercEvent:
    """A single percussion hit derived from an unvoiced transient frame."""

    time: float
    note: int  # GM drum note (used by the "gm" synth target)
    amp: float  # frame energy relative to the loudest frame (amplitude ratio)
    centroid: float  # spectral centroid (Hz); the "cr2" target centers its noise here


def _note_for_centroid(centroid: float) -> int:
    if centroid < 1000.0:
        return KICK  # low broadband bursts (voiced plosives b/d/g)
    if centroid < 2500.0:
        return SNARE  # mid bursts (t/k, affricates)
    if centroid < 4500.0:
        return CLOSED_HAT  # sibilant onsets (s)
    return OPEN_HAT  # very bright fricatives (sh, f)


def extract_percussion(frames: list[Frame], cfg: EncodeConfig) -> list[PercEvent]:
    """Detect unvoiced onsets in `frames` and map them to percussion events."""
    if not cfg.enable_percussion:
        return []

    energies = np.array([fr.energy for fr in frames], dtype=np.float64)
    peak_energy = float(energies.max()) if energies.size else 1e-9
    onset_lin = peak_energy * 10.0 ** (cfg.perc_onset_db / 20.0)
    fluxes = np.array([fr.flux for fr in frames], dtype=np.float64)
    flux_thresh = float(np.percentile(fluxes, 90)) if fluxes.size else 0.0
    min_interval = cfg.perc_min_interval_ms / 1000.0

    events: list[PercEvent] = []
    last_time = -1e9
    for i, fr in enumerate(frames):
        if fr.silent or fr.voiced or fr.energy < onset_lin:
            continue
        is_onset = fr.flux >= flux_thresh and (i == 0 or fr.flux >= fluxes[i - 1])
        if not (is_onset or fr.time - last_time >= min_interval):
            continue
        if fr.time - last_time < min_interval and not is_onset:
            continue
        events.append(
            PercEvent(
                time=fr.time,
                note=_note_for_centroid(fr.centroid),
                amp=fr.energy / peak_energy,
                centroid=fr.centroid,
            )
        )
        last_time = fr.time

    return events
