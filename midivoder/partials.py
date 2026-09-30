"""Link per-frame spectral peaks into partials (McAulay-Quatieri style tracking)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from midivoder.analysis import Frame
from midivoder.config import EncodeConfig


@dataclass
class Partial:
    """A tracked sinusoidal partial: per-frame times, frequencies and amplitudes."""

    times: np.ndarray  # seconds
    freqs: np.ndarray  # Hz
    amps: np.ndarray  # linear amplitude
    start_frame: int
    end_frame: int

    @property
    def duration(self) -> float:
        """Partial length in seconds."""
        return float(self.times[-1] - self.times[0]) if self.times.size else 0.0

    @property
    def peak_amp(self) -> float:
        """Largest linear amplitude along the partial."""
        return float(self.amps.max()) if self.amps.size else 0.0


class _Track:
    __slots__ = (
        "freq",
        "amp",
        "gap",
        "times",
        "freqs",
        "amps",
        "start_frame",
        "last_frame",
    )

    def __init__(self, time: float, freq: float, amp: float, frame: int) -> None:
        self.freq = freq
        self.amp = amp
        self.gap = 0
        self.times = [time]
        self.freqs = [freq]
        self.amps = [amp]
        self.start_frame = frame
        self.last_frame = frame

    def extend(self, time: float, freq: float, amp: float, frame: int) -> None:
        """Continue the track with a matched peak and reset its gap counter."""
        self.freq = freq
        self.amp = amp
        self.gap = 0
        self.times.append(time)
        self.freqs.append(freq)
        self.amps.append(amp)
        self.last_frame = frame


def _cents(f: float, ref: float) -> float:
    return abs(1200.0 * np.log2(f / ref))


def track_partials(frames: list[Frame], cfg: EncodeConfig) -> list[Partial]:
    """Greedy nearest-frequency peak continuation with birth/death and a simultaneity cap."""
    active: list[_Track] = []
    finished: list[_Track] = []
    max_active = cfg.max_partials

    for fr in frames:
        peaks = fr.peaks
        used = [False] * len(peaks)

        # Louder tracks claim their continuation first.
        active.sort(key=lambda t: t.amp, reverse=True)
        for t in active:
            best_j = -1
            best_cents = cfg.match_cents
            for j, (f, _a) in enumerate(peaks):
                if used[j]:
                    continue
                c = _cents(f, t.freq)
                if c <= best_cents:
                    best_cents = c
                    best_j = j
            if best_j >= 0:
                f, a = peaks[best_j]
                used[best_j] = True
                t.extend(fr.time, f, a, fr.index)
            else:
                t.gap += 1

        survivors: list[_Track] = []
        for t in active:
            (finished if t.gap > cfg.max_gap_frames else survivors).append(t)
        active = survivors

        # Birth new tracks from the loudest unclaimed peaks — only on tonal frames,
        # so broadband noise is left for the percussion path.
        if fr.voiced and len(active) < max_active:
            unused = sorted(
                (peaks[j] for j in range(len(peaks)) if not used[j]),
                key=lambda fa: fa[1],
                reverse=True,
            )
            for f, a in unused:
                if len(active) >= max_active:
                    break
                active.append(_Track(fr.time, f, a, fr.index))

    finished.extend(active)

    min_dur = cfg.min_partial_ms / 1000.0
    partials: list[Partial] = []
    for t in finished:
        if len(t.times) < 2:
            continue
        times = np.asarray(t.times)
        if times[-1] - times[0] < min_dur:
            continue
        partials.append(
            Partial(
                times=times,
                freqs=np.asarray(t.freqs),
                amps=np.asarray(t.amps),
                start_frame=t.start_frame,
                end_frame=t.last_frame,
            )
        )
    partials.sort(key=lambda p: p.times[0])
    return partials
