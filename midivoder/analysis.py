"""Short-time spectral analysis: STFT, peak picking, voiced/unvoiced classification."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from midivoder.config import EncodeConfig

# Classification heuristics (speech-tuned).
_VOICED_FLATNESS_MAX = 0.45  # tonal frames have low spectral flatness
_VOICED_ZCR_MAX = 0.18  # voiced speech crosses zero slowly
_SILENCE_DB = -55.0  # frames quieter than this (rel. peak) are dropped


@dataclass
class Frame:
    """Per-hop spectral analysis result."""

    index: int
    time: float  # seconds, frame center
    peaks: list[tuple[float, float]] = field(
        default_factory=list
    )  # (hz, linear amp), loudest first
    energy: float = 0.0  # linear RMS energy
    voiced: bool = False
    centroid: float = 0.0  # spectral centroid (Hz)
    flatness: float = 1.0  # spectral flatness 0..1
    flux: float = 0.0  # positive spectral flux vs prev frame


def _next_pow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def analyze(signal: np.ndarray, cfg: EncodeConfig) -> list[Frame]:
    """Run STFT-based analysis and return one `Frame` per hop."""
    sr = cfg.sample_rate
    win_len = cfg.frame_samples
    hop = cfg.hop_samples
    nfft = _next_pow2(win_len)
    window = np.hanning(win_len)
    win_norm = window.sum() / 2.0  # for single-sided amplitude estimate

    if signal.size < win_len:
        signal = np.pad(signal, (0, win_len - signal.size))
    pad = win_len // 2
    padded = np.pad(signal, (pad, pad))

    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    band = (freqs >= cfg.fmin) & (freqs <= cfg.fmax)
    floor_lin = 10.0 ** (cfg.peak_floor_db / 20.0)

    n_frames = 1 + (len(padded) - win_len) // hop
    global_peak = max(float(np.max(np.abs(signal))), 1e-9)

    frames: list[Frame] = []
    prev_mag: np.ndarray | None = None
    for i in range(n_frames):
        start = i * hop
        seg = padded[start : start + win_len]
        wseg = seg * window
        spectrum = np.fft.rfft(wseg, n=nfft)
        mag = np.abs(spectrum) / win_norm

        rms = float(np.sqrt(np.mean(seg**2)))
        flux = 0.0
        if prev_mag is not None:
            flux = float(np.sum(np.maximum(mag - prev_mag, 0.0)))
        prev_mag = mag

        fr = Frame(
            index=i, time=(start + win_len / 2 - pad) / sr, energy=rms, flux=flux
        )

        if rms < global_peak * 10.0 ** (_SILENCE_DB / 20.0):
            frames.append(fr)
            continue

        power = mag[band] ** 2
        bfreqs = freqs[band]
        psum = float(power.sum()) + 1e-12
        fr.centroid = float(np.sum(bfreqs * power) / psum)
        log_power = np.log(power + 1e-12)
        fr.flatness = float(np.exp(np.mean(log_power)) / (psum / power.size))
        zcr = float(np.mean(np.abs(np.diff(np.sign(seg))) > 0))
        fr.voiced = fr.flatness < _VOICED_FLATNESS_MAX and zcr < _VOICED_ZCR_MAX

        fr.peaks = _pick_peaks(mag, freqs, band, floor_lin, cfg.max_peaks_per_frame)
        frames.append(fr)

    return frames


def _pick_peaks(
    mag: np.ndarray,
    freqs: np.ndarray,
    band: np.ndarray,
    floor_lin: float,
    max_peaks: int,
) -> list[tuple[float, float]]:
    """Local maxima with parabolic interpolation for sub-bin freq/amplitude."""
    df = freqs[1] - freqs[0]
    thresh = max(float(mag.max()) * floor_lin, 1e-9)
    lo = int(np.argmax(band))
    hi = len(band) - int(np.argmax(band[::-1]))
    lo = max(lo, 1)
    hi = min(hi, len(mag) - 1)

    peaks: list[tuple[float, float]] = []
    for k in range(lo, hi):
        m = mag[k]
        if m < thresh or m <= mag[k - 1] or m < mag[k + 1]:
            continue
        a = np.log(mag[k - 1] + 1e-12)
        b = np.log(m + 1e-12)
        c = np.log(mag[k + 1] + 1e-12)
        denom = a - 2 * b + c
        p = 0.5 * (a - c) / denom if denom != 0 else 0.0
        p = float(np.clip(p, -0.5, 0.5))
        freq = (k + p) * df
        amp = float(np.exp(b - 0.25 * (a - c) * p))
        peaks.append((freq, amp))

    peaks.sort(key=lambda fa: fa[1], reverse=True)
    return peaks[:max_peaks]
